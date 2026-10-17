"""Minimal ComfyUI HTTP/WebSocket client: validate, submit, wait, record timing events."""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable

import httpx

Workflow = dict[str, dict[str, Any]]


class ComfyError(RuntimeError):
    """kind: rejected | oom | timeout | crash | error"""

    def __init__(self, message: str, kind: str = "error", details: Any = None):
        super().__init__(message)
        self.kind = kind
        self.details = details


# ------------------------------------------------------------------ validation

def is_link(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int))


def _combo_options(entry: Any) -> list | None:
    """Options of a combo input from /object_info, or None if not a combo / unknown format.

    Legacy nodes report ``[[opt, ...], {...}]``; V3-schema nodes report ``["COMBO", {"options": [...]}]``.
    """
    if not isinstance(entry, (list, tuple)) or not entry:
        return None
    head = entry[0]
    if isinstance(head, list):
        return head
    if head == "COMBO" and len(entry) > 1 and isinstance(entry[1], dict):
        options = entry[1].get("options")
        return list(options) if options else None
    return None


def validate_workflow(workflow: Workflow, object_info: dict) -> tuple[list[str], list[str]]:
    """Check an API-format workflow against the live server's node schemas.

    Returns (errors, warnings). Errors are things ComfyUI would reject (unknown node type,
    missing required input, a model file that isn't installed, a dangling link).
    """
    errors: list[str] = []
    warnings: list[str] = []
    for node_id, node in workflow.items():
        ctype = node.get("class_type")
        info = object_info.get(ctype)
        if info is None:
            errors.append(f"node {node_id}: unknown node type {ctype!r} "
                          f"(custom node missing, or renamed in this ComfyUI version)")
            continue
        spec = info.get("input") or {}
        required = spec.get("required") or {}
        optional = spec.get("optional") or {}
        inputs = node.get("inputs") or {}
        for name in required:
            if name not in inputs:
                errors.append(f"node {node_id} ({ctype}): missing required input {name!r}")
        for name, value in inputs.items():
            entry = required.get(name, optional.get(name))
            if entry is None:
                warnings.append(f"node {node_id} ({ctype}): input {name!r} is not in the schema (ignored)")
                continue
            if is_link(value):
                if value[0] not in workflow:
                    errors.append(f"node {node_id} ({ctype}): input {name!r} links to missing node {value[0]!r}")
                continue
            options = _combo_options(entry)
            if options is not None and value not in options:
                shown = ", ".join(map(str, options[:8])) or "(none installed)"
                errors.append(f"node {node_id} ({ctype}): {name}={value!r} is not available. "
                              f"Available: {shown}")
    return errors, warnings


# ---------------------------------------------------------------- error parsing

OOM_MARKERS = ("out of memory", "outofmemory", "allocation on device", "cudaerrormemoryallocation")


def _execution_error(status: dict) -> dict:
    for item in status.get("messages") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "execution_error":
            return item[1] or {}
    return {}


def classify_status(status: dict) -> tuple[str, str]:
    """(kind, human message) for a failed history status block."""
    err = _execution_error(status)
    if not err:
        interrupted = any(isinstance(m, (list, tuple)) and m and m[0] == "execution_interrupted"
                          for m in status.get("messages") or [])
        return ("timeout" if interrupted else "error",
                "execution interrupted" if interrupted else f"execution failed: {status}")
    text = f"{err.get('exception_type', '')}: {err.get('exception_message', '')}".strip()
    kind = "oom" if any(m in text.lower() for m in OOM_MARKERS) else "error"
    return kind, f"{err.get('node_type', '?')} (node {err.get('node_id', '?')}) failed: {text}"


# ------------------------------------------------------------- timing recorder

class EventRecorder(threading.Thread):
    """Records ComfyUI WebSocket events (executing / progress / ...) with monotonic timestamps.

    Used only for the timing breakdown; generation does not depend on it.
    """

    def __init__(self, ws_url: str):
        super().__init__(daemon=True)
        self.ws_url = ws_url
        self.events: list[tuple[float, str, dict]] = []
        self.error: str | None = None
        self._halt = threading.Event()
        self._connected = threading.Event()

    def run(self) -> None:
        from websockets.sync.client import connect
        try:
            # ping_interval=None: ComfyUI can be too busy to answer keepalive pings mid-generation
            with connect(self.ws_url, open_timeout=10, max_size=None, ping_interval=None) as ws:
                self._connected.set()
                while not self._halt.is_set():
                    try:
                        msg = ws.recv(timeout=1.0)
                    except TimeoutError:
                        continue
                    if isinstance(msg, bytes):
                        continue  # binary latent previews
                    try:
                        data = json.loads(msg)
                    except ValueError:
                        continue
                    self.events.append((time.monotonic(), data.get("type", ""), data.get("data") or {}))
        except Exception as exc:  # connection refused/closed, server died
            if not self._halt.is_set():
                self.error = repr(exc)
        finally:
            self._connected.set()

    def wait_connected(self, timeout: float = 10.0) -> bool:
        return self._connected.wait(timeout)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=5)


# ----------------------------------------------------------------------- client

class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.client_id = uuid.uuid4().hex
        self._object_info: dict | None = None

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "ComfyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def system_stats(self) -> dict:
        r = self.http.get("/system_stats")
        r.raise_for_status()
        return r.json()

    def object_info(self, refresh: bool = False) -> dict:
        if self._object_info is None or refresh:
            r = self.http.get("/object_info", timeout=180)
            r.raise_for_status()
            self._object_info = r.json()
        return self._object_info

    def validate(self, workflow: Workflow) -> tuple[list[str], list[str]]:
        return validate_workflow(workflow, self.object_info())

    def recorder(self) -> EventRecorder:
        ws_base = "ws" + self.base_url[len("http"):]
        return EventRecorder(f"{ws_base}/ws?clientId={self.client_id}")

    def submit(self, workflow: Workflow) -> str:
        r = self.http.post("/prompt", json={"prompt": workflow, "client_id": self.client_id})
        if r.status_code != 200:
            try:
                body: Any = r.json()
            except ValueError:
                body = r.text
            raise ComfyError(f"ComfyUI rejected the workflow: {json.dumps(body)[:2000]}",
                             kind="rejected", details=body)
        return r.json()["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        r = self.http.get(f"/history/{prompt_id}")
        r.raise_for_status()
        return r.json().get(prompt_id)

    def interrupt(self) -> None:
        try:
            self.http.post("/interrupt", timeout=10)
        except httpx.HTTPError:
            pass

    def free(self) -> None:
        try:
            self.http.post("/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        except httpx.HTTPError:
            pass

    def wait(self, prompt_id: str, timeout_s: float,
             alive: Callable[[], bool] | None = None, poll_s: float = 2.0) -> dict:
        """Block until the prompt finishes; return its history entry or raise ComfyError."""
        deadline = time.monotonic() + timeout_s
        while True:
            if alive is not None and not alive():
                raise ComfyError("ComfyUI process exited during generation "
                                 "(commonly system-RAM exhaustion or a driver crash)", kind="crash")
            entry = None
            try:
                entry = self.history(prompt_id)
            except httpx.HTTPError:
                pass  # transient while the server is busy; liveness is checked above
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    kind, message = classify_status(status)
                    raise ComfyError(message, kind=kind, details=status)
                return entry
            if time.monotonic() > deadline:
                self.interrupt()
                raise ComfyError(f"timed out after {timeout_s / 60:.0f} min", kind="timeout")
            time.sleep(poll_s)

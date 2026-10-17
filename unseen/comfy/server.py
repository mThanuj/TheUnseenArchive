"""Start and stop a private, headless ComfyUI process.

The app never shares a ComfyUI instance with the user's interactive sessions: each batch of
generation jobs starts its own process with a chosen memory mode and stops it afterwards, so
VRAM and system RAM are fully released for the other pipeline stages.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import httpx

from ..config import RuntimeConfig


class ComfyStartupError(RuntimeError):
    pass


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class ComfyServer:
    def __init__(self, rt: RuntimeConfig, memory_mode: str, log_path: Path,
                 extra_args: list[str] | None = None):
        if memory_mode not in rt.comfyui.memory_modes:
            raise KeyError(f"unknown memory mode {memory_mode!r}; "
                           f"choose from {sorted(rt.comfyui.memory_modes)}")
        self.rt = rt
        self.memory_mode = memory_mode
        self.log_path = log_path
        self.extra_args = list(extra_args or [])
        self.base_url = f"http://{rt.comfyui.host}:{rt.comfyui.port}"
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        self._log_start = 0

    # ------------------------------------------------------------------ helpers
    def work_dir(self, name: str) -> Path:
        return self.rt.path("comfy_work") / name

    def command(self) -> list[str]:
        rt = self.rt
        return [
            str(rt.comfy_python()), "main.py",
            "--listen", rt.comfyui.host,
            "--port", str(rt.comfyui.port),
            "--models-directory", str(rt.path("models_dir")),
            "--output-directory", str(self.work_dir("output")),
            "--temp-directory", str(self.work_dir("temp")),
            "--input-directory", str(self.work_dir("input")),
            "--user-directory", str(self.work_dir("user")),
            *rt.comfyui.base_args,
            *rt.comfyui.memory_modes[self.memory_mode],
            *self.extra_args,
        ]

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, lines: int = 60) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._log_start)
                return "".join(fh.readlines()[-lines:])
        except OSError:
            return ""

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "ComfyServer":
        rt = self.rt
        comfy_dir = rt.path("comfyui_dir")
        if not (comfy_dir / "main.py").is_file():
            raise ComfyStartupError(
                f"ComfyUI not found at {comfy_dir}. Run scripts\\windows\\setup.ps1 first.")
        if not rt.comfy_python().is_file():
            raise ComfyStartupError(
                f"ComfyUI venv not found at {rt.comfy_python()}. Run scripts\\windows\\setup.ps1 first.")
        if port_in_use(rt.comfyui.host, rt.comfyui.port):
            raise ComfyStartupError(
                f"port {rt.comfyui.port} is already in use. Close any other ComfyUI instance "
                f"or change comfyui.port in config/runtime.yaml.")

        for name in ("output", "temp", "input", "user"):
            self.work_dir(name).mkdir(parents=True, exist_ok=True)
        rt.path("models_dir").mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self.command()
        self._log_fh = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log_fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start "
                           f"memory_mode={self.memory_mode}\n$ {' '.join(cmd)}\n")
        self._log_fh.flush()
        self._log_start = self._log_fh.tell()

        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, cwd=comfy_dir, stdout=self._log_fh,
                                     stderr=subprocess.STDOUT, env=env, creationflags=flags)

        deadline = time.monotonic() + rt.comfyui.startup_timeout_s
        while time.monotonic() < deadline:
            if not self.alive():
                code = self.proc.returncode
                tail = self.log_tail()
                self.stop()
                raise ComfyStartupError(
                    f"ComfyUI exited during startup (exit code {code}). Last log lines:\n{tail}")
            try:
                if httpx.get(self.base_url + "/system_stats", timeout=3).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(1.0)

        tail = self.log_tail()
        self.stop()
        raise ComfyStartupError(
            f"ComfyUI did not become ready within {rt.comfyui.startup_timeout_s}s. "
            f"Last log lines:\n{tail}")

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    httpx.post(self.base_url + "/free",
                               json={"unload_models": True, "free_memory": True}, timeout=10)
                except httpx.HTTPError:
                    pass
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=30)
            self.proc = None
        if self._log_fh is not None:
            self._log_fh.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} stopped\n")
            self._log_fh.close()
            self._log_fh = None
        for _ in range(40):  # let the OS release the port before a restart
            if not port_in_use(self.rt.comfyui.host, self.rt.comfyui.port):
                break
            time.sleep(0.5)

    def __enter__(self) -> "ComfyServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

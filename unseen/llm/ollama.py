"""Local LLM via Ollama's HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md).

Uses `format` with a JSON schema for structured output, `keep_alive` to unload the model when the
story stage is done (so video/audio stages get the RAM back), and `/api/pull` for setup.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx

from .base import LLMError, schema_guide

Log = Callable[[str], None]


def _same_model(a: str, b: str) -> bool:
    strip = lambda s: s[:-len(":latest")] if s.endswith(":latest") else s  # noqa: E731
    return strip(a).lower() == strip(b).lower()


class OllamaProvider:
    name = "ollama"

    def __init__(self, url: str, model: str, num_ctx: int = 8192, timeout_s: float = 2400,
                 think: bool | str = False):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.think = think

    def version(self) -> str:
        try:
            r = httpx.get(f"{self.url}/api/version", timeout=5)
            r.raise_for_status()
            return r.json().get("version", "?")
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama is not reachable at {self.url} ({exc!r}). Install it with "
                           f"`winget install Ollama.Ollama` and make sure it is running.") from None

    def has_model(self) -> bool:
        r = httpx.get(f"{self.url}/api/tags", timeout=30)
        r.raise_for_status()
        return any(_same_model(m.get("name", ""), self.model) for m in r.json().get("models", []))

    def pull(self, log: Log = print) -> None:
        last: dict[str, float] = {}   # per layer digest
        with httpx.stream("POST", f"{self.url}/api/pull", json={"model": self.model, "stream": True},
                          timeout=None) as r:
            if r.status_code != 200:
                raise LLMError(f"ollama pull failed ({r.status_code}): {r.read()[:500]!r}")
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise LLMError(f"ollama pull: {msg['error']}")
                total, done = msg.get("total"), msg.get("completed")
                if total and done:
                    key = msg.get("digest", "")
                    pct = 100 * done / total
                    if pct - last.get(key, -10.0) >= 10:
                        log(f"  {msg.get('status', '')}: {pct:.0f}% of {total / 1e9:.2f} GB")
                        last[key] = pct
                elif msg.get("status"):
                    log(f"  {msg['status']}")

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        # `format` constrains the JSON shape; the field guide tells the model what each field means.
        user = f"{user}\n\nJSON fields to fill:\n{schema_guide(schema)}"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "think": self.think,
            "keep_alive": "10m",
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        try:
            r = httpx.post(f"{self.url}/api/chat", json=body, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc!r}") from None
        if r.status_code != 200:
            raise LLMError(f"Ollama returned {r.status_code}: {r.text[:800]}")
        d = r.json()
        msg = d.get("message") or {}
        usage = {"prompt_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
                 "eval_s": (d.get("eval_duration") or 0) / 1e9,
                 "prompt_s": (d.get("prompt_eval_duration") or 0) / 1e9,
                 "done_reason": d.get("done_reason"), "thinking_chars": len(msg.get("thinking") or "")}
        return msg.get("content", ""), usage

    def unload(self) -> None:
        """Free RAM/VRAM: an empty request with keep_alive=0 unloads the model."""
        try:
            httpx.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=60)
        except httpx.HTTPError:
            pass

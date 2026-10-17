"""OpenAI-compatible chat APIs with free tiers: Groq, Cerebras, OpenRouter, ... (configurable).

Uses JSON mode (`response_format: json_object`) when the model supports it, puts the schema's
field guide in the prompt, and validates locally like every other provider. Free-tier limits,
model names and data-use terms belong to each provider and change; `unseen llm-models` lists
the models your key can use right now.
"""

from __future__ import annotations

import os

import httpx

from .base import LLMError, TransientLLMError, retry_after_seconds, schema_guide


class OpenAICompatProvider:
    def __init__(self, name: str, url: str, model: str, key_env: str, timeout_s: float = 300,
                 extra_body: dict | None = None):
        key = os.environ.get(key_env, "").strip()
        if not key or key.startswith("your-"):
            raise LLMError(f"{name}: set {key_env} in .env to use it")
        self.name = name
        self.url = url.rstrip("/")
        self.model = model
        self._key = key
        self.timeout_s = timeout_s
        self.extra_body = dict(extra_body or {})   # e.g. {"reasoning_effort": "low"} for gpt-oss
        self.json_mode = True

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"}

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.url}/models", headers=self._headers(), timeout=30)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: {exc!r}") from None
        if r.status_code != 200:
            raise LLMError(f"{self.name} returned {r.status_code}: {r.text[:300]}")
        return sorted(m.get("id", "") for m in r.json().get("data", []))

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        user = (f"{user}\n\nJSON fields to fill:\n{schema_guide(schema)}\n\n"
                f"Answer with a single compact JSON object (no indentation, no commentary).")
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                **self.extra_body}
        for _ in range(2):
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            else:
                body.pop("response_format", None)
            try:
                r = httpx.post(f"{self.url}/chat/completions", json=body, headers=self._headers(),
                               timeout=self.timeout_s)
            except httpx.HTTPError as exc:
                raise TransientLLMError(f"{self.name} request failed: {exc!r}") from None
            if r.status_code == 400 and self.json_mode and "response_format" in r.text:
                self.json_mode = False          # model without JSON mode: rely on the prompt + validation
                continue
            break
        if r.status_code == 429:
            raise TransientLLMError(f"{self.name}: rate limit / free quota reached ({r.text[:200]})",
                                    retry_after_seconds(r.headers))
        if r.status_code in (500, 502, 503, 504):
            raise TransientLLMError(f"{self.name}: temporarily unavailable ({r.status_code})")
        if r.status_code in (401, 403):
            raise LLMError(f"{self.name}: API key rejected ({r.status_code})")
        if r.status_code == 404:
            raise LLMError(f"{self.name}: model '{self.model}' not found; run `uv run unseen llm-models "
                           f"--provider {self.name}` and set it in config/story.yaml")
        if r.status_code != 200:
            raise LLMError(f"{self.name} returned {r.status_code}: {r.text[:500]}")
        d = r.json()
        try:
            choice = d["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError):
            raise LLMError(f"{self.name} returned no content: {str(d)[:500]}") from None
        usage = d.get("usage") or {}
        return text, {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                      "done_reason": choice.get("finish_reason")}

    def unload(self) -> None:
        pass

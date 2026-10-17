"""Optional hosted LLM: Google Gemini API (free tier, key from GEMINI_API_KEY).

Free-tier limits, model names and data-use terms change; check
https://ai.google.dev/gemini-api/docs/rate-limits and the terms before relying on it. The schema is
included in the prompt and the output is validated locally, so this works without depending on
the API's own schema features.
"""

from __future__ import annotations

import json
import os
import re

import httpx

from .base import LLMError, TransientLLMError, retry_after_seconds


class GeminiProvider:
    name = "gemini"

    def __init__(self, url: str, model: str, timeout_s: float = 300, thinking_budget: int | None = 1024,
                 max_output_tokens: int = 16384):
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key or key.startswith("your-"):
            raise LLMError("Set GEMINI_API_KEY in .env to use the Gemini provider "
                           "(free key: https://aistudio.google.com/apikey).")
        self._key = key
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        # Thinking tokens count against maxOutputTokens: unbounded thinking truncated long scripts.
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.log = print

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.url}/models", headers={"x-goog-api-key": self._key},
                          params={"pageSize": 200}, timeout=30)
        except httpx.HTTPError as exc:
            raise LLMError(f"gemini: {exc!r}") from None
        if r.status_code != 200:
            raise LLMError(f"gemini returned {r.status_code}: {r.text[:300]}")
        return sorted(m["name"].removeprefix("models/") for m in r.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))

    def auto_candidates(self) -> list[str]:
        """Stable 'gemini-X.Y-flash' models, newest first (no lite/image/preview variants)."""
        found = []
        for name in self.list_models():
            m = re.fullmatch(r"gemini-(\d+)(?:\.(\d+))?-flash", name)
            if m:
                found.append(((int(m.group(1)), int(m.group(2) or 0)), name))
        return [n for _, n in sorted(found, reverse=True)]

    def _post(self, model: str, body: dict) -> httpx.Response:
        try:
            return httpx.post(f"{self.url}/models/{model}:generateContent", json=body,
                              headers={"x-goog-api-key": self._key}, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"Gemini request failed: {exc!r}") from None

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        config = {"temperature": temperature, "responseMimeType": "application/json",
                  "maxOutputTokens": max(max_tokens, self.max_output_tokens)}
        if self.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": f"{user}\n\nReturn one compact JSON object matching "
                                                              f"this JSON Schema:\n{json.dumps(schema)}"}]}],
            "generationConfig": config,
        }
        # model 'auto': newest stable Flash model the key can actually use (404/403/429 -> next one)
        auto = self.model == "auto"
        candidates = self.auto_candidates() if auto else [self.model]
        if not candidates:
            raise LLMError("gemini: no gemini-X.Y-flash model available to this key; set llm.gemini.model")
        r = None
        for model in candidates:
            for _ in range(2):
                r = self._post(model, body)
                if r.status_code == 400 and "thinking" in r.text.lower() and "thinkingConfig" in config:
                    config.pop("thinkingConfig")   # model without a thinking budget setting
                    self.thinking_budget = None
                    continue
                break
            if auto and r.status_code in (403, 404, 429) and model != candidates[-1]:
                continue   # not on this key's free tier: try the next newest
            if auto and r.status_code == 200:
                self.model = model
                self.log(f"  gemini model auto-selected: {model}")
            break
        if r.status_code == 429:
            raise TransientLLMError("Gemini free-tier rate limit reached", retry_after_seconds(r.headers))
        if r.status_code in (500, 502, 503, 504):
            raise TransientLLMError(f"Gemini temporarily unavailable ({r.status_code}): {r.text[:160]}")
        if r.status_code == 404:
            raise LLMError(f"gemini: model '{self.model}' not found; run `uv run unseen llm-models --provider gemini`")
        if r.status_code != 200:
            raise LLMError(f"Gemini returned {r.status_code}: {r.text[:800]}")
        d = r.json()
        try:
            cand = d["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError):
            raise LLMError(f"Gemini returned no content: {json.dumps(d)[:800]}") from None
        meta = d.get("usageMetadata") or {}
        return text, {"prompt_tokens": meta.get("promptTokenCount"),
                      "output_tokens": meta.get("candidatesTokenCount"),
                      "thinking_tokens": meta.get("thoughtsTokenCount"),
                      "done_reason": cand.get("finishReason")}

    def unload(self) -> None:
        pass

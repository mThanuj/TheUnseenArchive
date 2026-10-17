"""Optional hosted LLM: Google Gemini API (free tier, key from GEMINI_API_KEY).

Free-tier limits, model names and data-use terms change; check
https://ai.google.dev/gemini-api/docs/rate-limits and the terms before relying on it. The schema is
included in the prompt and the output is validated locally, so this works without depending on
the API's own schema features.
"""

from __future__ import annotations

import json
import os

import httpx

from .base import LLMError


class GeminiProvider:
    name = "gemini"

    def __init__(self, url: str, model: str, timeout_s: float = 300):
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key or key.startswith("your-"):
            raise LLMError("Set GEMINI_API_KEY in .env to use the Gemini provider "
                           "(free key: https://aistudio.google.com/apikey).")
        self._key = key
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

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

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": f"{user}\n\nReturn one compact JSON object matching "
                                                              f"this JSON Schema:\n{json.dumps(schema)}"}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max(max_tokens, 8192),
                                 "responseMimeType": "application/json"},
        }
        try:
            r = httpx.post(f"{self.url}/models/{self.model}:generateContent", json=body,
                           headers={"x-goog-api-key": self._key}, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed: {exc!r}") from None
        if r.status_code == 429:
            raise LLMError("Gemini free-tier rate limit reached; try later or use provider: ollama")
        if r.status_code != 200:
            raise LLMError(f"Gemini returned {r.status_code}: {r.text[:800]}")
        d = r.json()
        try:
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError(f"Gemini returned no content: {json.dumps(d)[:800]}") from None
        meta = d.get("usageMetadata") or {}
        return text, {"prompt_tokens": meta.get("promptTokenCount"),
                      "output_tokens": meta.get("candidatesTokenCount")}

    def unload(self) -> None:
        pass

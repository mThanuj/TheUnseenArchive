"""Optional hosted LLM: Google Gemini API (free tier, key from GEMINI_API_KEY).

Free-tier limits, model names and data-use terms change; check
https://ai.google.dev/gemini-api/docs/rate-limits and the terms before relying on it. The schema is
included in the prompt and the output is validated locally, so this works without depending on
the API's own schema features.
"""

from __future__ import annotations

import base64
import json
import os
import re

import httpx

from .base import LLMError, ProviderLLMError, SkipCallLLMError, TransientLLMError, retry_after_seconds


def quota_error(model: str, r) -> TransientLLMError:
    """429 -> which limit (per minute / per day, from QuotaFailure) and when to retry (RetryInfo)."""
    try:
        err = r.json().get("error") or {}
    except ValueError:
        err = {}
    retry = retry_after_seconds(r.headers)
    quotas = []
    for d in err.get("details") or []:
        kind = d.get("@type", "")
        if kind.endswith("QuotaFailure"):
            quotas += [f"{v.get('quotaId', '?')}={v.get('quotaValue', '?')}" for v in d.get("violations") or []]
        elif kind.endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
            if m:
                retry = float(m.group(1))
    message = str(err.get("message") or r.text)
    # "limit: 0" = this model has no free tier for the key: as good as a used-up daily quota
    daily = any("PerDay" in q for q in quotas) or bool(re.search(r"limit:\s*0\b", message))
    what = ", ".join(quotas) or message[:160]
    return TransientLLMError(f"gemini {model}: free-tier limit ({what})", retry, daily=daily)


class GeminiProvider:
    name = "gemini"
    supports_images = True

    def __init__(self, url: str, model: str, timeout_s: float = 300, thinking_budget: int | None = 1024,
                 max_output_tokens: int = 16384, generation: dict | None = None):
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
        self.json_mode = True   # responseMimeType JSON; switched off for models without it (Gemma)
        # per-model generationConfig overrides from config (e.g. Gemma: {"thinkingConfig": {"thinkingLevel": "minimal"}})
        self.generation = dict(generation or {})
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

    def _post(self, model: str, body: dict) -> httpx.Response:
        try:
            return httpx.post(f"{self.url}/models/{model}:generateContent", json=body,
                              headers={"x-goog-api-key": self._key}, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"Gemini request failed: {exc!r}") from None

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int, images: list[tuple[str, bytes]] | None = None) -> tuple[str, dict]:
        """`images`: (caption, JPEG bytes) pairs, sent after the prompt, each preceded by its caption."""
        config = {"temperature": temperature, "maxOutputTokens": max(max_tokens, self.max_output_tokens)}
        if self.json_mode:
            config["responseMimeType"] = "application/json"
        if self.thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        config.update(self.generation)
        parts = [{"text": f"{user}\n\nReturn one compact JSON object matching this JSON Schema:\n{json.dumps(schema)}"}]
        for caption, jpeg in images or []:
            parts += [{"text": caption},
                      {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}}]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": config,
        }
        for _ in range(3):
            r = self._post(self.model, body)
            low = r.text.lower() if r.status_code == 400 else ""
            if "thinking" in low and "thinkingConfig" in config:
                config.pop("thinkingConfig")   # model without this thinking setting
                self.thinking_budget = None
                self.generation.pop("thinkingConfig", None)
                continue
            if "json" in low and "responseMimeType" in config:
                config.pop("responseMimeType")   # e.g. Gemma: no JSON mode; the prompt + validation suffice
                self.json_mode = False
                continue
            break
        if r.status_code == 429:
            raise quota_error(self.model, r)
        if r.status_code in (500, 502, 503, 504):
            raise TransientLLMError(f"Gemini temporarily unavailable ({r.status_code}): {r.text[:160]}")
        if r.status_code == 401 or (r.status_code == 400 and "api key" in r.text.lower()):
            raise ProviderLLMError(f"gemini: API key rejected ({r.status_code}); check GEMINI_API_KEY in .env")
        if r.status_code in (403, 404):
            raise LLMError(f"gemini: model '{self.model}' is not available to this key ({r.status_code}); "
                           f"see `uv run unseen llm-models --provider gemini`")
        if r.status_code == 400:   # request-specific (e.g. too long): other calls may work
            raise SkipCallLLMError(f"gemini {self.model} returned 400: {r.text[:300]}")
        if r.status_code != 200:
            raise LLMError(f"Gemini returned {r.status_code}: {r.text[:800]}")
        d = r.json()
        try:
            cand = d["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError):
            # blocked prompt / safety stop: this request only, the model stays usable
            raise SkipCallLLMError(f"Gemini returned no content: {json.dumps(d)[:400]}") from None
        meta = d.get("usageMetadata") or {}
        return text, {"prompt_tokens": meta.get("promptTokenCount"),
                      "output_tokens": meta.get("candidatesTokenCount"),
                      "thinking_tokens": meta.get("thoughtsTokenCount"),
                      "done_reason": cand.get("finishReason")}

    def unload(self) -> None:
        pass

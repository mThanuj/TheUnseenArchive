"""OpenAI-compatible chat APIs with free tiers: Groq, Cerebras, OpenRouter, ... (configurable).

Uses JSON mode (`response_format: json_object`) when the model supports it, puts the schema's
field guide in the prompt, and validates locally like every other provider. Free-tier limits,
model names and data-use terms belong to each provider and change; `unseen llm-models` lists
the models your key can use right now.
"""

from __future__ import annotations

import os
import re

import httpx

from .base import (LLMError, ProviderLLMError, SkipCallLLMError, TransientLLMError, retry_after_seconds,
                   schema_guide, try_again_seconds)

DAILY = re.compile(r"per[ -]day|\bdaily\b|\b(?:tpd|rpd)\b", re.IGNORECASE)


def expand_env(name: str, url: str) -> str:
    """'https://x/accounts/${CF_ACCOUNT_ID}/v1' -> value from .env (some APIs put an account id in the URL)."""
    def sub(m: re.Match) -> str:
        value = os.environ.get(m.group(1), "").strip()
        if not value or value.startswith("your-"):
            raise LLMError(f"{name}: set {m.group(1)} in .env to use it")
        return value
    return re.sub(r"\$\{(\w+)\}", sub, url)


def error_text(r) -> str:
    """The API's own error message, whatever its shape: OpenAI-style {"error": {"message"}} (OpenRouter
    adds the upstream reason in metadata.raw), Cloudflare {"errors": [{"code", "message"}]}, or
    {"message"/"detail": ...} (Mistral)."""
    try:
        body = r.json()
    except (ValueError, AttributeError):
        return r.text[:300]
    if not isinstance(body, dict):
        return r.text[:300]
    err = body.get("error")
    if isinstance(err, dict):
        raw = (err.get("metadata") or {}).get("raw")
        msg = err.get("message", "")
        return f"{msg} {raw}".strip() if raw else (msg or r.text[:300])
    if err:
        return str(err)
    if isinstance(body.get("errors"), list) and body["errors"]:
        return "; ".join(f"{e.get('code', '')} {e.get('message', '')}".strip() if isinstance(e, dict) else str(e)
                         for e in body["errors"])
    return str(body.get("message") or body.get("detail") or r.text[:300])


KEY_PROBLEM = re.compile(r"api[ _-]?key|token|authenticat|unauthori[sz]ed|credential", re.IGNORECASE)


def limit_error(name: str, model: str, r) -> LLMError:
    """429 (or 413) -> skip-this-call (request too large), or a rate limit with its scope and retry time."""
    text = error_text(r)
    low = text.lower()
    if r.status_code == 413 or "request too large" in low:
        return SkipCallLLMError(f"{name}:{model}: request too large ({text[:200]})")
    retry = retry_after_seconds(r.headers) or try_again_seconds(text)
    # account-wide budgets cover every model of the provider: OpenRouter's "free-models-per-min/day",
    # Cloudflare's daily free Neurons
    family = "free-models-per" in low or "neuron" in low
    return TransientLLMError(f"{name}:{model}: rate limit / free quota reached ({text[:200]})", retry,
                             daily=bool(DAILY.search(low)), family=family)


class OpenAICompatProvider:
    def __init__(self, name: str, url: str, model: str, key_env: str, timeout_s: float = 300,
                 extra_body: dict | None = None):
        key = os.environ.get(key_env, "").strip()
        if not key or key.startswith("your-"):
            raise LLMError(f"{name}: set {key_env} in .env to use it")
        self.name = name
        self.url = expand_env(name, url).rstrip("/")
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

    supports_images = False

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int, images: list | None = None) -> tuple[str, dict]:
        if images:
            raise SkipCallLLMError(f"{self.name}:{self.model}: no image input configured")
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
        if r.status_code in (413, 429):
            raise limit_error(self.name, self.model, r)
        if r.status_code in (500, 502, 503, 504):
            raise TransientLLMError(f"{self.name}:{self.model}: temporarily unavailable ({r.status_code})")
        if r.status_code == 403 and re.search(r"flagged|moderation", r.text, re.IGNORECASE):
            raise SkipCallLLMError(f"{self.name}:{self.model}: request refused by moderation")
        if r.status_code == 401 or (r.status_code == 403 and KEY_PROBLEM.search(error_text(r))):
            raise ProviderLLMError(f"{self.name}: API key rejected ({r.status_code}: {error_text(r)[:200]})")
        if r.status_code == 403:   # e.g. a model that isn't on the free plan: the provider's other models may work
            raise LLMError(f"{self.name}:{self.model}: access denied (403: {error_text(r)[:200]})")
        if r.status_code == 404:
            raise LLMError(f"{self.name}: model '{self.model}' not found; run `uv run unseen llm-models "
                           f"--provider {self.name}` and set it in config/story.yaml")
        if r.status_code in (400, 422):
            # request-specific (context too long, Groq's json_validate_failed, ...): other calls may work
            raise SkipCallLLMError(f"{self.name}:{self.model} returned {r.status_code}: {error_text(r)[:300]}")
        if r.status_code != 200:
            raise LLMError(f"{self.name} returned {r.status_code}: {r.text[:500]}")
        d = r.json()
        if d.get("error"):   # OpenRouter reports upstream failures inside a 200 response
            code = (d["error"] or {}).get("code") if isinstance(d["error"], dict) else None
            if code in (429, 502, 503, 504, 408):
                raise TransientLLMError(f"{self.name}:{self.model}: upstream error {code} ({error_text(r)[:200]})")
            raise SkipCallLLMError(f"{self.name}:{self.model}: {error_text(r)[:300]}")
        try:
            choice = d["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError):
            raise SkipCallLLMError(f"{self.name} returned no content: {str(d)[:500]}") from None
        usage = d.get("usage") or {}
        return text, {"prompt_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                      "done_reason": choice.get("finish_reason")}

    def unload(self) -> None:
        pass

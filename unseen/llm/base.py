"""Provider-agnostic structured generation: JSON-schema output, validation, and re-asking.

Every call returns a validated pydantic object plus timing/token stats. If the model's output is
malformed, or a semantic `check` reports problems, the model is asked again with the exact errors
(up to `max_retries` times). Remaining semantic problems are returned, never silently dropped.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Log = Callable[[str], None]


class LLMError(RuntimeError):
    pass


class TransientLLMError(LLMError):
    """Rate limit, overload or timeout: worth retrying the same model a bit later.

    daily: the model's daily quota is used up (skip it for the rest of the run).
    family: the limit covers every model of the provider (e.g. OpenRouter's account-wide free cap).
    """

    def __init__(self, message: str, retry_after: float | None = None, daily: bool = False,
                 family: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily
        self.family = family


class ProviderLLMError(LLMError):
    """The whole provider is unusable this run (key rejected, service not running): skip all its models."""


class SkipCallLLMError(LLMError):
    """This model can't answer this particular request (larger than its per-minute token cap, blocked
    by a safety filter): skip the model for this call only; other calls can still use it."""


def retry_after_seconds(headers) -> float | None:
    try:
        return float(headers.get("retry-after")) if headers.get("retry-after") else None
    except (TypeError, ValueError):
        return None


TRY_AGAIN = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?", re.IGNORECASE)


def try_again_seconds(text: str) -> float | None:
    """'Please try again in 1m2.5s' (Groq) -> 62.5."""
    m = TRY_AGAIN.search(text)
    if not m or not any(m.groups()):
        return None
    h, mins, secs, unit = m.groups()
    total = int(h or 0) * 3600 + int(mins or 0) * 60
    if secs:
        total += float(secs) / (1000 if unit == "ms" else 1)
    return float(total)


def expand_model_pattern(pattern: str, available: list[str]) -> list[str]:
    """'gemini-*-flash' -> matching model names, newest version first ('*' = a version number)."""
    rx = re.compile(re.escape(pattern).replace(r"\*", r"(\d+(?:\.\d+)*)"))
    found = []
    for name in available:
        m = rx.fullmatch(name)
        if m:
            version = tuple(int(x) for g in m.groups() for x in g.split("."))
            found.append((version, name))
    return [n for _, n in sorted(found, reverse=True)]


class Provider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        """Return (raw text, usage dict with prompt_tokens/output_tokens if known)."""
        ...


@dataclass
class CallStats:
    name: str
    provider: str
    model: str
    seconds: float
    attempts: int
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens_per_s(self) -> float | None:
        return round(self.output_tokens / self.seconds, 2) if self.seconds and self.output_tokens else None

    def as_dict(self) -> dict:
        return {**asdict(self), "tokens_per_s": self.tokens_per_s}


def inline_refs(schema: dict) -> dict:
    """Resolve local `$ref`s into one self-contained schema (grammar converters handle that best)."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
                return {**target, **{k: resolve(v) for k, v in node.items() if k != "$ref"}}
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


def schema_guide(schema: dict) -> str:
    """Compact field guide (path, type, limits, description) from an inlined schema.

    Grammar-constrained decoding (Ollama `format`) enforces the JSON *shape* only; the model still
    needs the field descriptions and limits in the prompt. This is much shorter than the raw schema.
    """
    lines: list[str] = []

    def walk(node: dict, path: str) -> None:
        kind = node.get("type")
        if kind == "object" or "properties" in node:
            for name, sub in (node.get("properties") or {}).items():
                walk(sub, f"{path}.{name}" if path else name)
            return
        if kind == "array":
            limits = "/".join(f"{k}={node[k]}" for k in ("minItems", "maxItems") if k in node)
            items = node.get("items") or {}
            if items.get("type") == "object" or "properties" in items:
                lines.append(f"- {path}: list of objects{' (' + limits + ')' if limits else ''}"
                             + (f" - {node['description']}" if node.get("description") else ""))
                walk(items, f"{path}[]")
                return
            node = {**items, "description": node.get("description", items.get("description")),
                    "_limits": limits}
            kind = f"list of {items.get('type', 'value')}"
        extras = []
        if "enum" in node:
            extras.append("one of: " + ", ".join(map(str, node["enum"])))
        for k, label in (("minimum", "min"), ("maximum", "max")):
            if k in node:
                extras.append(f"{label} {node[k]}")
        if node.get("_limits"):
            extras.append(node["_limits"])
        desc = node.get("description", "")
        lines.append(f"- {path} ({kind}{'; ' + '; '.join(extras) if extras else ''})"
                     + (f": {desc}" if desc else ""))

    walk(schema, "")
    return "\n".join(lines)


def extract_json(text: str) -> str:
    """Strip code fences / chatter around the first JSON object."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the model output")
    return text[start:end + 1]


TRUNCATION_REASONS = {"length", "max_tokens", "MAX_TOKENS"}
MAX_TOKEN_CEILING = 16384


def looks_truncated(raw: str, exc: Exception, reason: str | None) -> bool:
    """Cut off at the token limit: reported by the API, or JSON that starts but never closes."""
    if reason in TRUNCATION_REASONS:
        return True
    tail = raw.rstrip()
    if "{" not in tail:
        return False   # no JSON at all: a different failure
    if isinstance(exc, json.JSONDecodeError) and exc.pos >= len(exc.doc) - 2:
        return True
    return not tail.endswith(("}", "```"))


def generate(provider: Provider, model_cls: type[T], *, system: str, user: str, name: str,
             temperature: float = 0.7, max_tokens: int = 4096, max_retries: int = 2,
             check: Callable[[T], list[str]] | None = None,
             log: Log = print) -> tuple[T, CallStats, list[str]]:
    """Generate a `model_cls` instance. Returns (object, stats, remaining semantic problems)."""
    schema = inline_refs(model_cls.model_json_schema())
    prompt = user
    t0 = time.monotonic()
    stats = CallStats(name, provider.name, provider.model, 0.0, 0)
    best: tuple[T, list[str]] | None = None
    while True:
        stats.attempts += 1
        log(f"  llm[{name}] attempt {stats.attempts} ({provider.name}:{provider.model}) ...")
        raw, usage = provider.complete_json(system, prompt, schema, temperature, max_tokens)
        stats.provider, stats.model = provider.name, provider.model   # a fallback may have switched
        stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        stats.output_tokens += int(usage.get("output_tokens") or 0)
        stats.seconds = round(time.monotonic() - t0, 1)
        try:
            obj = model_cls.model_validate(json.loads(extract_json(raw)))
        except (ValueError, ValidationError) as exc:
            reason = usage.get("done_reason")
            truncated = looks_truncated(raw, exc, reason)
            cut = " OUTPUT WAS CUT OFF AT THE TOKEN LIMIT;" if truncated else ""
            detail = (f"done_reason={reason};{cut} output_tokens={usage.get('output_tokens')} "
                      f"thinking={usage.get('thinking_tokens') or usage.get('thinking_chars')} | "
                      f"{str(exc)[:600]} | output starts: {raw[:200]!r}")
            log(f"  llm[{name}] invalid output: {detail[:400]}")
            if stats.attempts > max_retries:
                if best:
                    return best[0], stats, best[1]
                raise LLMError(f"{name}: invalid output after {stats.attempts} attempts: {detail}")
            if truncated:   # retrying with the same budget would fail the same way
                max_tokens = min(max_tokens * 2, MAX_TOKEN_CEILING)
                log(f"  llm[{name}] retrying with a larger token budget ({max_tokens}) and a shorter answer")
                prompt = (f"{user}\n\nYour previous answer was cut off before the JSON was complete. Keep every "
                          f"text field short and return ONLY the complete JSON object.")
            else:
                prompt = (f"{user}\n\nYour previous answer was not valid JSON for the schema:\n{str(exc)[:1500]}\n"
                          f"Return ONLY the corrected JSON object.")
            continue
        problems = check(obj) if check else []
        if best is None or len(problems) < len(best[1]):
            best = (obj, problems)
        if not problems or stats.attempts > max_retries:
            log(f"  llm[{name}] done in {stats.seconds:.0f} s, {stats.output_tokens} tokens"
                + (f", {len(best[1])} unresolved problem(s)" if best[1] else ""))
            return best[0], stats, best[1]
        log(f"  llm[{name}] {len(problems)} problem(s), asking for a fix")
        prompt = (f"{user}\n\nYour previous answer:\n{obj.model_dump_json()}\n\nIt has these problems:\n- "
                  + "\n- ".join(problems) + "\n\nReturn the complete corrected JSON object.")

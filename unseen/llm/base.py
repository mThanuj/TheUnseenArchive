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
        stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        stats.output_tokens += int(usage.get("output_tokens") or 0)
        stats.seconds = round(time.monotonic() - t0, 1)
        try:
            obj = model_cls.model_validate(json.loads(extract_json(raw)))
        except (ValueError, ValidationError) as exc:
            if stats.attempts > max_retries:
                if best:
                    return best[0], stats, best[1]
                raise LLMError(f"{name}: invalid output after {stats.attempts} attempts: {str(exc)[:800]}")
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

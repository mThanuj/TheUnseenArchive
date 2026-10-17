"""Try LLM providers in order; on a provider failure (quota, outage, bad key) use the next one.

The first provider that works stays active for the rest of the run, so a story isn't written by
several models unless something breaks.
"""

from __future__ import annotations

from typing import Callable

from .base import LLMError

Log = Callable[[str], None]


class FallbackProvider:
    def __init__(self, providers: list, log: Log = print):
        if not providers:
            raise LLMError("no LLM provider is available")
        self.providers = providers
        self.active = 0
        self._log = log

    @property
    def log(self) -> Log:
        return self._log

    @log.setter
    def log(self, fn: Log) -> None:
        self._log = fn
        for p in self.providers:
            if hasattr(p, "log"):
                p.log = fn

    @property
    def name(self) -> str:
        return self.providers[self.active].name

    @property
    def model(self) -> str:
        return self.providers[self.active].model

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        errors = []
        for i in range(self.active, len(self.providers)):
            p = self.providers[i]
            try:
                out = p.complete_json(system, user, schema, temperature, max_tokens)
            except LLMError as exc:
                errors.append(f"{p.name}: {exc}")
                if i + 1 < len(self.providers):
                    self._log(f"  {p.name} unavailable ({str(exc)[:160]}); switching to "
                              f"{self.providers[i + 1].name}")
                continue
            self.active = i
            return out
        raise LLMError("every LLM provider failed: " + " | ".join(errors))

    def unload(self) -> None:
        for p in self.providers:
            unload = getattr(p, "unload", None)
            if unload:
                unload()

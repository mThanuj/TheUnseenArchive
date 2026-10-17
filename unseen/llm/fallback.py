"""Try LLM providers in preference order, with retries and cool-downs for temporary failures.

- A temporary failure (rate limit, overload, timeout) gets one short retry on the same provider;
  if it fails again the provider cools down (60 s, then 120 s, 240 s, ...) and the next one is used.
- A permanent failure (bad key, unknown model) disables that provider for the rest of the run.
- Every call starts again from the most preferred provider that isn't cooling down, so one hiccup
  doesn't push the whole story onto the slow local model.
"""

from __future__ import annotations

import time
from typing import Callable

from .base import LLMError, TransientLLMError

Log = Callable[[str], None]


class FallbackProvider:
    def __init__(self, providers: list, log: Log = print, cooldown_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if not providers:
            raise LLMError("no LLM provider is available")
        self.providers = providers
        self.active = 0
        self._log = log
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._sleep = sleep
        self._cool: dict[str, tuple[float, int]] = {}   # name -> (available again at, strikes)

    # ------------------------------------------------------------------ views
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

    def excluding(self, names: set[str]) -> "FallbackProvider | None":
        """Same providers minus `names`, sharing cool-down state (used for an independent critic)."""
        kept = [p for p in self.providers if p.name not in names]
        if not kept:
            return None
        view = FallbackProvider(kept, self._log, self.cooldown_s, self._clock, self._sleep)
        view._cool = self._cool
        return view

    # ------------------------------------------------------------------ calls
    def _cool_down(self, name: str, permanent: bool = False) -> float:
        _, strikes = self._cool.get(name, (0.0, 0))
        wait = float("inf") if permanent else self.cooldown_s * 2 ** strikes
        self._cool[name] = (self._clock() + wait, strikes + 1)
        return wait

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        errors = []
        for i, p in enumerate(self.providers):
            until, _ = self._cool.get(p.name, (0.0, 0))
            if until > self._clock():
                errors.append(f"{p.name}: cooling down")
                continue
            for attempt in range(2):
                try:
                    out = p.complete_json(system, user, schema, temperature, max_tokens)
                except TransientLLMError as exc:
                    if attempt == 0:
                        wait = min(exc.retry_after or 5.0, 20.0)
                        self._log(f"  {p.name} busy ({str(exc)[:120]}); retrying in {wait:.0f} s")
                        self._sleep(wait)
                        continue
                    wait = self._cool_down(p.name)
                    errors.append(f"{p.name}: {exc}")
                    self._log(f"  {p.name} still unavailable; cooling down {wait:.0f} s, trying the next provider")
                    break
                except LLMError as exc:
                    self._cool_down(p.name, permanent=True)
                    errors.append(f"{p.name}: {exc}")
                    self._log(f"  {p.name} disabled for this run ({str(exc)[:160]})")
                    break
                self._cool.pop(p.name, None)
                if i != self.active:
                    self._log(f"  using {p.name}:{p.model}")
                self.active = i
                return out
        raise LLMError("every LLM provider failed: " + " | ".join(errors))

    def unload(self) -> None:
        for p in self.providers:
            unload = getattr(p, "unload", None)
            if unload:
                unload()

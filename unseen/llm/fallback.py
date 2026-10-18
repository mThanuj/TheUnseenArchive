"""Try LLM models in preference order: best models before normal ones, and within a tier every
model of a provider before the next provider.

The chain is a list of slots (one provider + one model each), built by the story engine in this
order, e.g. gemini:gemini-3.8-flash, gemini:gemini-3.7-flash, groq:openai/gpt-oss-120b, ...,
then the normal tier (gemini:...-flash-lite, groq:qwen/qwen3.8-27b, ...), then ollama.

- Rate limit on one model: a short retry if the API says it frees up within seconds, otherwise
  that model cools down (for the time the API names, else 60 s, 120 s, ...) and the next model of
  the same provider is used; only when all of them are limited does the chain move on.
- A used-up daily quota skips that model for the rest of the run; a provider-wide limit (or a
  rejected key, or Ollama not running) skips all of that provider's models.
- A request one model can't take (too large for its limits, safety-blocked) skips that model for
  this call only.
- When every hosted model is cooling down, the call waits for the first one to free up (up to
  `max_wait_s`) instead of failing or falling back to the slow local model, which comes last.
- Every call starts again from the most preferred slot that isn't cooling down.
"""

from __future__ import annotations

import time
from typing import Callable

from .base import LLMError, ProviderLLMError, SkipCallLLMError, TransientLLMError

Log = Callable[[str], None]
INF = float("inf")


def brief(exc: Exception, n: int = 160) -> str:
    """One line (API error bodies are often pretty-printed JSON), at most n characters."""
    return " ".join(str(exc).split())[:n]


class FallbackProvider:
    def __init__(self, providers: list, log: Log = print, cooldown_s: float = 60.0, max_wait_s: float = 300.0,
                 slow: tuple[str, ...] = ("ollama",), short_retry_s: float = 10.0, max_cooldown_s: float = 900.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if not providers:
            raise LLMError("no LLM provider is available")
        self.providers = providers
        self.active = 0
        self._log = log
        self.cooldown_s = cooldown_s
        self.max_wait_s = max_wait_s
        self.slow = tuple(slow)
        self.short_retry_s = short_retry_s
        self.max_cooldown_s = max_cooldown_s
        self._clock = clock
        self._sleep = sleep
        # "name:model" (one slot) or "name" (the whole provider) -> (available again at, strikes)
        self._cool: dict[str, tuple[float, int]] = {}

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

    @staticmethod
    def key(p) -> str:
        return f"{p.name}:{p.model}"

    def _view(self, providers: list) -> "FallbackProvider":
        view = FallbackProvider(providers, self._log, self.cooldown_s, self.max_wait_s, self.slow,
                                self.short_retry_s, self.max_cooldown_s, self._clock, self._sleep)
        view._cool = self._cool   # shared: a limit seen by the critic also applies to the writer
        return view

    def excluding(self, names: set[str]) -> "FallbackProvider | None":
        """Same slots minus every model of the providers in `names` (shares cool-down state)."""
        kept = [p for p in self.providers if p.name not in names]
        return self._view(kept) if kept else None

    def critic_view(self) -> "FallbackProvider | None":
        """Hosted models for an independent critic: never the model that wrote last; within each tier,
        other providers first (models grade their own family's work kindly), then the writer's."""
        writer = self.providers[self.active]
        fast = [p for p in self.providers if p.name not in self.slow and p is not writer]
        ordered = sorted(fast, key=lambda p: (getattr(p, "tier", 0), p.name == writer.name))   # stable
        return self._view(ordered) if ordered else None

    def minor_view(self) -> "FallbackProvider":
        """Same slots with the normal tier first: for small fix-up calls, keeping the best models'
        daily quota for the calls that shape the story."""
        return self._view(sorted(self.providers, key=lambda p: -getattr(p, "tier", 0)))   # stable

    # ------------------------------------------------------------------ state
    def _until(self, p) -> float:
        return max(self._cool.get(self.key(p), (0.0, 0))[0], self._cool.get(p.name, (0.0, 0))[0])

    def _cool_down(self, key: str, wait: float | None = None) -> float:
        """Mark a slot ("name:model") or a whole provider ("name") unavailable; returns the wait."""
        _, strikes = self._cool.get(key, (0.0, 0))
        if wait is None:
            wait = min(self.cooldown_s * 2 ** strikes, self.max_cooldown_s)
        wait = max(wait, 1.0)
        self._cool[key] = (self._clock() + wait, strikes + 1)
        return wait

    def _limited(self, p, exc: TransientLLMError) -> None:
        key = p.name if exc.family else self.key(p)
        what = f"every {p.name} model" if exc.family else key
        detail = brief(exc)
        if exc.daily:
            self._cool_down(key, INF)
            self._log(f"  {what}: daily free quota used up ({detail}); skipped for the rest of this run")
            return
        wait = self._cool_down(key, exc.retry_after)
        self._log(f"  {what} rate-limited ({detail}); cooling down {wait:.0f} s, trying the next model")

    # ------------------------------------------------------------------ calls
    def _try(self, i: int, p, args: tuple, errors: dict[str, str], skip: set[str]):
        """Call one slot; returns its output, or None after recording why it failed."""
        key = self.key(p)
        for attempt in range(2):
            try:
                out = p.complete_json(*args)
            except SkipCallLLMError as exc:
                skip.add(key)
                errors[key] = brief(exc, 300)
                self._log(f"  {key} can't take this request ({brief(exc)}); skipping it for this step")
                return None
            except TransientLLMError as exc:
                errors[key] = brief(exc, 300)
                soon = exc.retry_after is not None and exc.retry_after <= self.short_retry_s
                if attempt == 0 and soon and not exc.daily:
                    self._log(f"  {key} busy; retrying in {max(exc.retry_after, 1.0):.0f} s")
                    self._sleep(max(exc.retry_after, 1.0))
                    continue
                self._limited(p, exc)
                return None
            except LLMError as exc:
                errors[key] = brief(exc, 300)
                if isinstance(exc, ProviderLLMError):
                    self._cool_down(p.name, INF)
                    self._log(f"  {p.name} disabled for this run ({brief(exc)})")
                else:
                    self._cool_down(key, INF)
                    self._log(f"  {key} disabled for this run ({brief(exc)})")
                return None
            self._cool.pop(key, None)
            self._cool.pop(p.name, None)
            if i != self.active:
                self._log(f"  using {key}")
            self.active = i
            return out
        return None

    def _next_free(self, skip: set[str]):
        """(time, slot) of the hosted slot that frees up first, or None if none ever will."""
        pending = [(self._until(p), p) for p in self.providers
                   if p.name not in self.slow and self.key(p) not in skip and self._until(p) < INF]
        return min(pending, key=lambda x: x[0]) if pending else None

    def complete_json(self, system: str, user: str, schema: dict, temperature: float,
                      max_tokens: int) -> tuple[str, dict]:
        args = (system, user, schema, temperature, max_tokens)
        errors: dict[str, str] = {}
        skip: set[str] = set()   # slots that can't take this request: this call only
        waited = 0.0
        while True:
            for i, p in enumerate(self.providers):
                if p.name in self.slow or self.key(p) in skip or self._until(p) > self._clock():
                    continue
                out = self._try(i, p, args, errors, skip)
                if out is not None:
                    return out
            nxt = self._next_free(skip)
            if nxt is not None:
                delay = max(0.0, nxt[0] - self._clock())
                if waited + delay <= self.max_wait_s:
                    self._log(f"  every hosted model is rate-limited; waiting {delay:.0f} s for {self.key(nxt[1])}")
                    self._sleep(delay)
                    waited += max(delay, 1.0)
                    continue
            for i, p in enumerate(self.providers):   # the slow local model is the last resort
                if p.name in self.slow and self._until(p) <= self._clock():
                    out = self._try(i, p, args, errors, skip)
                    if out is not None:
                        return out
            detail = " | ".join(f"{k}: {v[:160]}" for k, v in errors.items()) or "all cooling down"
            raise LLMError("every LLM provider failed: " + detail)

    def unload(self) -> None:
        for p in self.providers:
            unload = getattr(p, "unload", None)
            if unload:
                unload()

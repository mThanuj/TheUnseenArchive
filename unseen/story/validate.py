"""Deterministic checks on LLM output. Every problem found is fed back to the model to fix.

These rules encode what the LLM is told but cannot be trusted to obey: timing that fits the target
length, every narration line placed in a scene, video prompts free of artefact words, no clichés,
and, for real mysteries, every factual line backed by a quote-verified claim with theories hedged.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .config import StoryConfig
from .models import Claim, StoryDraft


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _norm(text: str) -> str:
    """Lowercase alphanumerics only: robust to punctuation/quote/dash drift when copying quotes."""
    text = unicodedata.normalize("NFKD", text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def quote_in_source(quote: str, source: str) -> bool:
    q = _norm(quote)
    return len(q.split()) >= 4 and q in _norm(source)


def verify_claims(claims: list[Claim], source_text: str) -> tuple[list[Claim], list[Claim]]:
    """Split claims into (verified, rejected) by checking each quote really occurs in the source."""
    ok, bad = [], []
    for c in claims:
        (ok if quote_in_source(c.quote, source_text) else bad).append(c)
    return ok, bad


def _phrase_re(phrase: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(phrase.lower()).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)


def banned_in(text: str, phrases: list[str]) -> list[str]:
    """Phrases (whole-word, case-insensitive) that occur in `text`."""
    return [p for p in phrases if _phrase_re(p).search(text)]


def sanitize_video_prompt(text: str, banned: list[str]) -> str:
    """Last resort after the model failed to fix it: drop banned phrases, tidy punctuation."""
    out = text
    for p in sorted(banned, key=len, reverse=True):
        out = _phrase_re(p).sub("", out)
    out = re.sub(r"\s+([,.;])", r"\1", out)
    out = re.sub(r"([,;])\s*[,;]+", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def speech_seconds(text: str, wps: float) -> float:
    return len(words(text)) / wps


def validate_story(d: StoryDraft, cfg: StoryConfig, target_seconds: int, category: str,
                   claims: list[Claim] | None = None) -> list[str]:
    r, length = cfg.rules, cfg.length
    problems: list[str] = []
    wps = length.words_per_second

    # narration lines
    ids = [n.id for n in d.narration]
    if len(ids) != len(set(ids)):
        problems.append("narration ids must be unique")
    if d.narration:
        hook = d.narration[0].text
        if len(words(hook)) > r.hook_max_words:
            problems.append(f"the first narration line (hook) has {len(words(hook))} words; max {r.hook_max_words}")
    for n in d.narration:
        if len(words(n.text)) > r.narration_line_max_words:
            problems.append(f"narration line {n.id} has {len(words(n.text))} words; max {r.narration_line_max_words}")

    # scenes and coverage
    if len(d.scenes) > length.max_scenes:
        problems.append(f"{len(d.scenes)} scenes; max {length.max_scenes}")
    covered = Counter(i for s in d.scenes for i in s.narration_ids)
    for i in ids:
        if covered[i] == 0:
            problems.append(f"narration line {i} is not assigned to any scene")
        elif covered[i] > 1:
            problems.append(f"narration line {i} is assigned to {covered[i]} scenes; use exactly one")
    unknown = sorted(set(covered) - set(ids))
    if unknown:
        problems.append(f"scenes reference unknown narration ids {unknown}")

    # timing
    total = sum(s.duration_s for s in d.scenes)
    tolerance = max(5.0, 0.25 * target_seconds)
    if abs(total - target_seconds) > tolerance:
        problems.append(f"scene durations add up to {total:.0f} s; target is {target_seconds} s (+/- {tolerance:.0f})")
    by_id = {n.id: n for n in d.narration}
    speech = sum(speech_seconds(n.text, wps) + n.pause_after_s for n in d.narration)
    if speech > total * 1.05:
        problems.append(f"narration needs ~{speech:.0f} s spoken at a slow pace but the scenes last {total:.0f} s; "
                        f"cut words or lengthen scenes")
    for s in d.scenes:
        need = sum(speech_seconds(by_id[i].text, wps) for i in s.narration_ids if i in by_id)
        if need > s.duration_s + 1.0:
            problems.append(f"scene {s.id} lasts {s.duration_s} s but its narration needs ~{need:.1f} s; "
                            f"split the line across scenes or lengthen the scene (max {length.scene_seconds.max} s)")

    # video prompts
    for s in d.scenes:
        n_words = len(words(s.video_prompt))
        if not (r.video_prompt_words.min <= n_words <= r.video_prompt_words.max):
            problems.append(f"scene {s.id} video_prompt has {n_words} words; use "
                            f"{r.video_prompt_words.min}-{r.video_prompt_words.max}")
        bad = banned_in(s.video_prompt, r.video_prompt_banned_words)
        if bad:
            problems.append(f"scene {s.id} video_prompt contains {bad}; describe the scene itself, "
                            f"degradation and on-screen text are added later")

    # clichés
    text = " ".join([d.logline, *(n.text for n in d.narration)]).lower()
    for c in r.banned_cliches:
        if c.lower() in text:
            problems.append(f"avoid the cliche '{c}'")

    # real mysteries: grounding
    if category == "real_mystery":
        claim_by_id = {c.id: c for c in claims or []}
        cited = 0
        for n in d.narration:
            missing = [c for c in n.claim_ids if c not in claim_by_id]
            if missing:
                problems.append(f"narration line {n.id} cites unknown claim ids {missing}")
            tags = {claim_by_id[c].tag for c in n.claim_ids if c in claim_by_id}
            if n.claim_ids:
                cited += 1
            if tags & {"disputed", "theory", "legend"} and not banned_in(n.text, cfg.real_mystery.hedge_words):
                problems.append(f"narration line {n.id} states a {'/'.join(sorted(tags - {'fact'}))} claim as fact; "
                                f"hedge it (e.g. 'reportedly', 'some believe', 'according to')")
        if cited < max(1, len(d.narration) // 2):
            problems.append(f"only {cited} of {len(d.narration)} narration lines cite claims; factual lines "
                            f"must cite the claim ids they rely on")
    return problems

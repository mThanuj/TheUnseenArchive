"""Deterministic checks on LLM output. Every problem found is fed back to the model to fix.

These rules encode what the LLM is told but cannot be trusted to obey: timing that fits the target
length, every narration line placed in a scene, video prompts free of artefact words, no clichés,
and, for real mysteries, every factual line backed by a quote-verified claim with theories hedged.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime

from .config import StoryConfig
from .models import Claim, StoryDraft


def words(text: str) -> list[str]:
    return re.findall(r"[\w'’]+", text)


def _norm(text: str) -> str:
    """Lowercase alphanumerics only, accents stripped: robust to punctuation/quote/dash/accent drift
    when the model copies a quote."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()
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
    """Last resort after the model failed to fix it: drop the sentences that use banned phrases
    (removing single words left broken prompts like "renders the scene in, muted gray"); if every
    sentence has one, remove just the phrases."""
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s and not banned_in(s, banned)]
    if kept:
        return " ".join(kept)
    out = text
    for p in sorted(banned, key=len, reverse=True):
        out = _phrase_re(p).sub("", out)
    out = re.sub(r"\s+([,.;])", r"\1", out)
    out = re.sub(r"([,;])\s*[,;]+", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def speech_seconds(text: str, wps: float) -> float:
    return len(words(text)) / wps


FIRST_PERSON = re.compile(r"\bI\b|\b(?i:me|my|mine|myself)\b")


QUOTED = re.compile(r'["“”][^"“”]{1,40}["“”]|(?<![\w])\'[^\']{1,30}\'(?![\w])')
CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def readable_text_risk(prompt: str, risk_words: list[str]) -> list[str]:
    """Words/quotes/clock times that ask the video model to render legible text (it renders gibberish)."""
    found = banned_in(prompt, risk_words)
    found += [m.group(0) for m in QUOTED.finditer(prompt)]
    found += [m.group(0) for m in CLOCK_TIME.finditer(prompt)]
    return found


def device_mix_issues(d: StoryDraft, cfg: StoryConfig, category: str, event_year: int | None) -> dict[int, str]:
    """Real cases keep one look: a single device for events before any camera existed (a phone inside
    a 12th-century scene breaks the reconstruction), at most two device types otherwise."""
    if category != "real_mystery" or not d.scenes:
        return {}
    first_camera = min(dev.since for dev in cfg.ingredients.device)
    allowed = 1 if event_year is not None and event_year < first_camera else 2
    keep = [dev for dev, _ in Counter(s.device for s in d.scenes).most_common(allowed)]
    rule = "one device for the whole reconstruction" if allowed == 1 else "at most two device types"
    return {s.id: f"use {keep[0]} like the rest of this reconstruction ({rule})"
            for s in d.scenes if s.device not in keep}


def victim_shots(d: StoryDraft, cfg: StoryConfig, category: str) -> dict[int, list[str]]:
    """Real cases: scenes whose video prompt would show real victims (bodies, remains, injuries)."""
    if category != "real_mystery":
        return {}
    found = {s.id: banned_in(s.video_prompt, cfg.real_mystery.never_show) for s in d.scenes}
    return {i: w for i, w in found.items() if w}


def era_problems(d: StoryDraft, cfg: StoryConfig, category: str, event_year: int | None) -> list[str]:
    since = {dev.id: dev.since for dev in cfg.ingredients.device}
    problems = []
    for s in d.scenes:
        first = since.get(s.device, 1900)
        if first > d.recorded_year:
            problems.append(f"scene {s.id}: a {s.device} could not record footage in {d.recorded_year} "
                            f"(available from {first}); use a period device or change recorded_year")
    if category == "real_mystery" and event_year:
        now = datetime.now().year
        if not (event_year - 1 <= d.recorded_year <= event_year + 15 or d.recorded_year >= now - 3):
            problems.append(f"recorded_year {d.recorded_year} fits neither period footage of the case "
                            f"({event_year}-{event_year + 15}) nor a present-day revisit ({now - 3} or later)")
    return problems


def first_person_slip(text: str) -> bool:
    """The Archivist speaks in the third person; quoted log lines/voices may use 'I'."""
    if any(q in text for q in ('"', "“", "”")):
        return False
    return bool(FIRST_PERSON.search(text))


# times, dates, counts: the details a model invents to sound documentary ("one" is left out: "no one")
SPECIFIC = re.compile(r"\d|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
                      r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|o'clock)\b|\b[ap]\.m\.",
                      re.IGNORECASE)


def uncited_specific(line) -> bool:
    """Real cases: a narration line with a time/number/count but no claim behind it."""
    return not line.claim_ids and bool(SPECIFIC.search(line.text))


def min_cited(d: StoryDraft) -> int:
    """Real cases: at least two thirds of the narration lines must cite claims."""
    return max(1, (2 * len(d.narration)) // 3)


def repeated_lines(d: StoryDraft, overlap: float = 0.7) -> dict[int, int]:
    """{line id: earlier line id} for narration lines that mostly repeat an earlier one (share at least
    70% of the shorter line's content words, e.g. "2019 inspection found brickwork intact" twice)."""
    seen: list[tuple[int, set[str]]] = []
    out: dict[int, int] = {}
    for n in d.narration:
        content = {w.lower() for w in words(n.text) if len(w) > 3}
        for prev_id, prev in seen:
            small = min(len(content), len(prev))
            if small >= 3 and len(content & prev) / small >= overlap:
                out[n.id] = prev_id
                break
        seen.append((n.id, content))
    return out


def unhedged_lines(d: StoryDraft, cfg: StoryConfig, claims: list[Claim] | None) -> dict[int, list[str]]:
    """Real cases: {line id: tags} for lines stating a theory/disputed claim without a hedge, or telling
    a legend before it has been framed. A legend needs its framing once ("According to two medieval
    chroniclers, ..."); later lines are told within that frame, so the narration doesn't repeat
    "according to legend" on every line. Theories and disputed points are hedged every time."""
    by_id = {c.id: c for c in claims or []}
    framed = False
    out: dict[int, list[str]] = {}
    for n in d.narration:
        tags = {by_id[c].tag for c in n.claim_ids if c in by_id}
        hedged = bool(banned_in(n.text, cfg.real_mystery.hedge_words))
        need = sorted(tags & {"theory", "disputed"}) + (["legend"] if "legend" in tags and not framed else [])
        if need and not hedged:
            out[n.id] = need
        if "legend" in tags and hedged:
            framed = True
    return out


def expected_scenes(cfg: StoryConfig, target_seconds: int) -> int:
    """One generated clip per scene of about `typical` seconds (also the per-Reel GPU budget)."""
    ln = cfg.length
    return max(3, min(ln.max_scenes, round(target_seconds / ln.scene_seconds.typical)))


def validate_story(d: StoryDraft, cfg: StoryConfig, target_seconds: int, category: str,
                   claims: list[Claim] | None = None, event_year: int | None = None) -> list[str]:
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
        if first_person_slip(n.text):
            problems.append(f"narration line {n.id} uses I/me/my; the Archivist narrates in the third person")
        told = banned_in(n.text, r.narration_banned_words)
        if told:
            problems.append(f"narration line {n.id} uses {told}; state facts and let the footage do the work")
    for i, j in repeated_lines(d).items():
        problems.append(f"narration line {i} repeats line {j}; every line should add something new")

    # scenes and coverage
    if len(d.scenes) > length.max_scenes:
        problems.append(f"{len(d.scenes)} scenes; max {length.max_scenes}")
    want = expected_scenes(cfg, target_seconds)
    if len(d.scenes) < want - 1:
        problems.append(f"only {len(d.scenes)} scenes; a {target_seconds} s Reel needs {want} (one clip per scene)")
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
        text = readable_text_risk(s.video_prompt, r.video_prompt_text_risk)
        if text:
            problems.append(f"scene {s.id} video_prompt asks for readable text {text}; show the object "
                            f"without legible writing")
    problems += era_problems(d, cfg, category, event_year)
    problems += [f"scene {i}: {why}" for i, why in device_mix_issues(d, cfg, category, event_year).items()]
    problems += [f"scene {i} video_prompt shows real victims {w}; show the place, objects or traces instead"
                 for i, w in victim_shots(d, cfg, category).items()]

    # clichés
    text = " ".join([d.logline, *(n.text for n in d.narration)]).lower()
    for c in r.banned_cliches:
        if c.lower() in text:
            problems.append(f"avoid the cliche '{c}'")

    # real mysteries: grounding
    if category == "real_mystery":
        claim_by_id = {c.id: c for c in claims or []}
        cited = 0
        unhedged = unhedged_lines(d, cfg, claims)
        for n in d.narration:
            missing = [c for c in n.claim_ids if c not in claim_by_id]
            if missing:
                problems.append(f"narration line {n.id} cites unknown claim ids {missing}")
            if n.claim_ids:
                cited += 1
            if n.id in unhedged:
                problems.append(f"narration line {n.id} states a {'/'.join(unhedged[n.id])} claim as fact; "
                                f"hedge it (e.g. 'reportedly', 'some believe', 'according to')")
            if uncited_specific(n):
                problems.append(f"narration line {n.id} states a specific detail without citing a claim")
        if cited < min_cited(d):
            problems.append(f"only {cited} of {len(d.narration)} narration lines cite claims; factual lines "
                            f"must cite the claim ids they rely on")
    return problems

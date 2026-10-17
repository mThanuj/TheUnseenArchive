"""Repairs that avoid regenerating the whole script.

1. `auto_fix` applies deterministic fixes (timing, line placement, prompt length/words).
2. `line_issues` lists narration lines that need wording changes; the engine asks the LLM to
   rewrite only those lines (`LineFixes`, a few hundred tokens instead of thousands).
3. `apply_patch` applies an editor revision expressed as targeted edits (`StoryPatch`).
"""

from __future__ import annotations

import re

from .config import StoryConfig
from .models import Claim, LineFixes, StoryDraft, StoryPatch
from .validate import banned_in, first_person_slip, sanitize_video_prompt, speech_seconds, words


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _scene_need(d: StoryDraft, scene, wps: float) -> float:
    by_id = {n.id: n for n in d.narration}
    lines = [by_id[i] for i in scene.narration_ids if i in by_id]
    return sum(speech_seconds(n.text, wps) for n in lines) + sum(n.pause_after_s for n in lines[:-1])


def place_lines(d: StoryDraft) -> list[str]:
    """Every narration line in exactly one scene, in order; unknown ids removed."""
    fixes: list[str] = []
    ids = [n.id for n in d.narration]
    known, seen = set(ids), set()
    for s in d.scenes:
        keep = []
        for i in s.narration_ids:
            if i in known and i not in seen:
                keep.append(i)
                seen.add(i)
        if keep != s.narration_ids:
            fixes.append(f"scene {s.id}: narration ids {s.narration_ids} -> {keep}")
            s.narration_ids = keep
    for i in ids:
        if i in seen:
            continue
        # put it with the scene holding the nearest earlier line, else the first scene
        target = d.scenes[0]
        for s in d.scenes:
            if any(j < i for j in s.narration_ids):
                target = s
        target.narration_ids = sorted(target.narration_ids + [i], key=ids.index)
        seen.add(i)
        fixes.append(f"narration line {i} placed in scene {target.id}")
    return fixes


def fit_durations(d: StoryDraft, cfg: StoryConfig, target: float) -> list[str]:
    """Each scene long enough for its narration; total as close to the target as limits allow."""
    ln = cfg.length
    lo, hi = ln.scene_seconds.min, ln.scene_seconds.max
    before = [s.duration_s for s in d.scenes]
    mins = [_clamp(_scene_need(d, s, ln.words_per_second) + 0.3, lo, hi) for s in d.scenes]
    durs = [_clamp(s.duration_s, m, hi) for s, m in zip(d.scenes, mins)]
    for _ in range(20):
        diff = target - sum(durs)
        if abs(diff) < 0.2:
            break
        movable = [i for i in range(len(durs)) if (diff > 0 and durs[i] < hi) or (diff < 0 and durs[i] > mins[i])]
        if not movable:
            break
        step = diff / len(movable)
        for i in movable:
            durs[i] = _clamp(durs[i] + step, mins[i], hi)
    for s, v in zip(d.scenes, durs):
        s.duration_s = round(v, 1)
    after = [s.duration_s for s in d.scenes]
    return [f"scene durations {before} -> {after}"] if after != before else []


def fit_video_prompts(d: StoryDraft, cfg: StoryConfig) -> list[str]:
    r = cfg.rules
    fixes: list[str] = []
    for s in d.scenes:
        p = s.video_prompt
        if banned_in(p, r.video_prompt_banned_words):
            p = sanitize_video_prompt(p, r.video_prompt_banned_words)
        if len(words(p)) < r.video_prompt_words.min:   # enrich from the scene's own fields
            extra = [f"The camera is {s.camera.rstrip('.')}.", f"{s.lighting.rstrip('.').capitalize()}.",
                     f"{s.motion.rstrip('.').capitalize()}.", f"{s.visual.rstrip('.').capitalize()}."]
            for e in extra:
                if len(words(p)) >= r.video_prompt_words.min:
                    break
                if e.strip(". ") and not banned_in(e, r.video_prompt_banned_words):
                    p = f"{p.rstrip()} {e}"
        if len(words(p)) > r.video_prompt_words.max:   # cut at a sentence boundary
            kept, n = [], 0
            for sent in re.split(r"(?<=[.!?])\s+", p):
                if n + len(words(sent)) > r.video_prompt_words.max and kept:
                    break
                kept.append(sent)
                n += len(words(sent))
            p = " ".join(kept)
        if p != s.video_prompt:
            fixes.append(f"scene {s.id} video_prompt adjusted ({len(words(s.video_prompt))} -> {len(words(p))} words)")
            s.video_prompt = p
    return fixes


def auto_fix(d: StoryDraft, cfg: StoryConfig, target: float) -> list[str]:
    return place_lines(d) + fit_durations(d, cfg, target) + fit_video_prompts(d, cfg)


def line_issues(d: StoryDraft, cfg: StoryConfig, target: float, category: str,
                claims: list[Claim] | None) -> dict[int, str]:
    """Narration lines whose wording must change, with the reason for each."""
    r, wps = cfg.rules, cfg.length.words_per_second
    issues: dict[int, list[str]] = {}

    def add(i: int, why: str) -> None:
        issues.setdefault(i, []).append(why)

    if d.narration and len(words(d.narration[0].text)) > r.hook_max_words:
        add(d.narration[0].id, f"hook must be at most {r.hook_max_words} words")
    for n in d.narration:
        if len(words(n.text)) > r.narration_line_max_words:
            add(n.id, f"at most {r.narration_line_max_words} words")
        if first_person_slip(n.text):
            add(n.id, "the Archivist narrates in the third person: no I/me/my (quote a log line if needed)")
        for c in r.banned_cliches:
            if c.lower() in n.text.lower():
                add(n.id, f"remove the cliche '{c}'")
    total = sum(s.duration_s for s in d.scenes)
    speech = sum(speech_seconds(n.text, wps) + n.pause_after_s for n in d.narration)
    if speech > total * 1.05:   # still too long after timing fixes: shorten the longest lines
        excess_words = (speech - total) * wps
        for n in sorted(d.narration, key=lambda n: -len(words(n.text)))[:3]:
            add(n.id, f"shorten (the narration is ~{excess_words:.0f} words too long overall)")
    for s in d.scenes:   # one scene's narration longer than the clip can hold
        need = _scene_need(d, s, wps)
        if need > cfg.length.scene_seconds.max + 0.5:
            for i in s.narration_ids:
                add(i, f"scene {s.id} lasts at most {cfg.length.scene_seconds.max:g} s; shorten this line")
    if category == "real_mystery":
        by_id = {c.id: c for c in claims or []}
        cited = [n for n in d.narration if n.claim_ids]
        for n in d.narration:
            tags = {by_id[c].tag for c in n.claim_ids if c in by_id}
            if tags & {"disputed", "theory", "legend"} and not banned_in(n.text, cfg.real_mystery.hedge_words):
                add(n.id, "hedge it (reportedly / some believe / according to): it relies on a "
                          + "/".join(sorted(tags - {"fact"})) + " claim")
            if any(c not in by_id for c in n.claim_ids):
                add(n.id, "uses unknown claim ids; cite only the listed claims")
        if len(cited) < max(1, len(d.narration) // 2):
            for n in d.narration:
                if not n.claim_ids:
                    add(n.id, "cite the claim ids this line relies on (or make it purely atmospheric)")
    return {i: "; ".join(v) for i, v in issues.items()}


def apply_line_fixes(d: StoryDraft, fixes: LineFixes, wanted: set[int], category: str) -> list[str]:
    by_id = {n.id: n for n in d.narration}
    changed = []
    for f in fixes.lines:
        n = by_id.get(f.id)
        if n is None or f.id not in wanted or not f.text.strip():
            continue
        n.text = f.text.strip()
        if category == "real_mystery" and f.claim_ids:
            n.claim_ids = f.claim_ids
        changed.append(f"line {f.id} rewritten")
    return changed


def apply_patch(d: StoryDraft, patch: StoryPatch, category: str) -> list[str]:
    changed: list[str] = []
    by_line = {n.id: n for n in d.narration}
    for f in patch.narration:
        if f.id in by_line and f.text.strip():
            by_line[f.id].text = f.text.strip()
            if category == "real_mystery" and f.claim_ids:
                by_line[f.id].claim_ids = f.claim_ids
            changed.append(f"line {f.id}")
    by_scene = {s.id: s for s in d.scenes}
    for e in patch.scenes:
        s = by_scene.get(e.id)
        if s is None:
            continue
        for field in ("visual", "motion", "camera", "lighting", "video_prompt"):
            value = getattr(e, field).strip()
            if value:
                setattr(s, field, value)
                changed.append(f"scene {e.id}.{field}")
    if patch.ending_note.strip():
        d.ending_note = patch.ending_note.strip()
        changed.append("ending_note")
    return changed

"""Write a finished story to runs/stories/<id>/: story.json, story.md (for humans), prompts.json
(scene video prompts in the benchmark's prompt format, so `story render` can generate them)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import REPO_ROOT
from .models import StoryRecord

STORIES_DIR = REPO_ROOT / "runs" / "stories"


def story_dir(story_id: str) -> Path:
    return STORIES_DIR / story_id


def scene_prompts(record: StoryRecord) -> list[dict]:
    """[{id, seed, text}] per scene; seeds are stable per story+scene for reproducible renders."""
    short = hashlib.sha1(record.id.encode()).hexdigest()[:6]
    out = []
    for s in record.draft.scenes:
        seed = int(hashlib.sha256(f"{record.id}:{s.id}".encode()).hexdigest()[:8], 16)
        out.append({"id": f"{short}_s{s.id:02d}", "seed": seed, "text": s.video_prompt})
    return out


def to_markdown(r: StoryRecord) -> str:
    d = r.draft
    by_id = {n.id: n for n in d.narration}
    lines = [
        f"# {d.title}",
        "",
        f"*{d.archive_entry}*",
        "",
        f"- Category: {r.category} | target {r.target_seconds} s | scenes {len(d.scenes)} "
        f"({sum(s.duration_s for s in d.scenes):.0f} s)",
        f"- Logline: {d.logline}",
        f"- Concept novelty: max similarity to past stories {r.novelty_max_similarity}",
    ]
    if r.ingredients:
        lines.append(f"- Ingredients: {json.dumps(r.ingredients)}")
    if r.critique:
        lines.append(f"- Critique: {r.critique.scores()} -> {r.critique.verdict}"
                     f"{' (revised once)' if r.revised else ''}")
    if r.problems:
        lines.append("- **Unresolved checks:** " + "; ".join(r.problems))
    lines += ["", "## Narration", ""]
    for n in d.narration:
        cites = f" [claims {n.claim_ids}]" if n.claim_ids else ""
        lines.append(f"{n.id}. ({n.delivery}, pause {n.pause_after_s}s) {n.text}{cites}")
    lines += ["", "## Scenes", ""]
    for s in d.scenes:
        spoken = " / ".join(by_id[i].text for i in s.narration_ids if i in by_id) or "(silence)"
        sfx = ", ".join(f"{c.sound}@{c.at:.1f}" + (f" ({c.note})" if c.note else "") for c in s.sfx) or "-"
        lines += [
            f"### Scene {s.id} - {s.duration_s:g} s - {s.device}",
            f"- Narration: {spoken}",
            f"- Visual: {s.visual}",
            f"- Motion: {s.motion}",
            f"- Camera: {s.camera} | Lighting: {s.lighting}",
            f"- Continuity: {s.continuity}",
            f"- OSD: `{s.osd}` | SFX: {sfx}",
            f"- Video prompt: {s.video_prompt}",
            "",
        ]
    lines += ["## Ending", "", d.ending_note, "", "## Caption draft", "", d.caption, "", r.ai_disclosure, ""]
    if r.claims:
        lines += ["## Claims (quote-verified against the source)", ""]
        lines += [f"- [{c.id}] ({c.tag}) {c.text}  \n  > {c.quote}" for c in r.claims]
        lines.append("")
    if r.sources:
        lines += ["## Sources", ""]
        lines += [f"- [{s.title}]({s.url}) (revision {s.revision_id}, {s.license}, retrieved {s.retrieved_at})"
                  for s in r.sources]
        lines.append("")
    if r.llm:
        lines += ["## Generation stats", ""]
        lines += [f"- {c['name']}: {c['provider']}:{c['model']}, {c['seconds']} s, {c['attempts']} attempt(s), "
                  f"{c['output_tokens']} tokens ({c['tokens_per_s']} tok/s)" for c in r.llm]
    return "\n".join(lines) + "\n"


def save_story(record: StoryRecord) -> Path:
    out = story_dir(record.id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "story.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    (out / "story.md").write_text(to_markdown(record), encoding="utf-8")
    (out / "prompts.json").write_text(json.dumps(scene_prompts(record), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return out


def load_story(story_id: str) -> StoryRecord:
    path = story_dir(story_id) / "story.json"
    if not path.is_file():
        matches = sorted(STORIES_DIR.glob(f"{story_id}*/story.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"story {story_id!r} not found (or ambiguous) in {STORIES_DIR}")
        path = matches[0]
    return StoryRecord.model_validate_json(path.read_text(encoding="utf-8"))

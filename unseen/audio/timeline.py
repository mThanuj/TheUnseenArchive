"""Where every scene and narration line sits in time, from the lines' real spoken lengths.

Scenes follow each other back to back. Within a scene its lines are spoken in order after a short
lead-in, each followed by its scripted pause. When a scene's narration (as actually spoken) needs
more time than planned, the scene is stretched by up to `max_stretch_s` (M4 holds or slows the
clip); anything beyond that is reported as overflow, and the next scene's first line still waits
until the previous line has ended, so lines never overlap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..story.models import StoryDraft
from .config import TimingCfg

LINE_GAP_S = 0.25   # minimum silence between two lines that meet across a scene boundary


@dataclass
class LineSlot:
    id: int
    scene_id: int
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)   # [{"word", "start", "end"}] in Reel time


@dataclass
class SceneSlot:
    id: int
    start: float
    end: float
    planned_s: float
    stretched_s: float          # added to the planned length to fit the narration
    overflow_s: float           # narration that still runs past the scene's end
    line_ids: list[int]


@dataclass
class Timeline:
    scenes: list[SceneSlot]
    lines: list[LineSlot]

    @property
    def total_s(self) -> float:
        return self.scenes[-1].end if self.scenes else 0.0

    def as_dict(self) -> dict:
        return {"total_s": round(self.total_s, 3), "scenes": [asdict(s) for s in self.scenes],
                "lines": [asdict(line) for line in self.lines]}


def build_timeline(draft: StoryDraft, spoken_s: dict[int, float], cfg: TimingCfg,
                   words: dict[int, list[dict]] | None = None) -> Timeline:
    """`spoken_s` = each narration line's audio length in seconds; `words` = per-line word timings
    relative to the line's own start (shifted into Reel time here)."""
    by_id = {n.id: n for n in draft.narration}
    scenes: list[SceneSlot] = []
    lines: list[LineSlot] = []
    t = 0.0
    last_end = -LINE_GAP_S
    for k, s in enumerate(draft.scenes):
        start = t
        ids = [i for i in s.narration_ids if i in by_id and i in spoken_s]
        lead = cfg.first_lead_in_s if k == 0 else cfg.lead_in_s
        cursor = max(start + lead, last_end + LINE_GAP_S)
        for j, i in enumerate(ids):
            line = by_id[i]
            end = cursor + spoken_s[i]
            shifted = [{**w, "start": round(w["start"] + cursor, 3), "end": round(w["end"] + cursor, 3)}
                       for w in (words or {}).get(i, [])]
            lines.append(LineSlot(i, s.id, round(cursor, 3), round(end, 3), line.text, shifted))
            last_end = end
            cursor = end + (line.pause_after_s if j < len(ids) - 1 else 0.0)
        need = (last_end + cfg.tail_s - start) if ids else 0.0
        length = max(s.duration_s, min(need, s.duration_s + cfg.max_stretch_s))
        scenes.append(SceneSlot(s.id, round(start, 3), round(start + length, 3), s.duration_s,
                                round(length - s.duration_s, 3), round(max(0.0, need - length), 3), ids))
        t = start + length
    return Timeline(scenes, lines)

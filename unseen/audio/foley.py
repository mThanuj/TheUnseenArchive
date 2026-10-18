"""Sound from the footage itself: a vision model looks at frames of each rendered clip and lists the
sounds the camera's microphone would have picked up, timed within the clip; brightness jumps
measured from the pixels add light-flicker ticks.

For a rendered scene these cues replace the script's (the script was written before the footage
existed), except its deliberate `silence` / `dropout` choices, which are kept.
"""

from __future__ import annotations

from typing import Callable, Literal, get_args

from pydantic import BaseModel, Field

from ..llm.base import generate
from ..story.models import Sound, SoundName, StoryRecord
from .config import AudioConfig
from .frames import ClipLook
from .sfx import CONTROL
from .soundscape import Cue, script_cues

Log = Callable[[str], None]
MAX_PIXEL_FLICKERS = 4   # per scene: a strobing clip shouldn't become a machine gun of ticks


class FrameCue(BaseModel):
    scene: int = Field(description="scene number, as labelled on its frames")
    sound: Sound
    start_s: float = Field(ge=0, description="seconds from the start of that scene's clip")
    end_s: float | None = Field(default=None, description="beds only: when it stops (default: the clip's end)")
    level: Literal["quiet", "medium", "loud"] = "medium"
    seen: str = Field(default="", description="what in the frames makes this sound, max 12 words")


class FrameCues(BaseModel):
    cues: list[FrameCue]


SYSTEM = ("You are the sound designer of a found-footage horror channel. You hear what a camera's own "
          "microphone would record, nothing more: no music, no narration. Answer with one JSON object.")


def frame_prompt(record: StoryRecord, looks: dict[int, ClipLook]) -> tuple[str, list[tuple[str, bytes]]]:
    vocabulary = ", ".join(sorted(set(get_args(SoundName)) - CONTROL))
    scenes = {s.id: s for s in record.draft.scenes}
    described = []
    for sid, look in sorted(looks.items()):
        s = scenes[sid]
        idea = ", ".join(f"{c.sound}" + (f" ({c.note})" if c.note else "") for c in s.sfx) or "none"
        described.append(f"- scene {sid} ({look.duration_s:.1f} s clip, {s.device}): {s.visual}; motion: {s.motion}; "
                         f"the script's sound idea: {idea}")
    text = (f"Below are frames from {len(looks)} short clips of the Reel '{record.draft.title}', one clip per "
            "scene, each frame labelled with its scene and time. For every scene, list the sounds this footage "
            "itself would carry, based on what is VISIBLE in its frames and its description, timed in seconds "
            "within that scene's clip. Beds (room_tone, hum, wind, rain, fire, crickets, stream, ...) can span the "
            "clip; events (door, footsteps, impact, glass, whoosh, flicker, ...) go where they happen. 1-4 sounds "
            "per scene; never add a sound for something that is not shown or described.\n"
            f"Sound names you may use: {vocabulary}.\nScenes:\n" + "\n".join(described))
    images = [(f"Scene {sid}, frame at {t:.1f} s of {look.duration_s:.1f} s:", jpeg)
              for sid, look in sorted(looks.items()) for t, jpeg in zip(look.frame_times, look.jpegs)]
    return text, images


def cues_from_frames(result: FrameCues, looks: dict[int, ClipLook], cfg: AudioConfig) -> dict[int, list[Cue]]:
    """Model answer -> per-scene cues with positions relative to the clip (0-1), clamped."""
    out: dict[int, list[Cue]] = {}
    for c in result.cues:
        look = looks.get(c.scene)
        if look is None or c.sound in CONTROL or look.duration_s <= 0:
            continue
        at = min(max(c.start_s / look.duration_s, 0.0), 0.98)
        end = min(max(c.end_s / look.duration_s, at), 1.0) if c.end_s is not None else None
        out.setdefault(c.scene, []).append(Cue(c.sound, at, end, cfg.frames.level_db.get(c.level, 0.0),
                                               c.seen, "frames"))
    return out


def pixel_cues(looks: dict[int, ClipLook]) -> dict[int, list[Cue]]:
    """Light flickers measured from the frames' brightness: a short electrical tick at each jump."""
    out: dict[int, list[Cue]] = {}
    for sid, look in looks.items():
        for t in look.flickers[:MAX_PIXEL_FLICKERS]:
            out.setdefault(sid, []).append(Cue("flicker", t / look.duration_s, gain_db=-3.0,
                                               note="brightness jump in the frames", source="pixels"))
    return out


def merge_cues(record: StoryRecord, frames: dict[int, list[Cue]], pixels: dict[int, list[Cue]]) -> dict[int, list[Cue]]:
    """Frame cues where a scene has them (plus the script's silence/dropout), the script's otherwise;
    pixel flickers on top."""
    out: dict[int, list[Cue]] = {}
    for scene in record.draft.scenes:
        if frames.get(scene.id):
            cues = frames[scene.id] + [c for c in script_cues(scene) if c.sound in CONTROL]
        else:
            cues = script_cues(scene)
        out[scene.id] = cues + pixels.get(scene.id, [])
    return out


def ask_frame_sounds(provider, record: StoryRecord, looks: dict[int, ClipLook], cfg: AudioConfig,
                     log: Log = print) -> tuple[dict[int, list[Cue]], dict]:
    """One vision call for the whole story -> ({scene id: cues}, call stats)."""
    text, images = frame_prompt(record, looks)
    result, stats, _ = generate(provider, FrameCues, system=SYSTEM, user=text, name="frame-sounds",
                                temperature=0.3, max_tokens=2500, max_retries=1, log=log, images=images)
    return cues_from_frames(result, looks, cfg), stats.as_dict()

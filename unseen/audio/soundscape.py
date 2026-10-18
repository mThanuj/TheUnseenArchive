"""Turn the script's per-scene sound cues into one ambience track (the footage's own sound).

Every scene gets quiet room tone so the Reel is never digitally silent. Each cue starts at its
position within the scene (0 = start, 1 = end, as the story engine writes them): beds run to the
scene's end, one-shots play once. `silence` mutes the ambience from its cue to the end of the scene
(silence before the reveal); `dropout` cuts it briefly like a tape dropout.
"""

from __future__ import annotations

import math

import numpy as np

from ..story.models import StoryDraft
from . import sfx
from .config import AudioConfig
from .dsp import band, db, fade, place, white
from .timeline import Timeline


def render_soundscape(draft: StoryDraft, timeline: Timeline, cfg: AudioConfig,
                      rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """(ambience track at cfg.sample_rate, [{scene, sound, start, end, note}])."""
    sr = cfg.sample_rate
    n = int(math.ceil(timeline.total_s * sr)) + 1
    track = np.zeros(n, np.float32)
    mask = np.ones(n, np.float32)
    events: list[dict] = []
    level = lambda name: db(cfg.sfx_db.get(name, -6.0))   # noqa: E731
    for scene, slot in zip(draft.scenes, timeline.scenes):
        a, b = int(slot.start * sr), int(slot.end * sr)
        length = slot.end - slot.start
        place(track, fade(sfx.generate("room_tone", length, sr, rng), sr, 0.05, 0.05), a, level("room_tone"))
        for cue in sorted(scene.sfx, key=lambda c: c.at):
            t0 = slot.start + cue.at * length
            i0 = int(t0 * sr)
            if cue.sound == "silence":
                ramp = min(int(0.08 * sr), max(0, b - i0))
                mask[i0:i0 + ramp] *= np.linspace(1, 0, ramp, dtype=np.float32)
                mask[i0 + ramp:b] = 0.0
                end = slot.end
            elif cue.sound == "dropout":
                d = int(cfg.dropout_s * sr)
                mask[i0:i0 + d] = 0.0
                click = band(white(int(0.004 * sr), rng), sr, 2000, 9000) * 0.3
                place(track, click, i0 + d)
                end = t0 + cfg.dropout_s
            elif cue.sound in sfx.BEDS:
                seconds = slot.end - t0
                if seconds < 0.2:
                    continue
                place(track, fade(sfx.generate(cue.sound, seconds, sr, rng), sr, 0.3, 0.15), i0, level(cue.sound))
                end = slot.end
            elif cue.sound in sfx.ONE_SHOTS:
                clip = sfx.generate(cue.sound, 60.0, sr, rng)   # its own natural length
                place(track, clip, i0, level(cue.sound))
                end = t0 + len(clip) / sr
            else:
                continue
            events.append({"scene": scene.id, "sound": cue.sound, "start": round(t0, 2), "end": round(end, 2),
                           "note": cue.note})
    return track * mask, events

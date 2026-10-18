"""The Archivist's voice treatment: a campfire storyteller rather than a dry text-to-speech read.

Built as one ffmpeg filter chain applied to the narration track only:
- pitch down a little (asetrate + atempo keeps the length): deeper, more ominous
- warmth (low-mid boost) and softer highs: close, intimate, like someone across the fire
- gentle compression: even level, every word audible
- a small room (short echoes) around the voice
"""

from __future__ import annotations

from .config import NarratorStyle


def _atempo(factor: float) -> list[str]:
    """atempo accepts 0.5-2.0 per instance in older ffmpeg builds: chain it for safety."""
    out = []
    while factor > 2.0:
        out.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        out.append("atempo=0.5")
        factor /= 0.5
    out.append(f"atempo={factor:.6f}")
    return out


def style_filter(style: NarratorStyle, sr: int) -> str:
    """ffmpeg -af chain for a narrator style ("anull" when the style changes nothing)."""
    parts: list[str] = []
    if style.pitch_semitones:
        ratio = 2 ** (style.pitch_semitones / 12)
        parts += [f"asetrate={sr * ratio:.0f}", f"aresample={sr}", *_atempo(1 / ratio)]
    if style.warmth_db:
        parts.append(f"equalizer=f=160:t=q:w=0.9:g={style.warmth_db:g}")
    if style.air_db:
        parts.append(f"treble=g={style.air_db:g}:f=7000")
    if style.compress:
        parts.append("acompressor=threshold=0.1:ratio=2.5:attack=8:release=150:makeup=1.4")
    if style.room > 0:
        r = min(style.room, 1.0)
        parts.append(f"aecho=1.0:1.0:29|53|89:{0.5 * r:.3f}|{0.35 * r:.3f}|{0.2 * r:.3f}")
    return ",".join(parts) or "anull"

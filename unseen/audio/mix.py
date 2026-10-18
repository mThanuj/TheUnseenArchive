"""Narration + soundscape -> one mix: the ambience ducks under the Archivist's voice."""

from __future__ import annotations

import numpy as np

from .config import MixCfg
from .dsp import db, envelope


def duck_gain(narration: np.ndarray, sr: int, mix: MixCfg) -> np.ndarray:
    """Per-sample gain for the ambience: duck_db down while the narrator speaks, 1.0 otherwise."""
    if not len(narration) or not np.any(narration):
        return np.ones(len(narration), np.float32)
    env = envelope(narration, sr, mix.duck_attack_s, mix.duck_release_s)
    speaking = np.clip(env / 0.15, 0.0, 1.0)   # 15% of the narration's peak level = "speaking"
    return (1.0 - (1.0 - db(mix.duck_db)) * speaking).astype(np.float32)


def mixdown(narration: np.ndarray, ambience: np.ndarray, sr: int, mix: MixCfg) -> np.ndarray:
    n = max(len(narration), len(ambience))
    voice = np.zeros(n, np.float32)
    amb = np.zeros(n, np.float32)
    voice[:len(narration)] = narration
    amb[:len(ambience)] = ambience
    out = voice + amb * duck_gain(voice, sr, mix) * db(mix.ambience_db)
    peak = float(np.max(np.abs(out))) if n else 0.0
    return (out * (db(mix.peak_db) / peak)).astype(np.float32) if peak > 0 else out

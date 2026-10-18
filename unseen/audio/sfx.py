"""Procedural sound effects for the 18 cue names the story engine uses (models.SoundName).

Everything is synthesized with numpy, deterministic for a given seed: no samples to license, no
downloads. Beds (room tone, hum, wind...) run from their cue to the end of the scene; one-shots
(a knock, a door) have a fixed length. Levels are nominal here (beds at -20 dBFS RMS, one-shots
peaking at -6 dBFS); config/audio.yaml sets each sound's level in the mix.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .dsp import at_rms, band, db, decay, fade, pink, place, slow_random, white

BED_RMS_DB = -20.0
SHOT_PEAK_DB = -6.0
TAU = 2 * np.pi

Gen = Callable[[int, int, np.random.Generator], np.ndarray]


def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n, dtype=np.float32) / sr


def _peak(x: np.ndarray, target_db: float = SHOT_PEAK_DB) -> np.ndarray:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (db(target_db) / p)).astype(np.float32) if p > 0 else x


def _tone(freq: np.ndarray | float, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
    """Sine with a constant or time-varying frequency (phase-accumulated)."""
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    return np.sin(phase + TAU * np.cumsum(f) / sr).astype(np.float32)


def _bursts(n: int, sr: int, rng: np.random.Generator, rate_hz: float, length_s: float,
            lo: float, hi: float) -> np.ndarray:
    """Sparse short noise bursts (crackle, pops, clicks)."""
    out = np.zeros(n, np.float32)
    m = max(8, int(length_s * sr))
    for start in np.flatnonzero(rng.random(n) < rate_hz / sr):
        place(out, band(white(m, rng), sr, lo, hi) * decay(m, sr, length_s), int(start), rng.uniform(0.3, 1.0))
    return out


# ------------------------------------------------------------------ beds
def room_tone(n, sr, rng):
    x = band(pink(n, rng), sr, lo=40, hi=2500)
    return at_rms(x * (0.85 + 0.15 * slow_random(n, sr, 0.3, rng)), BED_RMS_DB)


def drone(n, sr, rng):
    f0 = rng.uniform(42, 62)
    x = sum(a * _tone(f0 * k + detune, n, sr, rng.uniform(0, TAU))
            for k, a, detune in ((1, 1.0, 0.0), (1, 0.7, 0.35), (2, 0.4, 0.2), (3, 0.2, 0.1)))
    x = x + 0.5 * band(pink(n, rng), sr, lo=30, hi=300)
    return at_rms(x * (0.55 + 0.45 * slow_random(n, sr, 0.15, rng)), BED_RMS_DB)


def hum(n, sr, rng):
    mains = 50.0
    x = sum(a * _tone(mains * k, n, sr, rng.uniform(0, TAU)) for k, a in ((1, 1.0), (2, 0.5), (3, 0.35), (5, 0.15)))
    return at_rms(x * (1 + 0.05 * np.sin(TAU * 0.7 * _t(n, sr))), BED_RMS_DB)


def electrical_buzz(n, sr, rng):
    x = sum(_tone(100.0 * k, n, sr, rng.uniform(0, TAU)) / k for k in range(1, 25))
    x = band(x, sr, lo=80, hi=6000)
    flicker = 0.75 + 0.25 * (slow_random(n, sr, 8.0, rng) > 0.35)
    return at_rms(x * flicker + 2.0 * _bursts(n, sr, rng, 6.0, 0.004, 2000, 9000), BED_RMS_DB)


def radio_static(n, sr, rng):
    x = band(white(n, rng), sr, lo=300, hi=4500)
    flutter = 0.7 + 0.3 * np.sin(TAU * rng.uniform(8, 14) * _t(n, sr))
    return at_rms(x * flutter + 3.0 * _bursts(n, sr, rng, 3.0, 0.006, 500, 6000), BED_RMS_DB)


def wind(n, sr, rng):
    gust = 0.2 + 0.8 * slow_random(n, sr, 0.25, rng)
    body = band(pink(n, rng), sr, lo=80, hi=1200) * gust
    whistle = band(white(n, rng), sr, lo=600, hi=900, soft=0.1) * gust ** 2 * 0.3
    return at_rms(body + whistle, BED_RMS_DB)


def machine(n, sr, rng):
    motor = sum(_tone(75.0 * k, n, sr, rng.uniform(0, TAU)) / k for k in range(1, 8))
    clicks = np.zeros(n, np.float32)
    m = int(0.006 * sr)
    for start in range(0, n, int(sr / 4)):
        place(clicks, band(white(m, rng), sr, 1500, 8000) * decay(m, sr, 0.006), start)
    return at_rms(motor + 4.0 * clicks + 0.2 * band(white(n, rng), sr, 200, 2000), BED_RMS_DB)


def water_drip(n, sr, rng):
    out = np.zeros(n, np.float32)
    m = int(0.12 * sr)
    tt = _t(m, sr)
    t = rng.uniform(0.1, 0.6)
    while t < n / sr:
        f = 700 + rng.uniform(600, 1100) * np.exp(-tt / 0.03)
        drop = _tone(f, m, sr) * decay(m, sr, 0.09)
        start = int(t * sr)
        for delay_s, gain in ((0.0, 1.0), (0.11, 0.35), (0.23, 0.15)):   # cave-like echoes
            place(out, drop, start + int(delay_s * sr), gain)
        t += rng.uniform(0.7, 1.9)
    return _peak(out)


def breathing(n, sr, rng):
    out = np.zeros(n, np.float32)
    t = 0.2
    while t < n / sr:
        for length, hi in ((1.3, 3000), (1.6, 2200)):   # inhale (brighter), exhale
            m = int(length * sr)
            shape = np.sin(np.linspace(0, np.pi, m)) ** 2
            place(out, band(white(m, rng), sr, 250, hi) * shape, int(t * sr))
            t += length
        t += rng.uniform(0.4, 0.9)
    return _peak(out)


# ------------------------------------------------------------------ one-shots
def footsteps(n, sr, rng):
    out = np.zeros(n, np.float32)
    m = int(0.09 * sr)
    tt = _t(m, sr)
    interval = rng.uniform(0.5, 0.65)
    t = 0.05
    while t < n / sr - 0.1:
        step = band(white(m, rng), sr, hi=900) * decay(m, sr, 0.06) + 0.8 * np.sin(TAU * 75 * tt) * decay(m, sr, 0.05)
        place(out, step, int(t * sr), rng.uniform(0.7, 1.0))
        t += interval * rng.uniform(0.9, 1.1)
    return _peak(out)


def creak(n, sr, rng):
    f = 220 + 120 * slow_random(n, sr, 2.5, rng)
    phase = TAU * np.cumsum(f) / sr
    saw = sum(np.sin(k * phase) / k for k in range(1, 13)).astype(np.float32)
    stick_slip = 0.4 + 0.6 * slow_random(n, sr, 18.0, rng)
    return _peak(fade(band(saw * stick_slip, sr, lo=150, hi=3500), sr, 0.08, 0.15))


def knock(n, sr, rng):
    out = np.zeros(n, np.float32)
    m = int(0.15 * sr)
    tt = _t(m, sr)
    t = 0.05
    for _ in range(3):
        hit = band(white(m, rng), sr, 120, 900) * decay(m, sr, 0.06) + 0.8 * np.sin(TAU * 110 * tt) * decay(m, sr, 0.08)
        place(out, hit, int(t * sr), rng.uniform(0.85, 1.0))
        t += 0.32 * rng.uniform(0.9, 1.1)
    return _peak(out)


def impact(n, sr, rng):
    tt = _t(n, sr)
    boom = _tone(35 + 60 * np.exp(-tt / 0.08), n, sr) * decay(n, sr, 1.0)
    burst = band(white(n, rng), sr, hi=500) * decay(n, sr, 0.25)
    return _peak(boom + 0.6 * burst)


def door(n, sr, rng):
    out = np.zeros(n, np.float32)
    swing = int(0.7 * n)
    place(out, creak(swing, sr, rng), 0, 0.8)
    m = n - swing
    thud = impact(m, sr, rng) * decay(m, sr, 0.35)
    latch = band(white(int(0.01 * sr), rng), sr, 2000, 9000)
    place(out, thud, swing)
    place(out, latch, swing, 0.3)
    return _peak(out)


def _syllables(n: int, sr: int, rng: np.random.Generator, rate_hz: float) -> np.ndarray:
    return np.clip((slow_random(n, sr, rate_hz, rng) - 0.4) * 4, 0, 1)


def whisper(n, sr, rng):
    x = band(white(n, rng), sr, lo=1500, hi=7000) * _syllables(n, sr, rng, 5.0)
    return _peak(fade(x, sr, 0.1, 0.3))


def distant_voice(n, sr, rng):
    f0 = rng.uniform(110, 140) * (1 + 0.04 * slow_random(n, sr, 1.0, rng))
    phase = TAU * np.cumsum(f0) / sr
    source = sum(np.sin(k * phase) / k for k in range(1, 30)).astype(np.float32)
    vowels = sum(band(source, sr, f * 0.85, f * 1.15, soft=0.1) * a for f, a in ((550, 1.0), (1100, 0.6), (2400, 0.3)))
    muffled = band(vowels * _syllables(n, sr, rng, 3.5), sr, hi=700)          # heard through a wall
    room = np.zeros(n, np.float32)
    place(room, muffled, 0)
    place(room, muffled, int(0.06 * sr), 0.3)
    return _peak(fade(room, sr, 0.2, 0.4))


BEDS: dict[str, Gen] = {"room_tone": room_tone, "drone": drone, "hum": hum, "electrical_buzz": electrical_buzz,
                        "radio_static": radio_static, "wind": wind, "machine": machine,
                        "water_drip": water_drip, "breathing": breathing}
ONE_SHOTS: dict[str, tuple[Gen, float]] = {"footsteps": (footsteps, 2.4), "door": (door, 1.6), "knock": (knock, 1.3),
                                           "impact": (impact, 1.8), "creak": (creak, 1.1), "whisper": (whisper, 1.6),
                                           "distant_voice": (distant_voice, 2.2)}
CONTROL = {"silence", "dropout"}   # not sounds: they cut the soundscape


def generate(name: str, seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A bed of `seconds`, or a one-shot (its own length; `seconds` caps it)."""
    if name in BEDS:
        return BEDS[name](max(1, int(seconds * sr)), sr, rng)
    if name in ONE_SHOTS:
        gen, length = ONE_SHOTS[name]
        return gen(max(1, int(min(length, seconds) * sr)), sr, rng)
    raise KeyError(f"no generator for sound {name!r}")

"""Small numpy DSP toolkit for the soundscape and the mix (no scipy needed), plus 16-bit WAV I/O.

Filters work on whole buffers in the frequency domain: Reels are under a minute, so a 48 kHz mono
buffer is ~3M samples and an FFT of it takes milliseconds.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def db(value: float) -> float:
    """Decibels -> linear gain."""
    return float(10 ** (value / 20))


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * np.log10(r) if r > 0 else -120.0


def at_rms(x: np.ndarray, target_db: float) -> np.ndarray:
    """Scale to a target RMS level (dBFS); silence stays silence."""
    current = rms_db(x)
    return x if current <= -119 else (x * db(target_db - current)).astype(np.float32)


def band(x: np.ndarray, sr: int, lo: float | None = None, hi: float | None = None,
         soft: float = 0.25) -> np.ndarray:
    """Band-limit with smooth (raised-cosine) edges `soft` octaves wide."""
    if len(x) == 0:
        return x
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    gain = np.ones_like(f)
    with np.errstate(divide="ignore"):
        octaves = np.log2(np.maximum(f, 1e-6))
    if lo:
        t = np.clip((octaves - (np.log2(lo) - soft)) / soft, 0, 1)
        gain *= 0.5 - 0.5 * np.cos(np.pi * t)
    if hi:
        t = np.clip(((np.log2(hi) + soft) - octaves) / soft, 0, 1)
        gain *= 0.5 - 0.5 * np.cos(np.pi * t)
    return np.fft.irfft(spec * gain, n=len(x)).astype(np.float32)


def white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise: warmer than white, the basis of room tone and wind."""
    if n == 0:
        return np.zeros(0, np.float32)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    out = np.fft.irfft(spec / np.sqrt(f), n=n)
    return (out / (np.std(out) or 1.0)).astype(np.float32)


def slow_random(n: int, sr: int, rate_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random curve in [0, 1] changing about `rate_hz` times per second (gusts, swells)."""
    points = max(2, int(n / sr * rate_hz) + 2)
    knots = rng.random(points)
    curve = np.interp(np.linspace(0, points - 1, n), np.arange(points), knots)
    return np.clip(band(curve.astype(np.float32) - 0.5, sr, hi=max(rate_hz * 2, 0.5)) + 0.5, 0, 1)


def fade(x: np.ndarray, sr: int, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
    y = x.copy()
    a, b = min(len(y), int(fade_in * sr)), min(len(y), int(fade_out * sr))
    if a:
        y[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b:
        y[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return y


def decay(n: int, sr: int, seconds: float) -> np.ndarray:
    """Exponential decay envelope reaching -60 dB after `seconds`."""
    t = np.arange(n, dtype=np.float32) / sr
    return np.exp(-6.9 * t / max(seconds, 1e-3)).astype(np.float32)


def place(track: np.ndarray, clip: np.ndarray, start: int, gain: float = 1.0) -> None:
    """Add `clip` into `track` at sample `start` (clipped to the track's bounds)."""
    if start >= len(track) or len(clip) == 0:
        return
    lo = max(0, start)
    hi = min(len(track), start + len(clip))
    track[lo:hi] += clip[lo - start:hi - start] * gain


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Band-limited (FFT) resampling, e.g. Kokoro's 24 kHz speech to the 48 kHz mix."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x)
    out_bins = n_out // 2 + 1
    if out_bins > len(spec):
        spec = np.concatenate([spec, np.zeros(out_bins - len(spec), dtype=spec.dtype)])
    else:
        spec = spec[:out_bins]
    return (np.fft.irfft(spec, n=n_out) * (n_out / len(x))).astype(np.float32)


def envelope(x: np.ndarray, sr: int, attack_s: float, release_s: float, block_s: float = 0.01) -> np.ndarray:
    """Peak-follower envelope (0..1 of the signal's peak level) with attack/release smoothing."""
    block = max(1, int(sr * block_s))
    n_blocks = (len(x) + block - 1) // block
    padded = np.zeros(n_blocks * block, np.float32)
    padded[:len(x)] = np.abs(x)
    levels = padded.reshape(n_blocks, block).max(axis=1)
    a = np.exp(-block_s / max(attack_s, 1e-3))
    r = np.exp(-block_s / max(release_s, 1e-3))
    out = np.zeros(n_blocks, np.float32)
    current = 0.0
    for i, v in enumerate(levels):   # a few thousand blocks: a Python loop is fine here
        coeff = a if v > current else r
        current = coeff * current + (1 - coeff) * v
        out[i] = current
    peak = float(out.max()) or 1.0
    centers = (np.arange(n_blocks) + 0.5) * block
    return np.interp(np.arange(len(x)), centers, out / peak).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sr: int) -> Path:
    """Mono 16-bit PCM WAV (clipped to [-1, 1])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> (mono float32 in [-1, 1], sample rate)."""
    with wave.open(str(path), "rb") as w:
        sr, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported (got {8 * width}-bit)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, sr

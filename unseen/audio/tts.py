"""Text-to-speech for the Archivist's narration.

`Narrator.synth(text, voice, speed)` returns the line's audio plus word timings (seconds from the
line's start; used for captions in M4). The engine is chosen in config/audio.yaml.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .dsp import db


@dataclass
class TTSResult:
    audio: np.ndarray            # mono float32 in [-1, 1]
    sample_rate: int
    words: list[dict] = field(default_factory=list)   # [{"word", "start", "end"}]


class Narrator(Protocol):
    def synth(self, text: str, voice: str, speed: float) -> TTSResult: ...


SENTENCE_END = re.compile(r"(?<=[.!?])[\"\u201d\u2019)]?\s+")


def sentences(text: str, min_words: int = 3) -> list[str]:
    """Split a narration line into sentences; fragments shorter than `min_words` (e.g. "Dr.") stay
    with the sentence before them."""
    out: list[str] = []
    for piece in (p.strip() for p in SENTENCE_END.split(text.strip())):
        if not piece:
            continue
        if out and len(out[-1].split()) < min_words:
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out or [text]


def synth_line(narrator: "Narrator", text: str, voice: str, speed: float, gap_s: float) -> TTSResult:
    """One narration line, sentence by sentence with an extra beat (gap_s) between sentences: the
    measured cadence of someone telling a story, not a continuous read."""
    parts = sentences(text) if gap_s > 0 else [text]
    if len(parts) == 1:
        return narrator.synth(text, voice, speed)
    audio: list[np.ndarray] = []
    words: list[dict] = []
    sr = 0
    t = 0.0
    for k, part in enumerate(parts):
        r = narrator.synth(part, voice, speed)
        sr = r.sample_rate
        x, offset = trim_silence(r.audio, sr)
        words += [{**w, "start": round(t + max(0.0, w["start"] - offset), 3),
                   "end": round(t + max(0.0, w["end"] - offset), 3)} for w in r.words]
        audio.append(x)
        t += len(x) / sr
        if k < len(parts) - 1:
            audio.append(np.zeros(int(gap_s * sr), np.float32))
            t += gap_s
    return TTSResult(np.concatenate(audio), sr, words)


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -45.0, keep_s: float = 0.03) -> tuple[np.ndarray, float]:
    """Cut the engine's leading/trailing silence (the timeline adds its own, scripted pauses).

    Returns (trimmed audio, seconds removed from the start) so word timings can be shifted."""
    if not len(x):
        return x, 0.0
    loud = np.flatnonzero(np.abs(x) > db(threshold_db) * (float(np.max(np.abs(x))) or 1.0))
    if not len(loud):
        return x[:0], 0.0
    keep = int(keep_s * sr)
    lo = max(0, int(loud[0]) - keep)
    hi = min(len(x), int(loud[-1]) + keep + 1)
    return x[lo:hi], lo / sr


class KokoroNarrator:
    """Kokoro-82M via the `kokoro` package, on the CPU (code and weights Apache-2.0).

    The first use downloads the model (`hexgrad/Kokoro-82M`: config, weights, then each voice used)
    from Hugging Face. English pronunciation comes from misaki, with espeak-ng (GPL-3.0, loaded as a
    library) as the fallback for unknown words. One model is shared by the American ('a*' voices)
    and British ('b*' voices) pipelines.
    """

    SAMPLE_RATE = 24000
    REPO = "hexgrad/Kokoro-82M"

    def __init__(self, device: str = "cpu"):
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("Kokoro TTS is not installed: run `uv sync --extra audio` "
                               "(run_this.ps1 does this)") from exc
        self._pipeline_cls = KPipeline
        for message in ("dropout option adds dropout", r"`torch\.nn\.utils\.weight_norm` is deprecated",
                        r"`torch\.jit\.script` is deprecated"):   # harmless warnings from inside Kokoro
            warnings.filterwarnings("ignore", message=message)
        self.device = device
        self._pipes: dict[str, object] = {}
        self._model = None

    def _pipeline(self, voice: str):
        lang = voice[:1]
        if lang not in ("a", "b"):
            raise ValueError(f"voice {voice!r}: only English voices (a... American, b... British) are supported")
        if lang not in self._pipes:
            pipe = self._pipeline_cls(lang_code=lang, repo_id=self.REPO, device=self.device,
                                      **({"model": self._model} if self._model is not None else {}))
            if getattr(pipe, "model", None) is not None:
                self._model = pipe.model
            self._pipes[lang] = pipe
        return self._pipes[lang]

    def synth(self, text: str, voice: str, speed: float) -> TTSResult:
        chunks: list[np.ndarray] = []
        words: list[dict] = []
        offset = 0.0
        for result in self._pipeline(voice)(text, voice=voice, speed=speed):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            samples = (audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
            samples = samples.astype(np.float32).reshape(-1)
            for tok in getattr(result, "tokens", None) or []:   # timestamps are relative to this chunk
                start, end = getattr(tok, "start_ts", None), getattr(tok, "end_ts", None)
                if start is None or end is None or not any(ch.isalnum() for ch in tok.text):
                    continue
                words.append({"word": tok.text, "start": round(offset + start, 3), "end": round(offset + end, 3)})
            chunks.append(samples)
            offset += len(samples) / self.SAMPLE_RATE
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for {text!r} (voice {voice})")
        return TTSResult(np.concatenate(chunks), self.SAMPLE_RATE, words)


def make_narrator() -> Narrator:
    return KokoroNarrator()

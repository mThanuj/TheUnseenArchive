"""Typed access to config/audio.yaml."""

from __future__ import annotations

from pydantic import BaseModel

from ..config import CONFIG_DIR, load_yaml


class Delivery(BaseModel):
    speed: float = 1.0
    gain_db: float = 0.0


class Audition(BaseModel):
    line: str
    voices: list[str]


class TTSCfg(BaseModel):
    voice: str
    speed: float = 0.88
    sentence_gap_s: float = 0.3
    last_line_speed: float = 0.94
    max_speed: float = 1.0
    delivery: dict[str, Delivery] = {}
    audition: Audition


class NarratorStyle(BaseModel):
    pitch_semitones: float = 0.0   # negative = deeper (the length is kept)
    warmth_db: float = 0.0         # low-mid boost around 160 Hz
    air_db: float = 0.0            # high shelf above ~7 kHz (negative = softer, closer)
    room: float = 0.0              # 0-1: amount of small-room echo around the voice
    compress: bool = False         # gentle voice compression (even, intimate level)
    fire_db: float | None = None   # a crackling fire this many dB under the narration


class NarratorCfg(BaseModel):
    style: str = "dry"
    styles: dict[str, NarratorStyle] = {"dry": NarratorStyle()}

    def current(self, name: str | None = None) -> NarratorStyle:
        key = name or self.style
        if key not in self.styles:
            raise KeyError(f"narrator style {key!r} not in config/audio.yaml (have: {', '.join(self.styles)})")
        return self.styles[key]


class TimingCfg(BaseModel):
    first_lead_in_s: float = 0.15
    lead_in_s: float = 0.35
    tail_s: float = 0.35
    max_stretch_s: float = 1.5


class Loudness(BaseModel):
    integrated_lufs: float = -14.0
    true_peak_db: float = -1.5
    lra: float = 11.0


class MixCfg(BaseModel):
    narration_rms_db: float = -20.0
    ambience_db: float = -10.0
    duck_db: float = -8.0
    duck_attack_s: float = 0.08
    duck_release_s: float = 0.6
    peak_db: float = -1.0
    loudness: Loudness = Loudness()


class FramesCfg(BaseModel):
    enabled: bool = True
    per_clip: int = 4
    max_side: int = 384
    flicker_jump: float = 0.08
    level_db: dict[str, float] = {"quiet": -6.0, "medium": 0.0, "loud": 4.0}


class AudioConfig(BaseModel):
    sample_rate: int = 48000
    tts: TTSCfg
    narrator: NarratorCfg = NarratorCfg()
    timing: TimingCfg = TimingCfg()
    mix: MixCfg = MixCfg()
    sfx_db: dict[str, float] = {}
    dropout_s: float = 0.25
    frames: FramesCfg = FramesCfg()


def load_audio_config() -> AudioConfig:
    return AudioConfig.model_validate(load_yaml(CONFIG_DIR / "audio.yaml"))

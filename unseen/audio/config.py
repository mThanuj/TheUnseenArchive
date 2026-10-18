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
    delivery: dict[str, Delivery] = {}
    audition: Audition


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


class AudioConfig(BaseModel):
    sample_rate: int = 48000
    tts: TTSCfg
    timing: TimingCfg = TimingCfg()
    mix: MixCfg = MixCfg()
    sfx_db: dict[str, float] = {}
    dropout_s: float = 0.25


def load_audio_config() -> AudioConfig:
    return AudioConfig.model_validate(load_yaml(CONFIG_DIR / "audio.yaml"))

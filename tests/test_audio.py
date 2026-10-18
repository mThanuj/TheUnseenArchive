"""Milestone 3 audio tests: CPU only, no TTS model and no ffmpeg (a fake voice, loudnorm stubbed)."""

import json
import shutil
from typing import get_args

import numpy as np
import pytest

from storyfakes import CONCEPTS, draft, draft_dict
from unseen.audio import dsp, sfx
from unseen.audio.config import load_audio_config
from unseen.audio.mix import duck_gain, mixdown
from unseen.audio.soundscape import render_soundscape
from unseen.audio.timeline import build_timeline
from unseen.audio.tts import TTSResult, trim_silence
from unseen.story.models import SfxCue, SoundName, StoryRecord

CFG = load_audio_config()
SLOW = CFG.model_copy(update={"sample_rate": 8000})   # small buffers keep the tests fast


def test_every_cue_name_has_a_generator():
    assert set(get_args(SoundName)) == set(sfx.BEDS) | set(sfx.ONE_SHOTS) | sfx.CONTROL


def test_generators_are_finite_audible_and_deterministic():
    for name in [*sfx.BEDS, *sfx.ONE_SHOTS]:
        a = sfx.generate(name, 2.0, 8000, np.random.default_rng(1))
        b = sfx.generate(name, 2.0, 8000, np.random.default_rng(1))
        assert np.all(np.isfinite(a)) and np.max(np.abs(a)) < 4.0, name
        assert np.array_equal(a, b), name
        assert dsp.rms_db(a) > -60, name
    rng = np.random.default_rng(2)
    assert len(sfx.generate("room_tone", 2.0, 8000, rng)) == 16000          # beds fill the time given
    assert len(sfx.generate("knock", 30.0, 8000, rng)) == int(1.3 * 8000)    # one-shots keep their length
    with pytest.raises(KeyError):
        sfx.generate("silence", 1.0, 8000, rng)


def test_timeline_keeps_lines_in_order_and_stretches_scenes_to_fit():
    d = draft()                                   # 7 scenes x 4 s, lines 1-4 in scenes 1-4, 5-6 in 6-7
    spoken = {n.id: 2.0 for n in d.narration}
    spoken[1] = 6.5                               # far longer than scene 1
    tl = build_timeline(d, spoken, CFG.timing, words={1: [{"word": "Entry", "start": 0.0, "end": 0.4}]})
    first = tl.scenes[0]
    assert first.stretched_s == CFG.timing.max_stretch_s and first.overflow_s == pytest.approx(1.5, abs=0.01)
    assert all(b.start >= a.end + 0.24 for a, b in zip(tl.lines, tl.lines[1:]))   # never overlapping
    assert [ln.id for ln in tl.lines] == [1, 2, 3, 4, 5, 6]
    assert tl.lines[0].start == CFG.timing.first_lead_in_s and tl.lines[0].words[0]["start"] == tl.lines[0].start
    assert tl.scenes[4].line_ids == [] and tl.scenes[4].stretched_s == 0       # silent scene keeps its length
    assert tl.total_s == pytest.approx(sum(s.end - s.start for s in tl.scenes))


def test_soundscape_cues_silence_and_dropout():
    d = draft()
    d.scenes[0].sfx = [SfxCue(at=0.0, sound="hum"), SfxCue(at=0.5, sound="silence")]
    d.scenes[1].sfx = [SfxCue(at=0.0, sound="dropout"), SfxCue(at=0.5, sound="knock")]
    tl = build_timeline(d, {}, CFG.timing)        # no narration: every scene keeps its 4 s
    amb, events = render_soundscape(d, tl, SLOW, np.random.default_rng(3))
    sr = SLOW.sample_rate
    assert np.any(amb[int(0.5 * sr):int(1.5 * sr)])                  # hum + room tone before the silence
    assert not np.any(amb[int(2.2 * sr):int(3.9 * sr)])              # silence until the end of scene 1
    assert not np.any(amb[int(4.0 * sr) + 1:int(4.2 * sr)])          # dropout at the start of scene 2
    assert np.any(amb[int(6.0 * sr):int(6.5 * sr)])                  # the knock
    sounds = [(e["scene"], e["sound"]) for e in events]
    assert (1, "silence") in sounds and (2, "knock") in sounds and (3, "door") in sounds


def test_ambience_ducks_only_while_the_narrator_speaks():
    sr = 8000
    voice = np.zeros(3 * sr, np.float32)
    voice[sr // 2:sr] = 0.3 * np.sin(np.arange(sr // 2) * 0.3)
    g = duck_gain(voice, sr, CFG.mix)
    assert g[int(0.3 * sr)] == pytest.approx(1.0, abs=0.01)
    assert g[int(0.9 * sr)] < dsp.db(CFG.mix.duck_db) + 0.05
    assert g[int(2.9 * sr)] > 0.8                                     # released afterwards
    mix = mixdown(voice, np.full(3 * sr, 0.05, np.float32), sr, CFG.mix)
    assert np.max(np.abs(mix)) == pytest.approx(dsp.db(CFG.mix.peak_db), rel=1e-3)


def test_resample_wav_roundtrip_and_trim(tmp_path):
    t = np.arange(12000) / 24000
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = dsp.resample(x, 24000, 48000)
    assert len(y) == 24000
    ideal = 0.5 * np.sin(2 * np.pi * 440 * np.arange(24000) / 48000)
    assert np.max(np.abs(y[100:-100] - ideal[100:-100])) < 0.01
    path = dsp.write_wav(tmp_path / "x.wav", y, 48000)
    back, sr = dsp.read_wav(path)
    assert sr == 48000 and np.max(np.abs(back - y)) < 1e-4
    padded = np.concatenate([np.zeros(2400, np.float32), x, np.zeros(2400, np.float32)])
    trimmed, offset = trim_silence(padded, 24000)
    assert offset == pytest.approx(0.1 - 0.03, abs=0.002) and len(trimmed) < len(padded)


class FakeVoice:
    """0.3 s of tone per word, with engine-style silence around it."""

    def synth(self, text, voice, speed):
        sr = 24000
        n_words = len(text.split())
        body = (0.2 * np.sin(np.arange(int(n_words * 0.3 / speed * sr)) * 0.05)).astype(np.float32)
        pad = np.zeros(int(0.1 * sr), np.float32)
        words = [{"word": w, "start": 0.1 + i * 0.3 / speed, "end": 0.1 + (i + 1) * 0.3 / speed}
                 for i, w in enumerate(text.split())]
        return TTSResult(np.concatenate([pad, body, pad]), sr, words)


def test_story_audio_end_to_end_with_a_fake_voice(tmp_path, monkeypatch):
    import unseen.audio.pipeline as pipeline
    from unseen.audio.pipeline import render_story_audio

    def fake_loudnorm(src, dst, lufs, tp, lra, sr=48000):
        shutil.copy(src, dst)
        return {"input_i": "-20.1", "input_tp": "-1.0", "input_lra": "3.0", "output_i": "-14.0",
                "output_tp": "-1.6", "output_lra": "3.0", "normalization_type": "linear"}

    monkeypatch.setattr(pipeline.media, "loudnorm", fake_loudnorm)
    record = StoryRecord.model_validate({"id": "20261018-000000-room-nine", "created_at": "2026-10-18T00:00:00",
                                         "category": "fiction_horror", "target_seconds": 30,
                                         "concept": CONCEPTS["concepts"][0], "draft": draft_dict()})
    summary = render_story_audio(record, FakeVoice(), SLOW.model_copy(update={"sample_rate": 16000}),
                                 out_dir=tmp_path, log=lambda m: None)
    for name in ("mix.wav", "narration.wav", "soundscape.wav", "timeline.json", "audio_report.md"):
        assert (tmp_path / name).is_file(), name
    saved = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    assert saved["duration_s"] >= saved["planned_s"] == 28.0
    assert [ln["id"] for ln in saved["timeline"]["lines"]] == [1, 2, 3, 4, 5, 6]
    first = saved["timeline"]["lines"][0]
    assert first["words"][0]["word"] == "Entry" and first["words"][0]["start"] >= first["start"]
    mix, sr = dsp.read_wav(tmp_path / "mix.wav")
    assert sr == 16000 and len(mix) / sr == pytest.approx(saved["duration_s"], abs=0.01)
    report = (tmp_path / "audio_report.md").read_text(encoding="utf-8")
    assert "## Scenes" in report and "-14.0 LUFS" in report

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


def fake_media(monkeypatch):
    """No ffmpeg in tests: the voice treatment and mastering just copy their input."""
    import unseen.audio.pipeline as pipeline

    def fake_master(src, dst, lufs, tp, lra, sr=48000):
        shutil.copy(src, dst)
        return {"input_i": "-15.0", "input_tp": "-1.0", "input_lra": "3.0", "output_i": "-14.0",
                "output_tp": "-1.6", "output_lra": "3.0", "normalization_type": "linear", "pre_master_i": "-21.0"}

    def fake_af(src, dst, af, sr=48000):
        shutil.copy(src, dst)
        return dst

    monkeypatch.setattr(pipeline.media, "master", fake_master)
    monkeypatch.setattr(pipeline.media, "ffmpeg_af", fake_af)


def record_from(d: dict, story_id: str = "20261018-000000-room-nine") -> StoryRecord:
    return StoryRecord.model_validate({"id": story_id, "created_at": "2026-10-18T00:00:00",
                                       "category": "fiction_horror", "target_seconds": 30,
                                       "concept": CONCEPTS["concepts"][0], "draft": d})


def test_story_audio_end_to_end_with_a_fake_voice(tmp_path, monkeypatch):
    from unseen.audio.pipeline import render_story_audio
    fake_media(monkeypatch)
    summary = render_story_audio(record_from(draft_dict()), FakeVoice(), SLOW.model_copy(update={"sample_rate": 16000}),
                                 out_dir=tmp_path, log=lambda m: None, use_frames=False)
    for name in ("mix.wav", "narration.wav", "soundscape.wav", "timeline.json", "audio_report.md"):
        assert (tmp_path / name).is_file(), name
    saved = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    assert saved["duration_s"] >= saved["planned_s"] == 28.0 and saved["style"] == CFG.narrator.style
    assert [ln["id"] for ln in saved["timeline"]["lines"]] == [1, 2, 3, 4, 5, 6]
    first = saved["timeline"]["lines"][0]
    assert first["words"][0]["word"] == "Entry" and first["words"][0]["start"] >= first["start"]
    mix, sr = dsp.read_wav(tmp_path / "mix.wav")
    assert sr == 16000 and len(mix) / sr == pytest.approx(saved["duration_s"], abs=0.01)
    report = (tmp_path / "audio_report.md").read_text(encoding="utf-8")
    assert "## Scenes" in report and "-14.0 LUFS" in report and "script's sound cues" in report
    assert not list(tmp_path.glob("premix*")) and not list(tmp_path.glob("narration_dry*"))


def test_sentences_get_a_storyteller_beat_between_them():
    from unseen.audio.tts import sentences, synth_line
    assert sentences("Entry four twelve. The camera kept recording. Watch.") == [
        "Entry four twelve.", "The camera kept recording.", "Watch."]
    assert sentences("Dr. Hale arrived at nine. Nobody answered.") == ["Dr. Hale arrived at nine.", "Nobody answered."]
    one = synth_line(FakeVoice(), "The tape starts mid-step and keeps going.", "x", 1.0, 0.3)
    two = synth_line(FakeVoice(), "The tape starts. It keeps going on.", "x", 1.0, 0.3)
    assert len(two.audio) / two.sample_rate > len(one.audio) / one.sample_rate   # the beat is extra time
    starts = [w["start"] for w in two.words]
    assert starts == sorted(starts) and starts[3] >= two.words[2]["end"] + 0.29   # "It" after the gap


def test_lines_too_long_for_their_scene_are_respoken_faster(tmp_path, monkeypatch):
    from unseen.audio.pipeline import render_story_audio

    class SpeedVoice(FakeVoice):
        speeds = []

        def synth(self, text, voice, speed):
            SpeedVoice.speeds.append(speed)
            return super().synth(text, voice, speed)

    fake_media(monkeypatch)
    d = draft_dict()
    d["narration"][0]["text"] = " ".join(["word"] * 20)          # ~6.8 s spoken: scene 1 (4 s + 1.5 s) overflows
    cfg = SLOW.model_copy(update={"tts": CFG.tts.model_copy(update={"max_speed": 1.3, "sentence_gap_s": 0.0})})
    summary = render_story_audio(record_from(d), SpeedVoice(), cfg, out_dir=tmp_path, log=lambda m: None,
                                 use_frames=False, style="dry")
    assert summary["respoken"] and summary["respoken"][0].startswith("line 1 re-spoken at x")
    assert max(SpeedVoice.speeds) > cfg.tts.speed and max(SpeedVoice.speeds) <= 1.3


def test_narrator_styles_build_ffmpeg_chains():
    from unseen.audio.voicefx import style_filter
    assert style_filter(CFG.narrator.current("dry"), 48000) == "anull"
    chain = style_filter(CFG.narrator.current("campfire"), 48000)
    for part in ("asetrate=", "aresample=48000", "atempo=", "equalizer=f=160", "treble=", "acompressor=", "aecho="):
        assert part in chain, part
    with pytest.raises(KeyError):
        CFG.narrator.current("opera")


def make_clip(path, fps=12, seconds=2.0, size=(64, 48)):
    """Dark for the first half, then bright: one light flicker at 1.0 s."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write video here")
    n = int(fps * seconds)
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), 20 if i < n // 2 else 200, np.uint8))
    writer.release()
    return path


class FakeVision:
    name, model, supports_images = "gemini", "fake-vision", True

    def __init__(self):
        self.images = None

    def complete_json(self, system, user, schema, temperature, max_tokens, images=None):
        self.images = images
        return json.dumps({"cues": [
            {"scene": 1, "sound": "rain", "start_s": 0.0, "level": "loud", "seen": "rain streaks on the window"},
            {"scene": 1, "sound": "silence", "start_s": 1.0},        # control cues are the script's choice
            {"scene": 9, "sound": "door", "start_s": 0.0},           # no clip for scene 9: ignored
        ]}), {}


def test_sounds_come_from_the_frames_when_the_scene_was_rendered(tmp_path, monkeypatch):
    import unseen.audio.pipeline as pipeline
    from unseen.audio.frames import look_at_clip
    from unseen.audio.pipeline import render_story_audio
    clip = make_clip(tmp_path / "clip.avi")
    look = look_at_clip(clip, 1, per_clip=3)
    assert look.duration_s == pytest.approx(2.0, abs=0.1) and len(look.jpegs) == 3
    assert look.flickers and look.flickers[0] == pytest.approx(1.0, abs=0.1)
    fake_media(monkeypatch)
    monkeypatch.setattr(pipeline, "find_clips", lambda story_id: {1: clip})
    vision = FakeVision()
    out = tmp_path / "audio"
    summary = render_story_audio(record_from(draft_dict()), FakeVoice(), SLOW, out_dir=out, log=lambda m: None,
                                 vision=vision, style="dry")
    assert summary["frames"]["clips"] == 1 and len(vision.images) == CFG.frames.per_clip
    assert vision.images[0][0].startswith("Scene 1, frame at 0.0 s")
    scene1 = [(e["sound"], e["source"]) for e in summary["events"] if e["scene"] == 1]
    assert ("rain", "frames") in scene1 and ("flicker", "pixels") in scene1
    assert ("door", "script") not in scene1                               # the frames replaced the script's idea
    assert ("door", "script") in [(e["sound"], e["source"]) for e in summary["events"] if e["scene"] == 2]
    assert "rain streaks on the window" in (out / "audio_report.md").read_text(encoding="utf-8")

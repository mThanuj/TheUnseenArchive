"""Milestone 3 pipeline: a saved story -> narration + soundscape + loudness-normalised mix.

Outputs in runs/stories/<id>/audio/: mix.wav (listen to this), narration.wav, soundscape.wav,
timeline.json (scene/line/word times for M4's captions and cuts), audio_report.md.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .. import media
from ..story.models import StoryRecord
from ..story.render import story_dir
from .config import AudioConfig
from .dsp import at_rms, db, place, resample, rms_db, write_wav
from .mix import mixdown
from .soundscape import render_soundscape
from .timeline import Timeline, build_timeline
from .tts import Narrator, trim_silence

Log = Callable[[str], None]


def story_seed(story_id: str) -> int:
    """Same story -> same soundscape (re-renders are reproducible)."""
    return int(hashlib.sha256(f"audio:{story_id}".encode()).hexdigest()[:8], 16)


def speak_lines(record: StoryRecord, narrator: Narrator, cfg: AudioConfig, voice: str,
                log: Log = print) -> tuple[dict[int, np.ndarray], dict[int, list[dict]]]:
    """Every narration line at the mix sample rate, levelled; plus word timings (line-relative)."""
    audio: dict[int, np.ndarray] = {}
    words: dict[int, list[dict]] = {}
    for n in record.draft.narration:
        d = cfg.tts.delivery.get(n.delivery)
        speed = cfg.tts.speed * (d.speed if d else 1.0)
        t0 = time.monotonic()
        result = narrator.synth(n.text, voice, speed)
        samples, offset = trim_silence(result.audio, result.sample_rate)
        x = at_rms(resample(samples, result.sample_rate, cfg.sample_rate), cfg.mix.narration_rms_db)
        audio[n.id] = x * db(d.gain_db if d else 0.0)
        words[n.id] = [{**w, "start": max(0.0, w["start"] - offset), "end": max(0.0, w["end"] - offset)}
                       for w in result.words]
        log(f"  line {n.id}: {len(x) / cfg.sample_rate:.1f} s spoken ({time.monotonic() - t0:.1f} s to "
            f"synthesize, {n.delivery}, x{speed:.2f})")
    return audio, words


def render_story_audio(record: StoryRecord, narrator: Narrator, cfg: AudioConfig, voice: str | None = None,
                       out_dir: Path | None = None, log: Log = print) -> dict:
    voice = voice or cfg.tts.voice
    sr = cfg.sample_rate
    out = out_dir or story_dir(record.id) / "audio"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Narrating '{record.draft.title}' ({len(record.draft.narration)} lines, voice {voice})")
    lines, words = speak_lines(record, narrator, cfg, voice, log)
    timeline = build_timeline(record.draft, {i: len(x) / sr for i, x in lines.items()}, cfg.timing, words)

    total = int(np.ceil(timeline.total_s * sr)) + 1
    narration = np.zeros(total, np.float32)
    for slot in timeline.lines:
        place(narration, lines[slot.id], int(slot.start * sr))
    ambience, events = render_soundscape(record.draft, timeline, cfg, np.random.default_rng(story_seed(record.id)))
    log(f"  soundscape: {len(events)} cue(s) over {timeline.total_s:.1f} s")
    premix = mixdown(narration, ambience, sr, cfg.mix)

    write_wav(out / "narration.wav", narration, sr)
    write_wav(out / "soundscape.wav", ambience, sr)
    write_wav(out / "premix.wav", premix, sr)
    lc = cfg.mix.loudness
    loud = media.loudnorm(out / "premix.wav", out / "mix.wav", lc.integrated_lufs, lc.true_peak_db, lc.lra, sr)
    (out / "premix.wav").unlink(missing_ok=True)

    summary = {
        "story_id": record.id, "title": record.draft.title, "voice": voice, "speed": cfg.tts.speed,
        "sample_rate": sr, "duration_s": round(timeline.total_s, 2),
        "planned_s": round(sum(s.duration_s for s in record.draft.scenes), 2),
        "loudness": loud, "narration_rms_db": round(rms_db(narration[narration != 0]), 1) if np.any(narration) else None,
        "events": events, "timeline": timeline.as_dict(),
        "files": {k: str(out / f) for k, f in (("mix", "mix.wav"), ("narration", "narration.wav"),
                                                ("soundscape", "soundscape.wav"), ("timeline", "timeline.json"))},
    }
    (out / "timeline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "audio_report.md").write_text(audio_report(summary, timeline, cfg), encoding="utf-8")
    log(f"Audio written to {out} ({timeline.total_s:.1f} s, {loud.get('output_i')} LUFS, "
        f"{loud.get('output_tp')} dBTP)")
    return summary


def audio_report(summary: dict, timeline: Timeline, cfg: AudioConfig) -> str:
    loud = summary["loudness"]
    stretched = [s for s in timeline.scenes if s.stretched_s > 0]
    overflow = [s for s in timeline.scenes if s.overflow_s > 0]
    lc = cfg.mix.loudness
    lines = [
        f"# Audio: {summary['title']}",
        "",
        f"- Story: `{summary['story_id']}`",
        f"- Voice: {summary['voice']} at x{summary['speed']:g}",
        f"- Duration: {summary['duration_s']} s (scenes planned {summary['planned_s']} s; "
        f"{len(stretched)} scene(s) stretched to fit their narration)",
        f"- Loudness: {loud.get('output_i')} LUFS integrated, {loud.get('output_tp')} dBTP true peak "
        f"(target {lc.integrated_lufs:g} / {lc.true_peak_db:g}; {loud.get('normalization_type')} normalisation; "
        f"input was {loud.get('input_i')} LUFS)",
        f"- Listen: `{summary['files']['mix']}` (narration only: `narration.wav`, sound only: `soundscape.wav`)",
        "",
        "## Scenes",
        "",
        "| scene | start | end | planned | stretched | lines | sounds |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in timeline.scenes:
        sounds = ", ".join(e["sound"] for e in summary["events"] if e["scene"] == s.id) or "room tone"
        lines.append(f"| {s.id} | {s.start:.2f} | {s.end:.2f} | {s.planned_s:g} | {s.stretched_s:g} | "
                     f"{', '.join(map(str, s.line_ids)) or '-'} | {sounds} |")
    lines += ["", "## Narration", "", "| line | start | end | text |", "|---|---|---|---|"]
    lines += [f"| {ln.id} | {ln.start:.2f} | {ln.end:.2f} | {ln.text} |" for ln in timeline.lines]
    if overflow:
        lines += ["", "## Warnings", ""]
        lines += [f"- scene {s.id}: narration runs {s.overflow_s:g} s past the scene even after stretching "
                  f"{s.stretched_s:g} s; shorten its lines" for s in overflow]
    return "\n".join(lines) + "\n"

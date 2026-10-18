"""Milestone 3 pipeline: a saved story -> narration + soundscape + mastered mix.

Outputs in runs/stories/<id>/audio/: mix.wav (listen to this), narration.wav (treated voice),
soundscape.wav, timeline.json (scene/line/word times and every sound event, for M4), and
audio_report.md. When the story's scenes have been rendered, the soundscape comes from the clips'
frames (audio/foley.py) instead of the script's cues.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .. import media
from ..llm.base import LLMError
from ..story.models import NarrationLine, StoryRecord
from ..story.render import story_dir
from . import sfx
from .config import AudioConfig
from .dsp import at_rms, db, place, read_wav, resample, rms_db, write_wav
from .foley import ask_frame_sounds, merge_cues, pixel_cues
from .frames import find_clips, look_at_clip
from .mix import mixdown
from .soundscape import render_soundscape
from .timeline import Timeline, build_timeline
from .tts import Narrator, synth_line, trim_silence
from .voicefx import style_filter

Log = Callable[[str], None]


def story_seed(story_id: str) -> int:
    """Same story -> same soundscape (re-renders are reproducible)."""
    return int(hashlib.sha256(f"audio:{story_id}".encode()).hexdigest()[:8], 16)


def line_speed(n: NarrationLine, cfg: AudioConfig, last: bool) -> float:
    d = cfg.tts.delivery.get(n.delivery)
    return cfg.tts.speed * (d.speed if d else 1.0) * (cfg.tts.last_line_speed if last else 1.0)


def speak(n: NarrationLine, narrator: Narrator, cfg: AudioConfig, voice: str, speed: float,
          log: Log) -> tuple[np.ndarray, list[dict]]:
    """One line at the mix sample rate, levelled, with word timings relative to its start."""
    d = cfg.tts.delivery.get(n.delivery)
    t0 = time.monotonic()
    result = synth_line(narrator, n.text, voice, speed, cfg.tts.sentence_gap_s)
    samples, offset = trim_silence(result.audio, result.sample_rate)
    x = at_rms(resample(samples, result.sample_rate, cfg.sample_rate), cfg.mix.narration_rms_db)
    words = [{**w, "start": round(max(0.0, w["start"] - offset), 3), "end": round(max(0.0, w["end"] - offset), 3)}
             for w in result.words]
    log(f"  line {n.id}: {len(x) / cfg.sample_rate:.1f} s spoken ({time.monotonic() - t0:.1f} s to synthesize, "
        f"{n.delivery}, x{speed:.2f})")
    return x * db(d.gain_db if d else 0.0), words


def fit_overflowing_scenes(record: StoryRecord, timeline: Timeline, lines: dict[int, np.ndarray],
                           words: dict[int, list[dict]], speeds: dict[int, float], narrator: Narrator,
                           cfg: AudioConfig, voice: str, log: Log) -> list[str]:
    """Re-speak the lines of scenes whose narration still runs past them, a little faster (never
    above tts.max_speed). Returns what was changed."""
    by_id = {n.id: n for n in record.draft.narration}
    changed = []
    for s in timeline.scenes:
        if s.overflow_s <= 0.05 or not s.line_ids:
            continue
        spoken = sum(len(lines[i]) for i in s.line_ids) / cfg.sample_rate
        factor = (spoken + s.overflow_s) / spoken
        for i in s.line_ids:
            new = min(cfg.tts.max_speed, speeds[i] * factor)
            if new <= speeds[i] + 0.005:
                continue
            lines[i], words[i] = speak(by_id[i], narrator, cfg, voice, new, log)
            changed.append(f"line {i} re-spoken at x{new:.2f} (was x{speeds[i]:.2f}) to fit scene {s.id}")
            speeds[i] = new
    return changed


def treat_voice(narration: np.ndarray, cfg: AudioConfig, style_name: str | None, out: Path) -> np.ndarray:
    """The narrator style's ffmpeg chain over the whole narration track (length kept exactly)."""
    af = style_filter(cfg.narrator.current(style_name), cfg.sample_rate)
    if af == "anull":
        return narration
    dry = write_wav(out / "narration_dry.wav", narration, cfg.sample_rate)
    try:
        media.ffmpeg_af(dry, out / "narration_fx.wav", af, cfg.sample_rate)
        wet, _ = read_wav(out / "narration_fx.wav")
    finally:
        dry.unlink(missing_ok=True)
        (out / "narration_fx.wav").unlink(missing_ok=True)
    fitted = np.zeros(len(narration), np.float32)
    fitted[:min(len(wet), len(narration))] = wet[:len(narration)]
    return at_rms(fitted, rms_db(narration)) if np.any(fitted) else narration


def frame_sound_cues(record: StoryRecord, cfg: AudioConfig, vision, log: Log) -> tuple[dict | None, dict]:
    """Cues from the rendered clips (None when the story has none), plus what happened, for the report."""
    info: dict = {"clips": 0, "vision": None, "error": None}
    if not cfg.frames.enabled:
        return None, info
    scene_ids = {s.id for s in record.draft.scenes}
    clips = {sid: p for sid, p in find_clips(record.id).items() if sid in scene_ids}
    info["clips"] = len(clips)
    if not clips:
        log("  no rendered clips for this story: sounds come from the script's cues")
        return None, info
    looks = {}
    for sid, path in sorted(clips.items()):
        try:
            looks[sid] = look_at_clip(path, sid, cfg.frames.per_clip, cfg.frames.max_side, cfg.frames.flicker_jump)
        except RuntimeError as exc:
            log(f"  scene {sid}: clip unreadable ({exc})")
    log(f"  looking at {len(looks)} rendered clip(s) ({sum(len(v.jpegs) for v in looks.values())} frames)")
    frames: dict = {}
    if vision is None:
        try:
            from ..story.config import load_story_config
            from ..story.engine import make_provider
            vision = make_provider(load_story_config().llm, log=log)
        except LLMError as exc:
            info["error"] = str(exc)
    if vision is not None:
        try:
            frames, info["vision"] = ask_frame_sounds(vision, record, looks, cfg, log)
        except LLMError as exc:
            info["error"] = str(exc)[:300]
            log(f"  frame sounds unavailable ({str(exc)[:200]}); using the script's cues + measured flickers")
    return merge_cues(record, frames, pixel_cues(looks)), info


def render_story_audio(record: StoryRecord, narrator: Narrator, cfg: AudioConfig, voice: str | None = None,
                       out_dir: Path | None = None, log: Log = print, style: str | None = None,
                       vision=None, use_frames: bool = True) -> dict:
    voice = voice or cfg.tts.voice
    style_name = style or cfg.narrator.style
    narrator_style = cfg.narrator.current(style_name)
    sr = cfg.sample_rate
    out = out_dir or story_dir(record.id) / "audio"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Narrating '{record.draft.title}' ({len(record.draft.narration)} lines, voice {voice}, style {style_name})")
    last_id = record.draft.narration[-1].id if record.draft.narration else None
    lines: dict[int, np.ndarray] = {}
    words: dict[int, list[dict]] = {}
    speeds: dict[int, float] = {}
    for n in record.draft.narration:
        speeds[n.id] = line_speed(n, cfg, n.id == last_id)
        lines[n.id], words[n.id] = speak(n, narrator, cfg, voice, speeds[n.id], log)
    timeline = build_timeline(record.draft, {i: len(x) / sr for i, x in lines.items()}, cfg.timing, words)
    respoken = fit_overflowing_scenes(record, timeline, lines, words, speeds, narrator, cfg, voice, log)
    if respoken:
        timeline = build_timeline(record.draft, {i: len(x) / sr for i, x in lines.items()}, cfg.timing, words)

    total = int(np.ceil(timeline.total_s * sr)) + 1
    narration = np.zeros(total, np.float32)
    for slot in timeline.lines:
        place(narration, lines[slot.id], int(slot.start * sr))
    narration = treat_voice(narration, cfg, style_name, out)

    rng = np.random.default_rng(story_seed(record.id))
    cues, frame_info = frame_sound_cues(record, cfg, vision, log) if use_frames else (None, {"clips": 0})
    ambience, events = render_soundscape(record.draft, timeline, cfg, rng, cues)
    log(f"  soundscape: {len(events)} sound(s) over {timeline.total_s:.1f} s")
    under = None
    if narrator_style.fire_db is not None:
        under = at_rms(sfx.generate("fire", timeline.total_s, sr, rng), cfg.mix.narration_rms_db + narrator_style.fire_db)
    premix = mixdown(narration, ambience, sr, cfg.mix, under)

    write_wav(out / "narration.wav", narration, sr)
    write_wav(out / "soundscape.wav", ambience, sr)
    write_wav(out / "premix.wav", premix, sr)
    lc = cfg.mix.loudness
    try:
        loud = media.master(out / "premix.wav", out / "mix.wav", lc.integrated_lufs, lc.true_peak_db, lc.lra, sr)
    finally:
        (out / "premix.wav").unlink(missing_ok=True)

    summary = {
        "story_id": record.id, "title": record.draft.title, "voice": voice, "style": style_name,
        "speed": cfg.tts.speed, "sample_rate": sr, "duration_s": round(timeline.total_s, 2),
        "planned_s": round(sum(s.duration_s for s in record.draft.scenes), 2), "loudness": loud,
        "narration_rms_db": round(rms_db(narration[narration != 0]), 1) if np.any(narration) else None,
        "respoken": respoken, "frames": frame_info, "events": events, "timeline": timeline.as_dict(),
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
    fr = summary.get("frames") or {}
    if fr.get("clips"):
        source = (f"{fr['clips']} rendered clip(s): sounds from their frames"
                  + (f" ({fr['vision']['provider']}:{fr['vision']['model']})" if fr.get("vision") else "")
                  + (f"; vision failed: {fr['error']}" if fr.get("error") else ""))
    else:
        source = "no rendered clips: the script's sound cues"
    lines = [
        f"# Audio: {summary['title']}",
        "",
        f"- Story: `{summary['story_id']}`",
        f"- Voice: {summary['voice']} at x{summary['speed']:g}, narrator style **{summary['style']}**",
        f"- Duration: {summary['duration_s']} s (scenes planned {summary['planned_s']} s; "
        f"{len(stretched)} scene(s) stretched to fit their narration)",
        f"- Loudness: {loud.get('output_i')} LUFS integrated, {loud.get('output_tp')} dBTP true peak "
        f"(target {lc.integrated_lufs:g} / {lc.true_peak_db:g}; {loud.get('normalization_type')} normalisation; "
        f"before mastering {loud.get('pre_master_i', loud.get('input_i'))} LUFS)",
        f"- Sound: {source}",
        f"- Listen: `{summary['files']['mix']}` (narration only: `narration.wav`, sound only: `soundscape.wav`)",
    ]
    if summary.get("respoken"):
        lines += [f"- {r}" for r in summary["respoken"]]
    lines += ["", "## Scenes", "",
              "| scene | start | end | planned | stretched | lines | sounds |", "|---|---|---|---|---|---|---|"]
    for s in timeline.scenes:
        sounds = ", ".join(e["sound"] + ("*" if e.get("source") == "frames" else "")
                           for e in summary["events"] if e["scene"] == s.id) or "room tone"
        lines.append(f"| {s.id} | {s.start:.2f} | {s.end:.2f} | {s.planned_s:g} | {s.stretched_s:g} | "
                     f"{', '.join(map(str, s.line_ids)) or '-'} | {sounds} |")
    if any(e.get("source") == "frames" for e in summary["events"]):
        lines += ["", "\\* chosen from the clip's frames"]
        lines += [f"- scene {e['scene']} {e['sound']} at {e['start']:.1f} s: {e['note']}"
                  for e in summary["events"] if e.get("source") == "frames" and e.get("note")]
    lines += ["", "## Narration", "", "| line | start | end | text |", "|---|---|---|---|"]
    lines += [f"| {ln.id} | {ln.start:.2f} | {ln.end:.2f} | {ln.text} |" for ln in timeline.lines]
    if overflow:
        lines += ["", "## Warnings", ""]
        lines += [f"- scene {s.id}: narration runs {s.overflow_s:g} s past the scene even after stretching "
                  f"{s.stretched_s:g} s; shorten its lines" for s in overflow]
    return "\n".join(lines) + "\n"

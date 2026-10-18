"""Thin wrappers around the system ffmpeg/ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


def which(tool: str) -> str | None:
    return shutil.which(tool)


def require(tool: str) -> str:
    exe = which(tool)
    if not exe:
        raise RuntimeError(f"{tool} not found on PATH. Install FFmpeg and reopen the terminal.")
    return exe


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{Path(cmd[0]).name} failed (exit {exc.returncode}): "
                           f"{(exc.stderr or '').strip()[-1500:]}") from None


def encode_frames(frames_dir: Path, fps: float, out_path: Path, crf: int = 14) -> Path:
    """Encode frame_00001.png, frame_00002.png, ... into a near-lossless H.264 MP4."""
    _run([
        require("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:g}",
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ])
    return out_path


def probe(path: Path) -> dict:
    """Return the key properties of the first video stream."""
    out = _run([
        require("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-count_frames", str(path),
    ])
    data = json.loads(out)
    video = next(s for s in data["streams"] if s.get("codec_type") == "video")
    fps = float(Fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"))
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return {
        "codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": round(fps, 3),
        "frames": frames,
        "duration_s": round(duration, 3),
        "size_bytes": int(data.get("format", {}).get("size") or 0),
    }


def parse_loudnorm(stderr: str) -> dict:
    """The JSON block ffmpeg's loudnorm filter prints (print_format=json) at the end of stderr."""
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end < start:
        raise RuntimeError("ffmpeg loudnorm printed no measurement")
    return json.loads(stderr[start:end + 1])


def _ffmpeg_stderr(args: list[str]) -> str:
    proc = subprocess.run([require("ffmpeg"), "-hide_banner", "-nostats", *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.strip()[-1500:]}")
    return proc.stderr


def loudnorm(src: Path, dst: Path, lufs: float, true_peak_db: float, lra: float, sr: int = 48000) -> dict:
    """Two-pass EBU R128 loudness normalisation (measure, then apply linearly) to a 16-bit WAV.

    Returns {"input_i", "input_tp", "output_i", "output_tp", ...} as reported by ffmpeg."""
    target = f"I={lufs}:TP={true_peak_db}:LRA={lra}"
    first = parse_loudnorm(_ffmpeg_stderr(["-i", str(src), "-af", f"loudnorm={target}:print_format=json",
                                           "-f", "null", "-"]))
    measured = (f"measured_I={first['input_i']}:measured_TP={first['input_tp']}:"
                f"measured_LRA={first['input_lra']}:measured_thresh={first['input_thresh']}:"
                f"offset={first['target_offset']}:linear=true")
    second = parse_loudnorm(_ffmpeg_stderr(["-y", "-i", str(src), "-af",
                                            f"loudnorm={target}:{measured}:print_format=json",
                                            "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]))
    return {**{k: first[k] for k in ("input_i", "input_tp", "input_lra")},
            **{k: second[k] for k in ("output_i", "output_tp", "output_lra", "normalization_type")}}


def ffmpeg_af(src: Path, dst: Path, af: str, sr: int = 48000) -> Path:
    """Run an audio filter chain over a WAV: mono 16-bit PCM at `sr` out."""
    _ffmpeg_stderr(["-y", "-i", str(src), "-af", af, "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(dst)])
    return dst


def master(src: Path, dst: Path, lufs: float, true_peak_db: float, lra: float, sr: int = 48000) -> dict:
    """Mastering for a Reel: gentle compression, gain towards the target with a peak limiter below
    the true-peak ceiling, then loudnorm (two passes). Limiting first lets loudnorm stay linear
    instead of falling back to its dynamic mode (which missed -14 LUFS by 1.9 LU on spiky speech)."""
    comp = dst.with_name(dst.stem + "_comp.wav")
    lim = dst.with_name(dst.stem + "_lim.wav")
    try:
        ffmpeg_af(src, comp, "acompressor=threshold=0.1:ratio=2.5:attack=10:release=200", sr)
        measured = parse_loudnorm(_ffmpeg_stderr(["-i", str(comp), "-af",
                                                  f"loudnorm=I={lufs}:TP={true_peak_db}:LRA={lra}:print_format=json",
                                                  "-f", "null", "-"]))
        gain = lufs - float(measured["input_i"]) + 0.3
        ceiling = 10 ** ((true_peak_db - 0.7) / 20)   # sample-peak ceiling with room for inter-sample peaks
        ffmpeg_af(comp, lim, f"volume={gain:.2f}dB,alimiter=limit={ceiling:.4f}:attack=3:release=60:level=0", sr)
        result = loudnorm(lim, dst, lufs, true_peak_db, lra, sr)
        result["pre_master_i"] = measured["input_i"]
        return result
    finally:
        comp.unlink(missing_ok=True)
        lim.unlink(missing_ok=True)

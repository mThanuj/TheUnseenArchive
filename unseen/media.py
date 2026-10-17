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

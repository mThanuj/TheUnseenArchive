"""Look at a story's rendered clips: which scenes have footage, a few frames of each to show a
vision model, and brightness jumps (a light flickering, a flash) measured directly from the pixels.

Clips are found where `unseen story render` leaves them: bench_results/<run>/<case>_<hash>_sNN/
<case>.mp4, where <hash>_sNN is the story's scene prompt id (story/render.py: scene_prompts). The
newest run wins when a scene was rendered more than once.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import load_runtime


def story_hash(story_id: str) -> str:
    return hashlib.sha1(story_id.encode()).hexdigest()[:6]   # the same short id as story/render.py


def find_clips(story_id: str, bench_dir: Path | None = None) -> dict[int, Path]:
    """{scene id: newest rendered clip} for this story (empty if it was never rendered)."""
    root = bench_dir or load_runtime().path("bench_dir")
    short = story_hash(story_id)
    found: dict[int, Path] = {}
    for mp4 in sorted(root.glob(f"*/*_{short}_s*/*.mp4"), key=lambda p: (p.parent.parent.name, p.name)):
        m = re.search(rf"_{short}_s(\d+)$", mp4.parent.name)
        if m:
            found[int(m.group(1))] = mp4   # later (newer) runs overwrite older ones
    return found


@dataclass
class ClipLook:
    scene_id: int
    path: Path
    duration_s: float
    frame_times: list[float] = field(default_factory=list)      # seconds into the clip, for the frames below
    jpegs: list[bytes] = field(default_factory=list)
    brightness: list[float] = field(default_factory=list)       # mean luma per frame, 0-1
    motion: list[float] = field(default_factory=list)           # mean abs change from the previous frame, 0-1
    flickers: list[float] = field(default_factory=list)         # seconds into the clip where the light jumps


def look_at_clip(path: Path, scene_id: int, per_clip: int = 4, max_side: int = 384,
                 flicker_jump: float = 0.08) -> ClipLook:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"{path} has no readable frames")
    look = ClipLook(scene_id, path, len(frames) / fps)
    small = [cv2.cvtColor(cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
             .astype(np.float32) / 255 for f in frames]
    look.brightness = [round(float(g.mean()), 4) for g in small]
    look.motion = [0.0] + [round(float(np.abs(b - a).mean()), 4) for a, b in zip(small, small[1:])]
    last = -1.0
    for i in range(1, len(small)):
        t = i / fps
        if abs(look.brightness[i] - look.brightness[i - 1]) >= flicker_jump and t - last > 0.15:
            look.flickers.append(round(t, 3))
            last = t
    picks = sorted({int(round(x)) for x in np.linspace(0, len(frames) - 1, max(1, per_clip))})
    for i in picks:
        f = frames[i]
        h, w = f.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            f = cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            look.frame_times.append(round(i / fps, 2))
            look.jpegs.append(buf.tobytes())
    return look

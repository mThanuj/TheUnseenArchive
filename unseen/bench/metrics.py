"""Objective checks on generated frames: is there genuine motion, and what does it look like?

"Local motion" is the optical flow left over after removing the best-fit global similarity
transform (zoom + pan + rotation) between consecutive frames. A Ken Burns move over a still
image is fully explained by that transform, so its local motion is ~0; objects moving, doors
opening, figures walking, or real 3D parallax from a moving camera are not. This is the
measurable form of the project's "no slideshows" requirement.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_frames(paths: list[Path]) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def _gray_small(frame: np.ndarray, width: int) -> np.ndarray:
    h = max(16, round(frame.shape[0] * width / frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, (width, h), interpolation=cv2.INTER_AREA)


def motion_metrics(frames: list[np.ndarray], width: int = 256, grid_step: int = 8,
                   border: float = 0.06) -> dict:
    """Per-clip motion statistics in pixels/frame at `width` px wide.

    Flow is sampled on a grid that skips a `border` fraction at each edge, where content
    entering the frame makes optical flow unreliable.
    """
    if len(frames) < 2:
        return {"pairs": 0, "flow_mean": 0.0, "local_motion_mean": 0.0, "local_motion_p98": 0.0,
                "frame_diff_mean": 0.0, "duplicate_frame_ratio": 1.0}
    grays = [_gray_small(f, width) for f in frames]
    h, w = grays[0].shape
    mx, my = max(grid_step, round(border * w)), max(grid_step, round(border * h))
    ys, xs = np.mgrid[my:h - my:grid_step, mx:w - mx:grid_step]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    iy, ix = pts[:, 1].astype(int), pts[:, 0].astype(int)

    flow_means, local_means, local_p98s, diffs = [], [], [], []
    for a, b in zip(grays, grays[1:]):
        diffs.append(float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))))
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vec = flow[iy, ix]
        mag = np.linalg.norm(vec, axis=1)
        flow_means.append(float(mag.mean()))
        dst = pts + vec
        m, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if m is None:
            resid = mag
        else:
            pred = pts @ m[:, :2].T + m[:, 2]
            resid = np.linalg.norm(dst - pred, axis=1)
        local_means.append(float(resid.mean()))
        local_p98s.append(float(np.percentile(resid, 98)))

    d = np.asarray(diffs)
    return {
        "pairs": len(diffs),
        "flow_mean": round(float(np.mean(flow_means)), 4),
        "local_motion_mean": round(float(np.mean(local_means)), 4),
        "local_motion_p98": round(float(np.mean(local_p98s)), 4),
        "frame_diff_mean": round(float(d.mean()), 4),
        "duplicate_frame_ratio": round(float(np.mean(d < 0.5)), 4),
    }


def contact_sheet(frames: list[np.ndarray], out_path: Path, n: int = 8, cols: int = 4,
                  thumb_w: int = 240) -> Path:
    """Evenly spaced frames tiled into one small JPEG for quick visual review."""
    idx = np.unique(np.linspace(0, len(frames) - 1, num=min(n, len(frames))).round().astype(int))
    h0, w0 = frames[0].shape[:2]
    th = round(h0 * thumb_w / w0)
    label_h = 18
    rows = math.ceil(len(idx) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (th + label_h)), "black")
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        x, y = (k % cols) * thumb_w, (k // cols) * (th + label_h)
        sheet.paste(Image.fromarray(frames[i]).resize((thumb_w, th), Image.Resampling.LANCZOS),
                    (x, y + label_h))
        draw.text((x + 4, y + 3), f"frame {i + 1}/{len(frames)}", fill=(220, 220, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=85)
    return out_path

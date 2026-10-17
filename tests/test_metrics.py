"""The motion metric must separate genuine motion from a Ken Burns move over a still image."""

import cv2
import numpy as np

from unseen.bench.metrics import contact_sheet, motion_metrics

W, H, N = 256, 448, 16


def _texture(seed: int, w: int = W, h: int = H) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return np.dstack([img] * 3)


def static_clip():
    frame = _texture(1)
    return [frame.copy() for _ in range(N)]


def ken_burns_clip():
    """Slow zoom + pan over one still image (exact similarity warps): global motion only."""
    big = _texture(2, W * 2, H * 2)
    frames = []
    for i in range(N):
        s = 0.6 * (1.0 + 0.01 * i)                  # zoom in 1% per frame
        tx, ty = -20.0 - 1.2 * i, -30.0 - 0.6 * i   # plus a slow pan
        m = np.float32([[s, 0, tx], [0, s, ty]])
        frames.append(cv2.warpAffine(big, m, (W, H), flags=cv2.INTER_LINEAR))
    return frames


def moving_object_clip():
    """Static background with a textured square moving 3 px/frame: local motion."""
    bg = _texture(3)
    patch = _texture(4, 64, 64)
    frames = []
    for i in range(N):
        f = bg.copy()
        x, y = 40 + 3 * i, 150 + i
        f[y:y + 64, x:x + 64] = patch
        frames.append(f)
    return frames


def test_static_clip_has_no_motion():
    m = motion_metrics(static_clip())
    assert m["duplicate_frame_ratio"] == 1.0
    assert m["flow_mean"] < 0.05
    assert m["local_motion_p98"] < 0.05


def test_ken_burns_is_global_not_local():
    m = motion_metrics(ken_burns_clip())
    assert m["flow_mean"] > 0.4            # plenty of apparent movement...
    assert m["local_motion_p98"] < 0.30    # ...but explained by zoom/pan


def test_moving_object_is_local_motion():
    m = motion_metrics(moving_object_clip())
    kb = motion_metrics(ken_burns_clip())
    assert m["local_motion_p98"] >= 0.30
    assert m["local_motion_p98"] > 3 * kb["local_motion_p98"]
    assert m["duplicate_frame_ratio"] == 0.0


def test_single_frame_is_handled():
    m = motion_metrics(static_clip()[:1])
    assert m["pairs"] == 0


def test_contact_sheet(tmp_path):
    out = contact_sheet(moving_object_clip(), tmp_path / "sheet.jpg")
    assert out.is_file() and out.stat().st_size > 1000

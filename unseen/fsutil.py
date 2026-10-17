"""Filesystem helpers that tolerate Windows' transient file locks."""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def move_with_retry(src: Path, dst: Path, attempts: int = 6, delay_s: float = 0.5) -> Path:
    """shutil.move, retried while antivirus/indexers briefly hold a fresh file (WinError 32)."""
    for attempt in range(attempts):
        try:
            shutil.move(str(src), str(dst))
            return dst
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s * (attempt + 1))
    return dst

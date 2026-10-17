"""Download the model weights listed in config/models.yaml and verify their sha256."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable

from .config import ModelsManifest, RuntimeConfig
from .fsutil import move_with_retry

Log = Callable[[str], None]


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _marker(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def is_verified(path: Path, sha256: str, size: int) -> bool:
    marker = _marker(path)
    return (
        path.is_file()
        and path.stat().st_size == size
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == sha256
    )


def missing_bytes(manifest: ModelsManifest, rt: RuntimeConfig, keys: list[str]) -> int:
    total = 0
    for key in keys:
        f = manifest.files[key]
        if not is_verified(manifest.local_path(rt, key), f.sha256, f.size):
            total += f.size
    return total


def fetch_file(manifest: ModelsManifest, rt: RuntimeConfig, key: str, log: Log = print) -> Path:
    from huggingface_hub import hf_hub_download  # heavy import, only needed here

    f = manifest.files[key]
    dest = manifest.local_path(rt, key)
    if is_verified(dest, f.sha256, f.size):
        log(f"  ok       {key} ({dest.name})")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.is_file() and dest.stat().st_size == f.size):
        log(f"  download {key}: {f.repo}/{f.path} ({f.size / 1e9:.2f} GB)")
        staging = rt.path("models_dir") / ".downloads" / f.repo.replace("/", "__")
        downloaded = Path(hf_hub_download(repo_id=f.repo, filename=f.path, local_dir=staging))
        if dest.exists():
            dest.unlink()
        move_with_retry(downloaded, dest)

    log(f"  verify   {key} (sha256, may take a minute)")
    digest = sha256_file(dest)
    if digest != f.sha256:
        bad = dest.with_name(dest.name + ".corrupt")
        dest.replace(bad)
        raise RuntimeError(
            f"sha256 mismatch for {key}: expected {f.sha256}, got {digest}. "
            f"Moved to {bad}; delete it and re-run fetch-models."
        )
    _marker(dest).write_text(digest, encoding="utf-8")
    log(f"  ok       {key}")
    return dest


def fetch_set(manifest: ModelsManifest, rt: RuntimeConfig, set_name: str, log: Log = print) -> list[Path]:
    if set_name not in manifest.sets:
        raise KeyError(f"unknown model set {set_name!r}; choose from {sorted(manifest.sets)}")
    keys = manifest.sets[set_name]

    log(f"Model set {set_name!r}: licenses")
    for lic_key in sorted({manifest.files[k].license for k in keys}):
        lic = manifest.licenses[lic_key]
        log(f"  - {lic.name}: commercial use {lic.commercial_use}")
        if lic.obligations:
            log(f"    obligations: {lic.obligations}")
        log(f"    {lic.url}")

    need = missing_bytes(manifest, rt, keys)
    models_dir = rt.path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(models_dir).free
    margin = 2 * 1024**3
    if need + margin > free:
        raise RuntimeError(
            f"not enough disk space in {models_dir}: need {need / 1e9:.1f} GB + 2 GB margin, "
            f"have {free / 1e9:.1f} GB free"
        )
    log(f"To download: {need / 1e9:.2f} GB (free: {free / 1e9:.1f} GB)")
    return [fetch_file(manifest, rt, k, log) for k in keys]

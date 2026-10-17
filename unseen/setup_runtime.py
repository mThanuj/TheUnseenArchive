"""Install the pinned video runtime: ComfyUI + custom nodes + a dedicated CUDA PyTorch venv.

Invoked by scripts/windows/setup.ps1 as `unseen setup-comfy`. Safe to re-run: existing pieces
are verified and reused; a checkout with local modifications is never overwritten.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .config import RuntimeConfig

Log = Callable[[str], None]

TORCH_PROBE = (
    "import json, torch; ok = torch.cuda.is_available(); "
    "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': ok, "
    "'device': torch.cuda.get_device_name(0) if ok else None, "
    "'vram_mb': round(torch.cuda.get_device_properties(0).total_memory / 2**20) if ok else None}))"
)


def _run(cmd: list[str], log: Log, cwd: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_checkout(repo: str, commit: str, dest: Path, log: Log) -> None:
    """Make `dest` a detached checkout of `commit`, cloning (blobless) if needed."""
    if not (dest / ".git").is_dir():
        if dest.exists() and any(dest.iterdir()):
            raise RuntimeError(f"{dest} exists but is not a git checkout; move it away and re-run")
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(dest)], log)
    # Nothing checked out yet = fresh (or interrupted) --no-checkout clone; git status would
    # report every file as deleted, so only check for local edits in a real working tree.
    checked_out = any(p.name != ".git" for p in dest.iterdir())
    if checked_out and git_head(dest) == commit:
        log(f"  ok: {dest.name} at {commit[:10]}")
        return
    if checked_out:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=dest,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            raise RuntimeError(f"{dest} has local modifications; refusing to switch it to {commit[:10]}")
    _run(["git", "fetch", "--filter=blob:none", "origin", commit], log, cwd=dest)
    _run(["git", "-c", "advice.detachedHead=false", "checkout", "--detach", commit], log, cwd=dest)
    log(f"  ok: {dest.name} at {commit[:10]}")


def torch_info(python: Path) -> dict:
    try:
        out = subprocess.run([str(python), "-c", TORCH_PROBE], capture_output=True, text=True,
                             timeout=300, check=True).stdout.strip().splitlines()[-1]
        return json.loads(out)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def expected_cuda_major(index_url: str) -> str | None:
    m = re.search(r"cu(\d+)", index_url)
    return m.group(1)[:-1] if m else None   # cu130 -> "13", cu126 -> "12"


def torch_ok(info: dict, index_url: str) -> bool:
    major = expected_cuda_major(index_url)
    return bool(info.get("available")) and (major is None or str(info.get("cuda") or "").startswith(major + "."))


def _ver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


def driver_cuda_version() -> str | None:
    """CUDA version supported by the installed NVIDIA driver, from the `nvidia-smi` header."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def resolve_torch_index(rt: RuntimeConfig, driver_cuda: str | None = None) -> str:
    """`torch_index_url: auto` -> newest wheel index the driver supports; explicit URLs pass through."""
    setting = rt.comfyui.torch_index_url
    if setting != "auto":
        return setting
    driver_cuda = driver_cuda or driver_cuda_version()
    if not driver_cuda:
        raise RuntimeError("cannot read the driver's CUDA version from `nvidia-smi`; install the NVIDIA driver")
    for option in sorted(rt.comfyui.torch_index_by_driver, key=lambda o: _ver(o.min_cuda), reverse=True):
        if _ver(driver_cuda) >= _ver(option.min_cuda):
            return option.url
    raise RuntimeError(f"NVIDIA driver supports only CUDA {driver_cuda}; update the driver")


def setup_comfy(rt: RuntimeConfig, log: Log = print) -> dict:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv not found on PATH (scripts\\windows\\setup.ps1 installs it)")
    if not shutil.which("git"):
        raise RuntimeError("git not found on PATH; install Git for Windows and reopen the terminal")

    comfy_dir = rt.path("comfyui_dir")
    log(f"ComfyUI {rt.comfyui.ref} -> {comfy_dir}")
    ensure_checkout(rt.comfyui.repo, rt.comfyui.commit, comfy_dir, log)
    for node in rt.custom_nodes:
        log(f"Custom node {node.name}")
        ensure_checkout(node.repo, node.commit, comfy_dir / "custom_nodes" / node.name, log)

    py = rt.comfy_python()
    index = resolve_torch_index(rt)
    log(f"PyTorch wheel index: {index}")
    if not py.is_file():
        log(f"Creating ComfyUI venv (Python {rt.comfyui.python})")
        _run([uv, "venv", str(rt.path("comfy_venv")), "--python", rt.comfyui.python], log)

    info = torch_info(py)
    if not torch_ok(info, index):
        log(f"Installing CUDA PyTorch from {index} (~3 GB download)")
        _run([uv, "pip", "install", "--python", str(py),
              "torch", "torchvision", "torchaudio", "--index-url", index], log)

    log("Installing ComfyUI requirements")
    _run([uv, "pip", "install", "--python", str(py), "-r", str(comfy_dir / "requirements.txt")], log)
    for node in rt.custom_nodes:
        req = comfy_dir / "custom_nodes" / node.name / "requirements.txt"
        if req.is_file():
            _run([uv, "pip", "install", "--python", str(py), "-r", str(req)], log)

    info = torch_info(py)
    if not torch_ok(info, index):
        log("CUDA PyTorch was replaced or is missing; reinstalling it from the CUDA index")
        _run([uv, "pip", "install", "--python", str(py), "--reinstall-package", "torch",
              "--reinstall-package", "torchvision", "--reinstall-package", "torchaudio",
              "torch", "torchvision", "torchaudio", "--index-url", index], log)
        info = torch_info(py)
    if not torch_ok(info, index):
        raise RuntimeError(
            f"PyTorch in {py} cannot use the GPU: {info}. Update the NVIDIA driver (needs CUDA "
            f"{rt.comfyui.min_driver_cuda}+ per `nvidia-smi`) and re-run setup.")
    log(f"  ok: torch {info['torch']} (CUDA {info['cuda']}) sees {info['device']} with {info['vram_mb']} MB")
    return info

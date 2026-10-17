"""`unseen doctor`: check that this machine is ready, with a concrete fix for every failure."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import psutil

from .config import load_models, load_runtime
from .models import is_verified
from .setup_runtime import git_head, torch_info, torch_ok


@dataclass
class Check:
    name: str
    status: str   # PASS | WARN | FAIL | INFO
    detail: str


def _out(cmd: list[str], timeout: int = 60) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:2])


def run_checks(model_set: str = "bench") -> list[Check]:
    rt = load_runtime()
    manifest = load_models()
    checks: list[Check] = []
    add = lambda *a: checks.append(Check(*a))  # noqa: E731

    add("OS", "PASS" if os.name == "nt" else "WARN",
        platform.platform() + ("" if os.name == "nt" else " (pipeline is developed for Windows 11)"))
    add("App Python", "PASS" if sys.version_info >= (3, 12) else "FAIL", sys.version.split()[0])

    smi = _out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits"])
    if smi:
        name, mem, driver = [x.strip() for x in smi.strip().splitlines()[0].split(",")]
        add("GPU", "PASS", f"{name}, {mem} MB VRAM, driver {driver}")
        header = _out(["nvidia-smi"]) or ""
        m = re.search(r"CUDA Version:\s*([\d.]+)", header)
        if m:
            ok = _version_tuple(m.group(1)) >= _version_tuple(rt.comfyui.min_driver_cuda)
            add("Driver CUDA", "PASS" if ok else "FAIL",
                f"{m.group(1)} (need >= {rt.comfyui.min_driver_cuda})"
                + ("" if ok else " -> update the NVIDIA driver (GeForce/Studio driver from nvidia.com)"))
    else:
        add("GPU", "FAIL", "nvidia-smi not found or failed -> install/repair the NVIDIA driver")

    ram = psutil.virtual_memory().total / 1024**3
    add("System RAM", "PASS" if ram >= 15 else "WARN", f"{ram:.1f} GB")
    free = shutil.disk_usage(rt.path("models_dir").parent).free / 1024**3
    add("Free disk", "PASS" if free >= 25 else "WARN" if free >= 12 else "FAIL",
        f"{free:.0f} GB free (models ~15 GB + runtime ~8 GB + outputs)")

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        filters = _out(["ffmpeg", "-hide_banner", "-filters"]) or ""
        encoders = _out(["ffmpeg", "-hide_banner", "-encoders"]) or ""
        has_x264 = "libx264" in encoders
        has_ass = bool(re.search(r"^\s*\S*\s+ass\s", filters, re.MULTILINE))
        add("FFmpeg", "PASS" if has_x264 else "FAIL",
            f"libx264={'yes' if has_x264 else 'NO'}, libass={'yes' if has_ass else 'no (needed from M4)'}"
            + ("" if has_x264 else " -> install a full build, e.g. `winget install Gyan.FFmpeg`"))
    else:
        add("FFmpeg", "FAIL", "ffmpeg/ffprobe not on PATH -> `winget install Gyan.FFmpeg`, reopen terminal")

    add("git", "PASS" if shutil.which("git") else "FAIL",
        "found" if shutil.which("git") else "not found -> `winget install Git.Git`")
    add("uv", "PASS" if shutil.which("uv") else "WARN",
        "found" if shutil.which("uv") else "not found -> run scripts\\windows\\setup.ps1")

    comfy_dir = rt.path("comfyui_dir")
    head = git_head(comfy_dir) if (comfy_dir / ".git").is_dir() else None
    if head is None:
        add("ComfyUI", "FAIL", f"not installed at {comfy_dir} -> uv run unseen setup-comfy")
    else:
        add("ComfyUI", "PASS" if head == rt.comfyui.commit else "WARN",
            f"{head[:10]} (pinned {rt.comfyui.ref} {rt.comfyui.commit[:10]})")
        for node in rt.custom_nodes:
            nh = git_head(comfy_dir / "custom_nodes" / node.name)
            add(node.name, "PASS" if nh == node.commit else "FAIL",
                f"{(nh or 'missing')[:10]} (pinned {node.commit[:10]})")

    py = rt.comfy_python()
    if py.is_file():
        info = torch_info(py)
        ok = torch_ok(info, rt.comfyui.torch_index_url)
        add("ComfyUI PyTorch", "PASS" if ok else "FAIL",
            (f"torch {info.get('torch')} CUDA {info.get('cuda')} on {info.get('device')}" if ok
             else f"{info} -> uv run unseen setup-comfy"))
    else:
        add("ComfyUI PyTorch", "FAIL", f"venv missing at {py} -> uv run unseen setup-comfy")

    if model_set in manifest.sets:
        missing = [k for k in manifest.sets[model_set]
                   if not is_verified(manifest.local_path(rt, k), manifest.files[k].sha256,
                                      manifest.files[k].size)]
        add(f"Models ({model_set})", "PASS" if not missing else "WARN",
            "all present and verified" if not missing
            else f"missing: {', '.join(missing)} -> uv run unseen fetch-models --set {model_set}")

    try:
        from zoneinfo import ZoneInfo
        ZoneInfo("Asia/Kolkata")
        add("Timezone data", "PASS", "Asia/Kolkata available")
    except Exception as exc:
        add("Timezone data", "FAIL", f"{exc!r} -> uv sync (installs tzdata)")

    if os.name == "nt":
        add("Sysmem fallback", "INFO",
            "manual check: NVIDIA Control Panel > Manage 3D settings > Program Settings > add "
            f"{py} > CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback")
    return checks


def run_doctor(model_set: str = "bench") -> int:
    checks = run_checks(model_set)
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status:4}] {c.name:<{width}}  {c.detail}")
    failed = [c for c in checks if c.status == "FAIL"]
    print(f"\n{len(failed)} failure(s)." if failed else "\nAll required checks passed.")
    return 1 if failed else 0

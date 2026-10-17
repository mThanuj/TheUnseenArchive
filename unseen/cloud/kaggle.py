"""Run benchmarks on Kaggle's free GPUs from the local machine (`unseen kaggle-bench`).

Uses the official Kaggle CLI (`kaggle kernels push/status/output/logs`, `kaggle quota`) with your
own account. A private script kernel named "Unseen Archive Runner" is created on first push and
gets a new version on every run. The kernel clones this repository from GitHub at the commit you
have checked out, so the cloud runs exactly the code you run locally.

Credentials come from the environment (.env): KAGGLE_USERNAME and KAGGLE_API_TOKEN (or the files
~/.kaggle/access_token / ~/.kaggle/kaggle.json). Nothing secret is sent to the kernel.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..config import REPO_ROOT, BenchConfig, ModelsManifest, RuntimeConfig

Log = Callable[[str], None]

KERNEL_TITLE = "Unseen Archive Runner"
KERNEL_SLUG = "unseen-archive-runner"
TEMPLATE = Path(__file__).with_name("kaggle_runner_template.py")
DONE_STATES = {"complete", "error", "cancelled"}


class KaggleError(RuntimeError):
    pass


@dataclass
class KaggleJob:
    model: str
    model_set: str
    repo_url: str
    commit: str
    phases: str = "quality"
    preset: str | None = None
    mode: str | None = None
    timeout_min: float | None = None
    prompts: list[dict] | None = None    # [{id, seed, text}] instead of the benchmark prompts


# ------------------------------------------------------------------ helpers

def kaggle_exe() -> str:
    exe = shutil.which("kaggle")
    if not exe:
        raise KaggleError("Kaggle CLI not found. Run `uv sync` (it is a project dependency).")
    return exe


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    return "" if value.startswith("your-") else value   # untouched .env.example placeholder


def username() -> str:
    user = _env("KAGGLE_USERNAME")
    if not user:
        raise KaggleError("Set KAGGLE_USERNAME in .env (your kaggle.com username).")
    return user


def check_credentials() -> None:
    username()
    home = Path.home() / ".kaggle"
    if _env("KAGGLE_API_TOKEN"):
        return
    os.environ.pop("KAGGLE_API_TOKEN", None)   # don't hand a placeholder to the CLI
    if (home / "access_token").is_file() or (home / "kaggle.json").is_file():
        return
    raise KaggleError("No Kaggle credentials. kaggle.com > Settings > API > Create New Token, then put it "
                      "in .env as KAGGLE_API_TOKEN=... (see .env.example).")


def kernel_ref() -> str:
    return f"{username()}/{KERNEL_SLUG}"


def _kaggle(args: list[str], timeout: int = 600) -> str:
    proc = subprocess.run([kaggle_exe(), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise KaggleError(f"`kaggle {' '.join(args)}` failed (exit {proc.returncode}): {out.strip()[-1500:]}")
    return out


def https_repo_url(remote: str) -> str:
    """git@github.com:user/repo.git -> https://github.com/user/repo.git (the kernel clones anonymously)."""
    m = re.match(r"^git@github\.com:(.+)$", remote.strip())
    if m:
        return f"https://github.com/{m.group(1)}"
    return remote.strip()


def local_commit(log: Log = print) -> tuple[str, str]:
    """(https repo url, commit) the kernel should run; refuses commits GitHub doesn't have."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    remote = https_repo_url(git("remote", "get-url", "origin"))
    if git("status", "--porcelain", "--untracked-files=no"):
        log("Note: you have uncommitted local changes; the Kaggle job runs the committed code from GitHub.")
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=REPO_ROOT, capture_output=True)
    if not git("branch", "-r", "--contains", commit):
        raise KaggleError(f"commit {commit[:10]} is not on GitHub yet; the Kaggle job clones from GitHub. "
                          f"Run `git pull` (or push your commit) first.")
    return remote, commit


def parse_status(text: str) -> tuple[str, str]:
    """`user/slug has status "KernelWorkerStatus.COMPLETE"` -> ("complete", failure message)."""
    m = re.search(r'has status "([^"]+)"', text)
    raw = (m.group(1) if m else text).lower()
    state = raw
    for key in ("complete", "error", "running", "queued", "cancel"):
        if key in raw:
            state = "cancelled" if key == "cancel" else key
            break
    fm = re.search(r'Failure message: "(.*)"', text, re.DOTALL)
    return state, (fm.group(1).strip() if fm else "")


def render_runner(job: KaggleJob) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(asdict(job), ensure_ascii=True)
    if '"""' in payload:
        raise KaggleError("job parameters may not contain triple quotes")
    return template.replace("__JOB_JSON__", payload, 1)


def kernel_metadata(accelerator: str) -> dict:
    return {
        "id": kernel_ref(),
        "title": KERNEL_TITLE,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def write_bundle(job: KaggleJob, accelerator: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "runner.py").write_text(render_runner(job), encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(accelerator), indent=2),
                                                  encoding="utf-8")
    return folder


# ------------------------------------------------------------------ actions

NO_KERNEL_MARKERS = ("404", "not found", "kernels.get", "permission")


def status() -> tuple[str, str]:
    try:
        return parse_status(_kaggle(["kernels", "status", kernel_ref()], timeout=120))
    except KaggleError as exc:
        # Kaggle answers a missing *private* kernel with "Permission 'kernels.get' was denied"
        # (the CLI maps both 401 and 403 to that text), not with 404.
        if any(m in str(exc).lower() for m in NO_KERNEL_MARKERS):
            return "none", "no runner notebook yet (the first kaggle-bench creates it)"
        raise


def quota() -> str:
    return _kaggle(["quota"], timeout=120).strip()


def push(job: KaggleJob, accelerator: str, timeout_hours: float, log: Log = print) -> str:
    state, _ = status()
    if state in ("running", "queued"):
        raise KaggleError(f"{kernel_ref()} is already {state}; wait for it or cancel it on kaggle.com")
    with tempfile.TemporaryDirectory(prefix="unseen-kaggle-") as tmp:
        folder = write_bundle(job, accelerator, Path(tmp))
        out = _kaggle(["kernels", "push", "-p", str(folder), "--accelerator", accelerator,
                       "-t", str(int(timeout_hours * 3600))], timeout=600)
    log(out.strip())
    if "error" in out.lower() and "successfully" not in out.lower():
        raise KaggleError(f"push was not accepted: {out.strip()}")
    return out


def wait(timeout_hours: float, poll_s: int = 60, log: Log = print) -> tuple[str, str]:
    t0 = time.monotonic()
    last = None
    time.sleep(15)  # give Kaggle a moment to register the new version
    while True:
        try:
            state, message = status()
        except (KaggleError, subprocess.TimeoutExpired) as exc:  # transient API hiccups
            state, message = "unknown", str(exc)[:300]
        elapsed = (time.monotonic() - t0) / 60
        if state != last:
            log(f"  [{elapsed:5.1f} min] {state}{' - ' + message if message else ''}")
            last = state
        if state in DONE_STATES:
            return state, message
        if elapsed > timeout_hours * 60 + 15:
            raise KaggleError(f"gave up waiting after {elapsed:.0f} min (state {state}); "
                              f"try `uv run unseen kaggle-fetch` later")
        time.sleep(poll_s)


def fetch(rt: RuntimeConfig, log: Log = print) -> dict:
    """Download the latest run's outputs and merge benchmark runs into bench_results/."""
    bench_dir = rt.path("bench_dir")
    dest = bench_dir / f"kaggle_{datetime.now():%Y%m%d-%H%M%S}"
    dest.mkdir(parents=True, exist_ok=True)
    _kaggle(["kernels", "output", kernel_ref(), "-p", str(dest), "-o"], timeout=3600)
    summary_file = dest / "job_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
    merged = []
    for res in sorted((dest / "bench_results").glob("*/results.json")):
        run_dir = res.parent
        target = bench_dir / run_dir.name
        if target.exists():
            target = bench_dir / f"{run_dir.name}_kaggle"
        shutil.move(str(run_dir), str(target))
        merged.append(target)
    summary["_merged_runs"] = [str(p) for p in merged]
    summary["_download_dir"] = str(dest)
    log(f"Downloaded to {dest}; merged runs: {[p.name for p in merged] or 'none'}")
    return summary


def describe_summary(summary: dict) -> str:
    if not summary:
        return "No job_summary.json in the output (the job may have been killed before it started)."
    lines = [f"Cloud job status: {summary.get('status')}"]
    env = summary.get("env") or {}
    if env:
        lines.append(f"  machine: {env.get('nvidia_smi')} | {env.get('driver_cuda')} | python {env.get('python')}")
    for s in summary.get("steps") or []:
        lines.append(f"  [{'ok' if s.get('ok') else 'FAILED'}] {s.get('name')} ({s.get('seconds')} s)")
        if not s.get("ok") and s.get("error"):
            lines.append("    " + s["error"][-1200:].replace("\n", "\n    "))
    return "\n".join(lines)


def model_set_for(model_key: str, cfg: BenchConfig, manifest: ModelsManifest) -> str:
    files = set(cfg.models[model_key].files.values())
    candidates = [name for name, keys in manifest.sets.items() if files <= set(keys)]
    if not candidates:
        raise KaggleError(f"no model set in config/models.yaml contains all files for {model_key}")
    return min(candidates, key=lambda n: len(manifest.sets[n]))  # the tightest set

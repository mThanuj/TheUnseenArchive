"""Command-line entry point: `uv run unseen <command>`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import load_bench, load_dotenv, load_models, load_runtime


def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor
    return run_doctor(args.set)


def cmd_setup_comfy(args: argparse.Namespace) -> int:
    from .setup_runtime import setup_comfy
    setup_comfy(load_runtime())
    return 0


def cmd_fetch_models(args: argparse.Namespace) -> int:
    from .models import fetch_set, is_verified
    rt, manifest = load_runtime(), load_models()
    if args.list or not args.set:
        for name, keys in manifest.sets.items():
            size = sum(manifest.files[k].size for k in keys) / 1e9
            print(f"{name:10} {size:5.1f} GB  {', '.join(keys)}")
        print("\nInstalled:")
        for key, f in manifest.files.items():
            state = "verified" if is_verified(manifest.local_path(rt, key), f.sha256, f.size) else "-"
            print(f"  {state:8} {key}")
        return 0
    fetch_set(manifest, rt, args.set)
    return 0


def parse_preset(text: str | None):
    """'704x1216x97' -> Preset(width=704, height=1216, frames=97)."""
    if not text:
        return None
    from .config import Preset
    try:
        w, h, f = (int(x) for x in text.lower().rstrip("f").split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--preset must look like 704x1216x97, got {text!r}") from None
    return Preset(width=w, height=h, frames=f)


def cmd_bench_video(args: argparse.Namespace) -> int:
    from .bench.compare import write_compare
    from .bench.video_bench import VideoBench
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    bench = VideoBench(args.model)
    if args.timeout_min:
        bench.cfg.case_timeout_minutes = args.timeout_min
    out = bench.run(phases=phases, quick=args.quick, modes=args.mode, preset=args.preset)
    verdict = bench.results.get("verdict", {})
    compare = write_compare(bench.rt.path("bench_dir"))
    print(f"\nDone. {verdict.get('cases_passed', 0)}/{verdict.get('cases_run', 0)} cases passed the "
          f"automatic gate.\nReport: {out / 'report.md'}\nSide-by-side clips: {compare}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .bench.compare import write_compare
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    out = write_compare(load_runtime().path("bench_dir"), models=models, latest_only=not args.all_runs)
    print(f"Wrote {out} - open it in a browser")
    return 0


def cmd_shootout(args: argparse.Namespace) -> int:
    """Render every test prompt with each installed model at its quick preset, then compare."""
    from .bench.compare import write_compare
    from .bench.video_bench import VideoBench
    from .models import is_verified
    rt, manifest, cfg = load_runtime(), load_models(), load_bench()
    wanted = [m.strip() for m in args.models.split(",")] if args.models else list(cfg.models)
    ready, skipped = [], []
    for key in wanted:
        if key not in cfg.models:
            raise KeyError(f"unknown model {key!r}; choose from {sorted(cfg.models)}")
        files = cfg.models[key].files.values()
        ok = all(is_verified(manifest.local_path(rt, f), manifest.files[f].sha256, manifest.files[f].size)
                 for f in files)
        (ready if ok else skipped).append(key)
    if skipped:
        print(f"Skipping (models not downloaded): {', '.join(skipped)}")
    if not ready:
        print("Nothing to run. Download a set first, e.g. uv run unseen fetch-models --set ltxv")
        return 1
    summary = []
    for key in ready:
        print(f"\n===== {key} =====")
        bench = VideoBench(key)
        try:
            bench.run(phases=["quality"], preset=args.preset)
            v = bench.results.get("verdict", {})
            summary.append(f"{key}: {v.get('cases_passed', 0)}/{v.get('cases_run', 0)} passed -> {bench.out_dir}")
        except Exception as exc:  # keep going with the other models
            summary.append(f"{key}: FAILED ({exc})")
    out = write_compare(rt.path("bench_dir"), models=ready)
    print("\nShootout summary:\n  " + "\n  ".join(summary))
    print(f"\nOpen {out} in a browser to compare the clips side by side.")
    return 0


def cmd_export_workflow(args: argparse.Namespace) -> int:
    from .comfy import workflows
    from .bench.video_bench import VideoBench
    bench = VideoBench(args.model)
    case = bench._case("export", "export", bench.cfg.memory_modes[0], bench.model.quick)
    wf = workflows.build(bench.model.workflow, bench._job(case), bench.files)
    out = Path(args.out or f"{args.model}_workflow_api.json")
    out.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} (API format; in ComfyUI use Workflow > Open to inspect it)")
    return 0


def cmd_comfy_serve(args: argparse.Namespace) -> int:
    from .comfy.server import ComfyServer
    rt = load_runtime()
    log = rt.path("comfy_work") / f"serve_{datetime.now():%Y%m%d-%H%M%S}.log"
    with ComfyServer(rt, args.mode, log) as server:
        print(f"ComfyUI running at {server.base_url} (memory mode {args.mode}); log: {log}")
        print("Press Ctrl+C to stop.")
        try:
            while server.alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def cmd_kaggle_bench(args: argparse.Namespace) -> int:
    from .bench.compare import write_compare
    from .bench.video_bench import check_constraints
    from .cloud import kaggle as kg
    rt, manifest, cfg = load_runtime(), load_models(), load_bench()
    if args.model not in cfg.models:
        raise KeyError(f"unknown model {args.model!r}; choose from {sorted(cfg.models)}")
    if args.preset and (problems := check_constraints(cfg.models[args.model], args.preset)):
        raise ValueError("invalid --preset: " + "; ".join(problems))
    kg.check_credentials()
    repo, commit = kg.local_commit()
    job = kg.KaggleJob(
        model=args.model, model_set=kg.model_set_for(args.model, cfg, manifest),
        repo_url=repo, commit=commit, phases=args.phases,
        preset=args.preset.label.rstrip("f") if args.preset else None,
        mode=args.mode, timeout_min=args.timeout_min)
    print(f"Pushing Kaggle job ({args.accelerator}): model={job.model}, set={job.model_set}, "
          f"phases={job.phases}, preset={job.preset or 'model default'}, commit={commit[:10]}")
    kg.push(job, args.accelerator, args.timeout_hours)
    print(f"Running as https://www.kaggle.com/code/{kg.kernel_ref()} (private). "
          f"Setup + model download take ~10-15 min before the first clip.")
    if args.no_wait:
        print("Not waiting. Later: `uv run unseen kaggle-status` and `uv run unseen kaggle-fetch`.")
        return 0
    state, message = kg.wait(args.timeout_hours)
    summary = kg.fetch(rt)
    print(kg.describe_summary(summary))
    print(f"Side-by-side clips: {write_compare(rt.path('bench_dir'))}")
    return 0 if summary.get("status") == "ok" else 1


def cmd_kaggle_status(args: argparse.Namespace) -> int:
    from .cloud import kaggle as kg
    kg.check_credentials()
    try:
        print("Weekly accelerator quota (this also proves your token works):\n" + kg.quota())
    except kg.KaggleError as exc:
        print(f"Your Kaggle credentials were rejected or the API is unreachable:\n  {exc}\n"
              f"Check KAGGLE_API_TOKEN in .env (kaggle.com > Settings > API > Create New Token).")
        return 1
    state, message = kg.status()
    print(f"\nRunner {kg.kernel_ref()}: {state}{' - ' + message if message else ''}")
    if state == "none":
        print(f"If you have already run kaggle-bench, check that KAGGLE_USERNAME ({kg.username()}) "
              f"matches the account the token belongs to.")
    return 0


def cmd_kaggle_fetch(args: argparse.Namespace) -> int:
    from .bench.compare import write_compare
    from .cloud import kaggle as kg
    kg.check_credentials()
    rt = load_runtime()
    summary = kg.fetch(rt)
    print(kg.describe_summary(summary))
    print(f"Side-by-side clips: {write_compare(rt.path('bench_dir'))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unseen", description="The Unseen Archive pipeline tools")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check this machine is ready")
    p.add_argument("--set", default="bench", help="model set to check (default: bench)")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("setup-comfy", help="install pinned ComfyUI + CUDA PyTorch into runtime/")
    p.set_defaults(func=cmd_setup_comfy)

    p = sub.add_parser("fetch-models", help="download and verify model weights")
    p.add_argument("--set", help="model set from config/models.yaml (e.g. bench, ltxv, wan21)")
    p.add_argument("--list", action="store_true", help="list sets and what is installed")
    p.set_defaults(func=cmd_fetch_models)

    models = sorted(load_bench().models)
    p = sub.add_parser("bench-video", help="Milestone 1: generate real AI clips and measure them")
    p.add_argument("--model", required=True, choices=models)
    p.add_argument("--quick", action="store_true", help="one clip only (smoke test)")
    p.add_argument("--phases", default="probe,ladder,quality",
                   help="comma-separated subset of probe,ladder,quality")
    p.add_argument("--mode", action="append",
                   help="memory mode(s) to use instead of config (repeatable)")
    p.add_argument("--timeout-min", type=float, help="per-clip timeout override in minutes")
    p.add_argument("--preset", type=parse_preset,
                   help="WIDTHxHEIGHTxFRAMES for --quick or the quality phase, e.g. 704x1216x97")
    p.set_defaults(func=cmd_bench_video)

    p = sub.add_parser("compare", help="write bench_results/compare.html with all clips side by side")
    p.add_argument("--models", help="comma-separated model keys (default: all)")
    p.add_argument("--all-runs", action="store_true", help="include every run, not just the latest per model")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("shootout", help="all test prompts x every downloaded model, then compare.html")
    p.add_argument("--models", help="comma-separated model keys (default: every model with files present)")
    p.add_argument("--preset", type=parse_preset, help="same WxHxF for all models (default: each model's quick preset)")
    p.set_defaults(func=cmd_shootout)

    p = sub.add_parser("kaggle-bench", help="run bench-video on a free Kaggle GPU and download the clips")
    p.add_argument("--model", required=True, choices=models)
    p.add_argument("--phases", default="quality", help="probe,ladder,quality subset (default: quality)")
    p.add_argument("--preset", type=parse_preset, help="WIDTHxHEIGHTxFRAMES for the quality phase")
    p.add_argument("--mode", help="ComfyUI memory mode on Kaggle (default: config's first mode)")
    p.add_argument("--accelerator", default="NvidiaTeslaT4",
                   help="NvidiaTeslaT4 (free T4 x2, default), or try NvidiaL4X1 / NvidiaL4 if your account has them")
    p.add_argument("--timeout-hours", type=float, default=3.0, help="Kaggle session time limit")
    p.add_argument("--timeout-min", type=float, help="per-clip timeout inside the job")
    p.add_argument("--no-wait", action="store_true", help="push and return immediately")
    p.set_defaults(func=cmd_kaggle_bench)

    p = sub.add_parser("kaggle-status", help="state of the Kaggle runner + your weekly GPU quota")
    p.set_defaults(func=cmd_kaggle_status)

    p = sub.add_parser("kaggle-fetch", help="download the latest Kaggle run's results into bench_results")
    p.set_defaults(func=cmd_kaggle_fetch)

    p = sub.add_parser("export-workflow", help="write a model's API-format workflow JSON")
    p.add_argument("--model", required=True, choices=models)
    p.add_argument("--out")
    p.set_defaults(func=cmd_export_workflow)

    p = sub.add_parser("comfy-serve", help="run the private ComfyUI in the foreground for debugging")
    p.add_argument("--mode", default="dynamic")
    p.set_defaults(func=cmd_comfy_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # never crash on a cp1252 console
        except (AttributeError, ValueError):
            pass
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("(for a traceback re-run as: uv run unseen --debug <command> ...)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

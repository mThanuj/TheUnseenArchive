"""Command-line entry point: `uv run unseen <command>`."""

from __future__ import annotations

import argparse
import json
import os
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


def load_prompts_file(path: str):
    """JSON or YAML list of {id, seed, text} -> [BenchPrompt]."""
    from .config import BenchPrompt, load_yaml
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8")) if p.suffix.lower() == ".json" else load_yaml(p)
    return [BenchPrompt.model_validate(x) for x in data]


def cmd_bench_video(args: argparse.Namespace) -> int:
    from .bench.compare import write_compare
    from .bench.video_bench import VideoBench
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    bench = VideoBench(args.model)
    if args.timeout_min:
        bench.cfg.case_timeout_minutes = args.timeout_min
    if args.prompts_file:
        bench.cfg.prompts = load_prompts_file(args.prompts_file)
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


def run_kaggle_job(model: str, *, phases: str, preset, mode: str | None, timeout_min: float | None,
                   accelerator: str, timeout_hours: float, no_wait: bool,
                   prompts: list[dict] | None = None) -> int:
    from .bench.compare import write_compare
    from .bench.video_bench import check_constraints
    from .cloud import kaggle as kg
    rt, manifest, cfg = load_runtime(), load_models(), load_bench()
    if model not in cfg.models:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(cfg.models)}")
    if preset and (problems := check_constraints(cfg.models[model], preset)):
        raise ValueError("invalid --preset: " + "; ".join(problems))
    kg.check_credentials()
    repo, commit = kg.local_commit()
    job = kg.KaggleJob(
        model=model, model_set=kg.model_set_for(model, cfg, manifest), repo_url=repo, commit=commit,
        phases=phases, preset=preset.label.rstrip("f") if preset else None, mode=mode,
        timeout_min=timeout_min, prompts=prompts)
    print(f"Pushing Kaggle job ({accelerator}): model={job.model}, set={job.model_set}, phases={job.phases}, "
          f"preset={job.preset or 'model default'}, prompts={len(prompts) if prompts else 'benchmark'}, "
          f"commit={commit[:10]}")
    kg.push(job, accelerator, timeout_hours)
    print(f"Running as https://www.kaggle.com/code/{kg.kernel_ref()} (private). "
          f"Setup + model download take ~5-15 min before the first clip.")
    if no_wait:
        print("Not waiting. Later: `uv run unseen kaggle-status` and `uv run unseen kaggle-fetch`.")
        return 0
    kg.wait(timeout_hours)
    summary = kg.fetch(rt)
    print(kg.describe_summary(summary))
    print(f"Side-by-side clips: {write_compare(rt.path('bench_dir'))}")
    return 0 if summary.get("status") == "ok" else 1


def cmd_kaggle_bench(args: argparse.Namespace) -> int:
    prompts = None
    if args.prompts_file:
        prompts = [p.model_dump() for p in load_prompts_file(args.prompts_file)]
    return run_kaggle_job(args.model, phases=args.phases, preset=args.preset, mode=args.mode,
                          timeout_min=args.timeout_min, accelerator=args.accelerator,
                          timeout_hours=args.timeout_hours, no_wait=args.no_wait, prompts=prompts)


# ------------------------------------------------------------------ Milestone 2: stories

def cmd_llm_setup(args: argparse.Namespace) -> int:
    from pydantic import BaseModel
    from .llm.base import generate
    from .story.config import load_story_config
    from .story.engine import make_provider
    cfg = load_story_config()
    names = [n.strip() for n in args.provider.split(",") if n.strip()] if args.provider else cfg.llm.chain
    from .llm.base import LLMError

    class SpeedTest(BaseModel):
        title: str
        lines: list[str]

    results = []
    for name in names:
        print(f"\n== {name}")
        try:
            provider = make_provider(cfg.llm, name)
        except LLMError as exc:
            print(f"  skipped: {exc}")
            results.append(f"{name:10} skipped ({str(exc)[:80]})")
            continue
        slots = getattr(provider, "providers", [provider])
        tiers = {0: "best", 1: "normal"}
        print("  models, in the order they are tried: "
              + ", ".join(f"{p.model} ({tiers.get(getattr(p, 'tier', 0), '?')})" for p in slots))
        try:
            if provider.name == "ollama":
                print(f"  Ollama {provider.version()} at {provider.url}")
                if provider.has_model():
                    print(f"  model present: {provider.model}")
                else:
                    print(f"  pulling {provider.model} (several GB, one time) ...")
                    provider.pull()
            obj, stats, _ = generate(provider, SpeedTest, name="speed-test", temperature=0.7, max_tokens=1000,
                                     max_retries=1, system="You write eerie, restrained micro-fiction. Answer in JSON.",
                                     user="Write a 4-line micro-story: the last log entry of a lighthouse keeper.")
            print(f"  {obj.title}\n    " + "\n    ".join(obj.lines))
            results.append(f"{name:10} {provider.model}: {stats.output_tokens} tokens in {stats.seconds} s "
                           f"({stats.tokens_per_s} tok/s incl. latency); {len(slots)} model(s) in rotation")
        except LLMError as exc:
            print(f"  FAILED: {exc}")
            results.append(f"{name:10} FAILED: {str(exc)[:120]}")
        finally:
            unload = getattr(provider, "unload", None)
            if unload:
                unload()
    print("\nSummary (story new uses the best models of every provider first, then the normal ones, then "
          "the local model):\n  " + "\n  ".join(results))
    return 0


def cmd_llm_models(args: argparse.Namespace) -> int:
    from .llm.base import LLMError
    from .story.config import load_story_config
    from .story.engine import build_provider, provider_models
    cfg = load_story_config().llm
    provider = build_provider(cfg, args.provider)
    lister = getattr(provider, "list_models", None)
    if lister is None:
        print(f"{args.provider} has no model listing; for Ollama use `ollama list`.")
        return 1
    models = lister()
    print(f"{len(models)} models available to your {args.provider} key:\n  " + "\n  ".join(models))
    try:
        used = [f"{p.model} ({'best' if p.tier == 0 else 'normal'})" for p in provider_models(cfg, args.provider)]
        print("\nUsed by the story engine, in this order:\n  " + "\n  ".join(used))
    except LLMError as exc:
        print(f"\n{exc}")
    print("(change the list in config/story.yaml)")
    return 0


def cmd_topics(args: argparse.Namespace) -> int:
    """Check every real-mystery topic title against Wikipedia (no LLM involved)."""
    from datetime import datetime
    from .story.config import load_story_config, load_topics
    from .story.research import ResearchError, fetch_article
    cfg, year = load_story_config(), datetime.now().year
    bad = 0
    for t in load_topics():
        eligible = t.sensitivity == "low" or year - t.year >= cfg.real_mystery.min_years_ago_sensitive
        try:
            text, src = fetch_article(t.wikipedia)
            resolved = src.title.removeprefix("Wikipedia: ")
            note = "" if resolved == t.wikipedia else f"  (resolved to '{resolved}': update config/topics.yaml)"
            print(f"[ok]   {t.id:16} {len(text):>7} chars  {'' if eligible else '(not eligible) '}{note}")
        except ResearchError as exc:
            bad += 1
            print(f"[FAIL] {t.id:16} {exc}")
    print(f"\n{bad} topic(s) failed." if bad else "\nAll topics resolve.")
    return 1 if bad else 0


def cmd_story_new(args: argparse.Namespace) -> int:
    from .story.engine import StoryEngine
    engine = StoryEngine(provider_name=args.provider, seed=args.seed)
    record, out = engine.new_story(category=args.category, topic_id=args.topic, target_seconds=args.seconds)
    print(f"\n{record.draft.title}  ({record.id})")
    print(f"  {record.draft.logline}")
    if record.critique:
        print(f"  critique: {record.critique.scores()}{' -> revised' if record.revised else ''}")
    if record.problems:
        print("  unresolved checks:\n    - " + "\n    - ".join(record.problems))
    print(f"\nRead it: {out / 'story.md'}\nRender its scenes: uv run unseen story render {record.id} --kaggle")
    return 0


def cmd_story_list(args: argparse.Namespace) -> int:
    from .state.db import StateDB
    rows = StateDB().recent(args.limit)
    if not rows:
        print("No stories yet. Create one: uv run unseen story new")
    for r in rows:
        print(f"{r['id']:52} {r['category']:22} {r['status']:20} critique={r['critique_avg']}  {r['title']}")
    return 0


def cmd_story_show(args: argparse.Namespace) -> int:
    from .story.render import load_story, story_dir, to_markdown
    record = load_story(args.id)
    print(to_markdown(record))
    print(f"(files: {story_dir(record.id)})")
    return 0


def cmd_story_render(args: argparse.Namespace) -> int:
    """Generate the story's scenes as video clips (preview size by default) and show them in compare.html."""
    from .bench.compare import write_compare
    from .bench.video_bench import VideoBench
    from .config import BenchPrompt, Preset
    from .story.render import load_story, scene_prompts
    record = load_story(args.id)
    prompts = scene_prompts(record)
    if args.scenes:
        keep = {int(x) for x in args.scenes.split(",")}
        prompts = [p for p, s in zip(prompts, record.draft.scenes) if s.id in keep]
    preset = args.preset or Preset(width=512, height=768, frames=49)
    model = args.model or ("ltxv13b_q8" if args.kaggle else "ltxv13b")
    print(f"Rendering {len(prompts)} scene(s) of '{record.draft.title}' with {model} at {preset.label}")
    if args.kaggle:
        return run_kaggle_job(model, phases="quality", preset=preset, mode=args.mode, timeout_min=None,
                              accelerator=args.accelerator, timeout_hours=args.timeout_hours,
                              no_wait=args.no_wait, prompts=prompts)
    bench = VideoBench(model)
    bench.cfg.prompts = [BenchPrompt.model_validate(p) for p in prompts]
    bench.run(phases=["quality"], preset=preset, modes=[args.mode] if args.mode else None)
    print(f"Side-by-side clips: {write_compare(bench.rt.path('bench_dir'))}")
    return 0


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
    p.add_argument("--prompts-file", help="JSON/YAML list of {id, seed, text} replacing the test prompts")
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
    p.add_argument("--prompts-file", help="JSON/YAML list of {id, seed, text} replacing the test prompts")
    p.set_defaults(func=cmd_kaggle_bench)

    p = sub.add_parser("kaggle-status", help="state of the Kaggle runner + your weekly GPU quota")
    p.set_defaults(func=cmd_kaggle_status)

    p = sub.add_parser("kaggle-fetch", help="download the latest Kaggle run's results into bench_results")
    p.set_defaults(func=cmd_kaggle_fetch)

    # ---- Milestone 2: stories
    p = sub.add_parser("llm-setup", help="check each LLM provider in the chain, pull the local model, measure speed")
    p.add_argument("--provider", help="only these providers, comma-separated (e.g. mistral,cloudflare)")
    p.set_defaults(func=cmd_llm_setup)

    p = sub.add_parser("llm-models", help="list the models a hosted provider offers your API key")
    p.add_argument("--provider", required=True, help="gemini, groq, cerebras, openrouter, ...")
    p.set_defaults(func=cmd_llm_models)

    p = sub.add_parser("topics", help="check every real-mystery topic title against Wikipedia")
    p.set_defaults(func=cmd_topics)

    story = sub.add_parser("story", help="Milestone 2: write, list, show and render stories")
    ssub = story.add_subparsers(dest="story_command", required=True)
    p = ssub.add_parser("new", help="write a new story (research, concepts, script, critique)")
    p.add_argument("--category", choices=["fiction_horror", "psychological_mystery", "real_mystery"])
    p.add_argument("--topic", help="real_mystery topic id from config/topics.yaml")
    p.add_argument("--seconds", type=int, help="target Reel length (default: varied)")
    p.add_argument("--provider", help="force one provider (default: the chain in config/story.yaml)")
    p.add_argument("--seed", type=int, help="random seed for category/ingredient choices")
    p.set_defaults(func=cmd_story_new)
    p = ssub.add_parser("list", help="recent stories")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_story_list)
    p = ssub.add_parser("show", help="print a story's script, scenes, claims and sources")
    p.add_argument("id", help="story id (or a unique prefix)")
    p.set_defaults(func=cmd_story_show)
    p = ssub.add_parser("render", help="generate a story's scenes as clips (preview size) -> compare.html")
    p.add_argument("id", help="story id (or a unique prefix)")
    p.add_argument("--kaggle", action="store_true", help="render on Kaggle (default model ltxv13b_q8)")
    p.add_argument("--model", choices=models, help="default: ltxv13b_q8 on Kaggle, ltxv13b locally")
    p.add_argument("--preset", type=parse_preset, help="default 512x768x49 (fast preview)")
    p.add_argument("--scenes", help="comma-separated scene numbers to render (default: all)")
    p.add_argument("--mode", help="ComfyUI memory mode, e.g. dynamic_fp16 on Kaggle")
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--timeout-hours", type=float, default=4.0)
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(func=cmd_story_render)

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
            # never crash on a cp1252 console; flush every line even into a pipe (live logs)
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless on Windows without symlinks
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

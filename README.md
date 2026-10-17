# The Unseen Archive

*"Some stories should never be forgotten."*

An automated pipeline that will produce one original found-footage horror/mystery Instagram Reel
per day, using **genuinely AI-generated moving footage**, at zero ongoing cost wherever practical.

This repository grows milestone by milestone. Each milestone is runnable on its own and is
verified on the target machine before the next one starts.

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Generate real AI video clips on the RTX 3050 (4 GB) and measure memory, time, motion, output | **Implemented, awaiting first run on Windows** |
| M2 | Story engine: concepts, research with sources, scene breakdown (local LLM) | Not started |
| M3 | Narration (Kokoro TTS) and procedural soundscape | Not started |
| M4 | Assembly, found-footage looks, subtitles, 1080x1920 encode, validation | Not started |
| M5 | Daily orchestration: resumable stages, retries, Task Scheduler (Asia/Kolkata) | Not started |
| M6 | Instagram publishing (official API), disabled by default | Not started |
| M7 | Analytics feedback loop | Not started |

Design decisions, research evidence, licenses and open risks: [docs/decisions.md](docs/decisions.md).

## Target machine

Windows 11, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM, Intel i5-12450H.
Needs: NVIDIA driver supporting **CUDA 13.0+** (check the top-right of `nvidia-smi`), Git, FFmpeg,
about **30 GB free disk**, and internet for the first setup.

## Milestone 1: first genuine AI video clip

### 1. One-time setup (PowerShell)

```powershell
git clone git@github.com:mThanuj/TheUnseenArchive.git   # or the https:// URL
cd TheUnseenArchive
copy .env.example .env          # optional; only HF_TOKEN is used in M1
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
```

The script installs `uv` if needed, creates the app environment, installs the pinned ComfyUI with a
CUDA PyTorch build in `runtime\` (a few GB), and runs `uv run unseen doctor`. Missing Git/FFmpeg:
`winget install Git.Git` / `winget install Gyan.FFmpeg`, then reopen PowerShell.

**Then do this once by hand:** NVIDIA Control Panel > Manage 3D settings > Program Settings > Add >
`<repo>\runtime\comfy-venv\Scripts\python.exe` > *CUDA - Sysmem Fallback Policy* = **Prefer No
Sysmem Fallback** > Apply. Otherwise the driver silently uses system RAM as VRAM and generation
becomes extremely slow instead of failing cleanly.

### 2. Check, download, smoke test

```powershell
uv run pytest                                    # CPU-only unit tests, ~1 min
uv run unseen doctor                             # every line should be PASS (models WARN until downloaded)
uv run unseen fetch-models --set ltxv            # ~8 GB, sha256-verified, resumable
uv run unseen bench-video --model ltxv --quick   # ONE clip: the core milestone
```

The quick run prints progress and writes `bench_results\<timestamp>_ltxv\`:
`report.md`, `results.json`, and per case: `quick.mp4`, `contact_sheet.jpg`, `workflow_api.json`.
**Watch the mp4**: it should show real movement (light sweeping, a door moving), not a still image.

### 3. Full benchmark (after the quick run works)

```powershell
uv run unseen bench-video --model ltxv           # probe -> ladder -> quality; roughly 30-90 min
uv run unseen fetch-models --set wan21           # ~7 GB more
uv run unseen bench-video --model wan21          # slower model; can take several hours
```

`probe` tries each memory mode with a small clip and keeps the fastest clean one; `ladder` raises
resolution/length until something fails or exceeds 20 min/clip; `quality` renders all three test
prompts at the largest passing size. Ctrl+C is safe; the report keeps everything finished so far.

### 4. What to send back

- The console output (especially any `ERROR`), and
- `bench_results\<run>\report.md` (paste it, or commit the run folder: mp4/frames/logs are
  git-ignored, the report, `results.json` and contact sheets are small), and
- your honest opinion of the clips: does the motion look real? Is it usable as found footage?

`results.json` contains your GPU/CPU model and timings, nothing secret; ComfyUI's command line
(which contains local paths) is stripped.

### Useful commands

```powershell
uv run unseen fetch-models --list                # sets, sizes, what is installed
uv run unseen export-workflow --model ltxv       # API workflow JSON for inspection in ComfyUI
uv run unseen comfy-serve --mode dynamic         # run the private ComfyUI at http://127.0.0.1:8188
uv run unseen --debug bench-video --model ltxv --quick   # full traceback on errors
uv run unseen bench-video --model ltxv --quick --mode classic_lowvram   # force a memory mode
```

## Repository layout

```
config/            runtime.yaml (pinned ComfyUI), models.yaml (weights + sha256 + licenses), bench.yaml
unseen/            Python package (CLI: `uv run unseen ...`)
  comfy/           private headless ComfyUI: process control, API client, workflow builders
  bench/           Milestone 1 benchmark: runner, resource monitor, motion metrics, report
scripts/windows/   setup.ps1
tests/             CPU-only unit tests
docs/decisions.md  architecture decisions, research evidence, licensing, risks
runtime/ models/ bench_results/   created locally (git-ignored, except benchmark reports)
```

## Honesty rules built into the project

- Published Reels always disclose AI generation (also required by the LTX-Video license, Attachment A (e)).
- Fiction is labelled as fiction; real cases are labelled as documented and dramatized, with
  sources kept in metadata and claims tagged verified / disputed / speculative.
- Publishing stays disabled until you enable it after testing.
- Nothing here claims to work on your hardware until a benchmark run on it says so.

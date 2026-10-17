# The Unseen Archive

*"Some stories should never be forgotten."*

An automated pipeline that will produce one original found-footage horror/mystery Instagram Reel
per day, using **genuinely AI-generated moving footage**, at zero ongoing cost wherever practical.

This repository grows milestone by milestone. Each milestone is runnable on its own and is
verified on the target machine before the next one starts.

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Generate real AI video clips and measure memory, time, motion, output | **Done**: LTX-Video 13B Q8 on Kaggle T4 (704x1216, 4 s, 18 min/clip); LTX 13B Q3 locally as fallback |
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

### 3. Compare models side by side (round 2)

```powershell
uv run unseen shootout                               # 3 test prompts x every downloaded model, then compare.html
uv run unseen fetch-models --set wan22               # Wan 2.2 5B Turbo, ~8 GB (best quality candidate)
uv run unseen fetch-models --set ltxv13b             # LTX 13B distilled Q3, ~9 GB new (experimental on 16 GB RAM)
uv run unseen fetch-models --set wan21               # Wan 2.1 1.3B, ~3 GB new
uv run unseen shootout --models wan22,ltxv13b        # run just the new ones; compare.html shows all latest runs
```

Open `bench_results\compare.html` in a browser: every clip plays side by side, grouped by prompt
(press Space to restart them together). Single runs: `uv run unseen bench-video --model ltxv
--phases quality --preset 704x1216x97`.

### 4. Cloud GPU on Kaggle (optional, free quota)

Runs the same benchmark on Kaggle's free T4 GPUs with less-compressed models (LTX 13B Q8, Wan 2.2
5B fp16) and downloads the clips into `compare.html`.

One-time setup: create a kaggle.com account, verify your phone (Settings > Phone verification;
needed for GPU + internet), create an API token (Settings > API > Create New Token), read
kaggle.com/terms, then add to `.env` (see `.env.example`):

```
KAGGLE_USERNAME=your-username
KAGGLE_API_TOKEN=the-token
```

```powershell
git pull; uv sync
uv run unseen kaggle-status                                # checks credentials, shows weekly GPU quota
uv run unseen kaggle-bench --model ltxv13b_q8              # ~10-15 min setup on Kaggle, then 3 clips
uv run unseen kaggle-bench --model wan22_5b
uv run unseen kaggle-bench --model ltxv13b_q8 --accelerator NvidiaL4X1   # only if your account has L4
```

The job runs as a private notebook "Unseen Archive Runner" on your account and clones this repo at
your current commit. If the local command is interrupted, `uv run unseen kaggle-fetch` downloads
the latest results later. Each run uses GPU quota (setup included), so check `kaggle-status`.

### 5. Full benchmark for the chosen model

```powershell
uv run unseen bench-video --model ltxv               # probe -> ladder -> quality
```

`probe` tries each memory mode with a small clip and keeps the fastest clean one; `ladder` raises
resolution/length until something fails or exceeds 20 min/clip; `quality` renders all three test
prompts at the largest passing size. Ctrl+C is safe; the report keeps everything finished so far.

### 6. What to send back

- The console output (especially any `ERROR`), and
- the `report.md` of each run (paste it; `bench_results\` is git-ignored), and
- your honest opinion of the clips in `compare.html`: which model looks most believable, does the
  motion look real, is it usable as found footage?

**About grain:** the raw model output should be clean and coherent. The found-footage look
(grain, compression, timestamps, tracking errors) is applied deliberately in post-production (M4),
where it can be controlled. Ugly noise in the raw clip is a generation problem worth reporting.

### Useful commands

```powershell
uv run unseen fetch-models --list                # sets, sizes, what is installed
uv run unseen compare                            # rebuild bench_results\compare.html
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

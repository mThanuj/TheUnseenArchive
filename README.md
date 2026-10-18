# The Unseen Archive

*"Some stories should never be forgotten."*

An automated pipeline that will produce one original found-footage horror/mystery Instagram Reel
per day, using **genuinely AI-generated moving footage**, at zero ongoing cost wherever practical.

This repository grows milestone by milestone. Each milestone is runnable on its own and is
verified on the target machine before the next one starts.

| Milestone | Scope | Status |
|---|---|---|
| **M1** | Generate real AI video clips and measure memory, time, motion, output | **Done**: LTX-Video 13B Q8 on Kaggle T4 (704x1216, 4 s, 18 min/clip); LTX 13B Q3 locally as fallback |
| M2 | Story engine: concepts, research with sources, scene breakdown (hosted free LLMs, local fallback) | **Implemented, tuning story quality** |
| M3 | Narration (Kokoro TTS) and procedural soundscape | Not started |
| M4 | Assembly, found-footage looks, subtitles, 1080x1920 encode, validation | Not started |
| M5 | Daily orchestration: resumable stages, retries, Task Scheduler (Asia/Kolkata) | Not started |
| M6 | Instagram publishing (official API), disabled by default | Not started |
| M7 | Analytics feedback loop | Not started |

Design decisions, research evidence, licenses and open risks: [docs/decisions.md](docs/decisions.md).

## Each round: one script, one report

```powershell
powershell -ExecutionPolicy Bypass -File .\run_this.ps1
```

`run_this.ps1` pulls the latest code (restarting itself if it changed), runs this round's steps,
and writes `run_logs\<timestamp>\REPORT.md` (commit, machine, configured key *names*, each step's
exit code/duration/output, plus any new story.md / benchmark report). The report is copied to the
clipboard: paste it into the chat. `-Only name1,name2` runs a subset of the steps.

## Target machine

Windows 11, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM, Intel i5-12450H.
Needs: NVIDIA driver supporting **CUDA 13.0+** (check the top-right of `nvidia-smi`), Git, FFmpeg,
about **30 GB free disk**, and internet for the first setup.

## Milestone 2: story engine

Writes one complete Reel script per run: category rotation, varied ingredients with cooldowns,
5 candidate concepts filtered for similarity to every past story, a structured script (narration
lines with pauses, scenes with visual/motion/camera/lighting/continuity, sound cues, camera overlay
text, and a ready-to-render video prompt per scene), an editor critique, and one revision if needed.

Real-world mysteries are researched on Wikipedia: the model extracts claims with verbatim quotes,
quotes not found in the source are rejected, narration may only use the remaining claims, and
theories/legends must be hedged. Sources (URL, revision, license) are stored with the story.

```powershell
winget install Ollama.Ollama                     # local LLM runtime (runs in the background)
git pull; uv sync
uv run pytest
uv run unseen llm-setup                          # pulls Gemma 4 E4B (~5 GB, Apache-2.0) and measures speed
uv run unseen story new                          # random category (~10-25 min on CPU)
uv run unseen topics                             # checks every real-mystery Wikipedia title
uv run unseen story new --category real_mystery  # or --topic max_headroom (see config/topics.yaml)
uv run unseen story list
uv run unseen story show <id>                    # same as runs\stories\<id>\story.md
uv run unseen story render <id> --kaggle         # renders every scene at 512x768x49 (preview) on Kaggle
```

Optional but recommended: free API keys make story writing far faster and better than the laptop.
Add any of the keys listed in `.env.example` (each line links to the signup page). Every provider
lists several models in `config/story.yaml`; the engine uses every provider's best models first
(rotating to the next model of the same provider when one hits its free limit, then to the next
provider), then the normal tier, and only then the small local model. When all hosted models are
limited at once it waits up to 5 minutes for one to free up. `uv run unseen llm-setup` shows each
provider's rotation and speed; `uv run unseen llm-models --provider groq` lists the models your key
can use. Free-tier limits and data-use terms are the providers' own.

Outputs per story in `runs\stories\<id>\`: `story.md` (read this), `story.json` (full record),
`prompts.json` (scene video prompts). State is kept in `state\unseen.db`.

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
scripts/windows/   setup.ps1, ensure_ollama.ps1 (starts the local LLM server if it isn't running)
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

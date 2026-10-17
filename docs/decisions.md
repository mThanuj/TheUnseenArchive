# Architecture decisions and research record

Research was done on 2026-10-17. **V** = verified against a primary source (link), **U** =
unverified (vendor claim, recollection, or source unreachable). Nothing is claimed to work on the
target machine until a benchmark run on it shows it.

## Constraints that drive everything

RTX 3050 Laptop with **4 GB VRAM**, 16 GB RAM, Windows 11, ₹0 budget, one Reel per day, and the
hard requirement of genuine temporal motion (no still images with simulated camera moves).

## D1. Video generation: open-weight models, run locally, through headless ComfyUI

**Candidates and evidence**

| Model | 4 GB evidence | License | Decision |
|---|---|---|---|
| LTX-Video 2B 0.9.6 distilled (GGUF) | Only NVIDIA 4 GB report found: GTX 1650 4 GB, 768x512x49f, 8 steps, ~1 min, on a 32 GB RAM box (V, [Lightricks/LTX-Video#144](https://github.com/Lightricks/LTX-Video/issues/144)) | LTXV Open Weights: commercial OK under $10M revenue; Attachment A restrictions (V, [license](https://huggingface.co/city96/LTX-Video-0.9.6-distilled-gguf/blob/main/LICENSE.md)) | **Benchmark first** |
| Wan 2.1 T2V 1.3B | Ran on a 4 GB AMD card, no timing (V, [issue](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/issues/7)); vendor says 8.19 GB without offload (V) | Apache-2.0 (V) | **Benchmark second** |
| Wan 2.2 TI2V 5B Turbo (GGUF) | Uploader claim "usable with an old 4GB GPU", no numbers (U) | Apache-2.0 (V) | Only if the two above pass but look poor |
| LTX-2.x (19-22B) | Smallest GGUF 8.46 GB; Lightricks lists 32 GB+ VRAM (V) | — | Ruled out |
| HunyuanVideo / FramePack | FramePack minimum 6 GB; 8 GB 3050 took ~2 h per 5 s (V) | Tencent license excludes EU/UK/KR | Ruled out |
| CogVideoX-5B, SVD, Mochi, Pyramid Flow | No 4 GB evidence; SVD measured 10.9 GB peak (V) | mixed | Ruled out |
| AnimateDiff (SD1.5) | No verified 4 GB measurement found | OpenRAIL-M | Not needed unless all above fail |

**Hosted alternatives (all checked against docs, V unless noted):** Gemini Veo is paid-only and
SynthID-watermarked; Hugging Face Inference Providers gives free accounts $0 credit; HF ZeroGPU is
5 GPU-minutes/day; Kling/Runway/Luma/Pika have app-only free credits and paid APIs (U). None
supports a free, automated daily Reel. **Kaggle** (free T4 x2, weekly quota, headless
`kaggle kernels push/status/output`) is the only credible free remote GPU; its hours and ToS for
automation are U. It is the designated fallback (M1b) if local generation fails the gate.

**Why ComfyUI rather than diffusers.** The binding constraint is memory. ComfyUI with
city96/ComfyUI-GGUF keeps the T5-XXL / UMT5-XXL text encoders quantized in RAM (2.9-4.2 GB),
whereas transformers dequantizes GGUF to fp16 (~9.5 GB for T5-XXL), which does not fit next to
everything else in 16 GB. ComfyUI's partial loading/offloading is where essentially all low-VRAM
evidence comes from, and the same workflow JSON can run on Kaggle unchanged. Costs: a server
process (managed by `unseen.comfy.server`, private port, started/stopped per job) and version
drift (pinned to a commit; workflows validated against `/object_info` before every run).

**Memory modes.** In ComfyUI v0.41 "dynamic VRAM" is the NVIDIA default and `--lowvram` is a no-op
unless dynamic VRAM is disabled (V, `comfy/cli_args.py` at v0.41.3). The benchmark measures both
rather than assuming. On Windows the driver can spill VRAM to system RAM; setup instructs setting
*Sysmem Fallback Policy = Prefer No Sysmem Fallback* and the benchmark detects spills from
performance counters (heuristic).

**Resolution.** Generate low-res portrait (~480-576 x 768-1024), upscale with ffmpeg to 1080x1920
under the found-footage treatment (M4). Camcorder/CCTV footage is low-res by nature; an AI
upscaler can be added later if needed.

**Proving "genuine motion".** `unseen/bench/metrics.py` computes optical flow between frames and
removes the best-fit global zoom/pan/rotation; what remains ("local motion") cannot be produced by
a Ken Burns move over a still image. Unit tests check that a synthetic Ken Burns clip scores low
and a moving object scores high. Thresholds are provisional until calibrated on real clips.

## D2. Later stages (decided now, each benchmarked in its milestone)

| Stage | Choice | Why | Status |
|---|---|---|---|
| LLM (story, scenes, captions) | Ollama + Gemma 4 (E4B / 12B, Apache-2.0, V) on CPU with JSON-schema output; optional Gemini free tier fallback (limits U) | Free, offline, no quota; text stages run before video so RAM is free | M2 |
| Fact research | Wikipedia/Wikidata, Library of Congress, Chronicling America, Smithsonian Open Access (CC0, V); claims tagged verified/disputed/speculative with source URLs | Never invent facts; paraphrase CC BY-SA text | M2 |
| Anti-repetition | fastembed `BAAI/bge-small-en-v1.5` (MIT, V) cosine similarity + tag cooldowns in SQLite | Tiny CPU model | M2 |
| Narration | Kokoro-82M (Apache-2.0 code+weights, native word timestamps, V) | Commercially clean, CPU-friendly | M3 |
| Sound | Procedural synthesis (numpy/scipy/ffmpeg) + user-curated CC0 samples with a license manifest | Generative audio models checked were non-commercial (MMAudio, AudioLDM2, Tango, ThinkSound) or registration-gated (Stable Audio Open) | M3 |
| Subtitles | Kokoro timestamps; faster-whisper (MIT) as fallback aligner and QA; ASS via pysubs2; OFL fonts | | M3/M4 |
| Loudness | 2-pass loudnorm, target -14 LUFS / -1.5 dBTP (U, common practice; Instagram docs unreachable) | | M4 |
| Scheduling | Windows Task Scheduler (WakeToRun, StartWhenAvailable) running an idempotent stage runner with SQLite checkpoints | No always-on daemon; survives sleep | M5 |
| Publishing | Instagram API with Instagram Login, resumable upload (no Facebook Page, no public hosting) (U, Meta docs unreachable during research; re-verify in M6) | Official API only | M6 |

## D3. Honesty and policy

- **AI disclosure on every Reel.** Required by LTXV license Attachment A (e) (V) and by Meta's
  policy for realistic AI media (U, re-verify in M6). Implemented as caption text (and optionally
  an end card) because no API field for the "AI info" label was found (U).
- **Fiction vs. fact.** Recovered-footage fiction must not be presented as real events.
- **No platform circumvention, no credential exposure** (`.env` is git-ignored; `.env.example` has placeholders).

## Free vs. limited vs. paid

| Component | Classification |
|---|---|
| ComfyUI, ComfyUI-GGUF, LTX-Video 2B, Wan 2.1/2.2, Gemma 4, Kokoro, faster-whisper, FFmpeg, uv | Completely free (local) |
| Hugging Face downloads | Free (rate limits; optional free token) |
| Kaggle GPU fallback | Limited free weekly quota (U) |
| Gemini API fallback | Limited free tier (U); never required |
| Instagram Graph API | Free for own account (U) |
| Veo / Kling / Runway / fal / Replicate | Paid; not used |
| Extra hardware | None required if M1 passes; otherwise Kaggle |

## Open risks / unverified assumptions

1. Whether the GTX 1650 LTX result reproduces with 16 GB RAM and Windows WDDM overhead (M1 measures it).
2. Prompt adherence of 2B/1.3B models for dark, handheld, found-footage scenes (M1 quality phase).
3. Exact `/object_info` schemas at the pinned ComfyUI commit (mitigated by pre-submit validation).
4. Windows "GPU Adapter Memory" counters may be missing or localized (spill detection then reports "not measured").
5. Kaggle quota/ToS, Gemini free-tier limits, all Instagram API details, Instagram loudness targets.
6. Laptop sleep/battery behaviour with Task Scheduler wake timers (M5).

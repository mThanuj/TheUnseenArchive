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

**M1 results so far (target machine, 2026-10-17).** First run: LTX 2B 0.9.6 distilled, 512x768x49,
dynamic VRAM mode: 36 s wall, 1.01 s/step, peak VRAM 3368/4096 MB, ComfyUI RAM 7.4 GB, local
motion p98 0.82 (PASS). User verdict: motion OK, but ugly grain. Causes identified and fixed:
- Schedule: we used the generic `LTXVScheduler`; the distilled model expects the fixed sigmas
  `1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219, 0.0` with `euler_ancestral`
  (V, Lightricks ComfyUI-LTXVideo `ltxvideo-i2v-distilled.json` and 13B distilled workflows).
- Prompts asked for "sensor noise / low-resolution"; rewritten in LTX's documented style
  (chronological, literal, <200 words) with no degradation words. Degradation moves to M4 post.
- Resolution: LTX "works best under 720x1280"; ladder now goes up to 704x1216 (we have time budget).
- VAE decode was checked and is not the cause: core ComfyUI uses the official decode timestep 0.05
  and noise 0.025 (V, `causal_video_autoencoder.py` at v0.41.3).

Round-2 candidates (shootout): LTX 2B (official schedule), **Wan 2.2 TI2V 5B Turbo** (4 steps,
cfg 1, native 704x1280; Apache-2.0), **LTX 13B 0.9.8 distilled Q3_K_M** (6.5 GB; no 4 GB report
exists; RAM is the risk), Wan 2.1 1.3B (official 30 steps / cfg 6 / shift 8). No found-footage,
VHS or CCTV LoRAs exist for these small bases (V, HF search); the closest (war-footage LoRA) is
deliberately not used. Wan 1.3B step-distill LoRAs (CausVid) have unclear/non-commercial
licensing and are not used.

Round-2 results (target machine): LTX 13B Q3_K_M 512x768x49 in 1.7-1.9 min/clip (9.2 s/step,
peak VRAM 3.2 GB, ComfyUI RSS 11-12.5 GB, system RAM near 100%); Wan 2.2 5B Turbo 576x1024x49 in
2.4-2.7 min (VAE decode ~90 s). User preferred LTX 13B visually. The "VRAM spill" verdicts were a
false alarm: ComfyUI's dynamic-VRAM offload keeps weights in pinned host memory, which Windows
counts as "shared GPU memory" (~1 GB for every model). The detector now requires dedicated VRAM
to be full as well. Real local limits: system RAM and the heavy Q3 quantization.

## D1b. Cloud GPU: Kaggle as an optional second backend

The local 3050 stays the primary generator (no account, quota or ToS dependency). Kaggle is added
as a switchable backend (`unseen kaggle-bench`) so less-quantized models can be compared and,
if clearly better, used for production with the laptop as fallback.

- Mechanism (V, Kaggle CLI docs): private script kernel, `kaggle kernels push` with
  `machine_shape`/`--accelerator`, `status`, `output`, `logs`, `kaggle quota`. The kernel clones
  this repo at the local commit and runs the same `setup-comfy` / `fetch-models` / `bench-video`
  code, so cloud and local share one code path. PyTorch's CUDA build is chosen from the driver.
- Caveat (V, ComfyUI `supported_models.py`): LTX-Video runs only in bf16/fp32; the free T4
  (Turing) has no bf16, so LTX runs in fp32 there and is expected to be no faster per clip than the
  laptop (estimate). Gains: Q8 instead of Q3, more RAM, 2 GPUs, laptop stays free. Wan runs fp16 on
  T4 natively. The CLI docs also list L4 accelerators (bf16, 24 GB) "only available to some users".
- Unverified: weekly GPU hours, Kaggle's terms on automated use and on running ComfyUI (kaggle.com
  was unreachable from the authoring machine), session disk size, driver CUDA version.

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

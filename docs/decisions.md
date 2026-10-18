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

## M1 decision (2026-10-17): LTX-Video 13B 0.9.8 distilled Q8 on Kaggle, local Q3 as fallback

First Kaggle run (T4, driver 580.178 / CUDA 13.0, 31 GB RAM, 2 vCPU, cu130 PyTorch worked):
setup 5.5 min (ComfyUI 112 s, 21.6 GB models 212 s); 704x1216x97 (4.0 s @ 24 fps) in
**18.0-18.3 min/clip**: 8 steps at ~123 s/step (fp32, as predicted) + ~70 s VAE decode; peak VRAM
13.0-13.8 of 15.4 GB; all 3 cases passed the gate (local motion p98 0.66-31). The user judged it
the best-looking of all candidates.

Chosen production path: **LTX 13B Q8 on Kaggle** (primary), **LTX 13B Q3 on the laptop** (fallback
when Kaggle is unavailable or out of quota; 1.8 min per 2 s clip at 512x768).

Budget: a Reel needs ~8-10 scene clips of 3-4 s, i.e. ~2.5-3 h of single-T4 time per day at
current speed. To be validated against the account's weekly quota (`unseen kaggle-status`).
Planned speed-ups, cheapest first: (1) `--mode dynamic_fp16` (fp16 on the T4's tensor cores; may
break LTX), (2) the L4 accelerator if the account has it (bf16), (3) both T4s in parallel
(two ComfyUI processes; the session has two GPUs and quota is per session), (4) Q6_K (10.9 GB)
so the model fits in T4 VRAM without streaming.

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

## M2 design (2026-10-17): story engine

- **LLM:** Ollama + `hf.co/google/gemma-4-12B-it-qat-q4_0-gguf` (Google's official QAT 4-bit build,
  Apache-2.0, ungated; V via HF API). Ollama supports Gemma 4 (V, release notes) and JSON-schema
  `format` output (V, docs/api.md). Runs on the laptop CPU/GPU before any video work; the model is
  unloaded afterwards (`keep_alive: 0`). Speed on the i5-12450H is measured by `unseen llm-setup` (U).
  Gemini free tier is an optional comparison provider, never required.
- **Structure:** pydantic models are the JSON schemas the model fills; deterministic validators
  (timing vs. target length at 2.3 words/s, every narration line in exactly one scene, hook <= 12
  words, video-prompt length and banned artefact words, cliches) feed problems back for a fix.
  An editor critique (7 dimensions, 1-10) triggers one revision when any score is below 7.
- **Variety:** weighted category rotation (never 3 in a row), length mix 20-50 s, ingredient pools
  (device, location, anomaly, structure, twist) with a 5-story cooldown, 5 concepts per story, and
  the concept least similar to all past stories (bge-small-en-v1.5 embeddings, MIT) is chosen.
- **Real mysteries:** curated historical topics only (cases involving deaths/disappearances must be
  40+ years old); Wikipedia text via the MediaWiki API; claims need verbatim quotes that are
  checked against the source (punctuation-insensitive); theory/legend/disputed claims must be
  hedged; sources keep URL, revision id and CC BY-SA license; captions disclose "dramatized
  reconstruction, AI-generated". Single-source (Wikipedia) for now; multi-source corroboration is a
  later improvement.
- **Brand device:** every story is an archive entry narrated by "the Archivist".

## M2 update (2026-10-18): hosted models first, rotated per model

Observed on the target machine (round 6): free tiers limit per model and per minute/day. Gemini
returned 429 after one or two calls per model, Groq's per-minute token caps (8k on gpt-oss-120b,
6k on qwen3.8-27b) rejected or delayed long script calls, OpenRouter's free models were rate-limited
upstream, and Ollama was not running. With one model per provider, every provider was cooling down
at once: a real-mystery story was saved without its critique and fact-check. Changes:

- **Slots, not providers.** Each provider lists several models (`models:` in config/story.yaml;
  `*` = version number, expanded against the key's model list, newest first). Order: every
  provider's **best** models (provider by provider, each provider's models top to bottom), then the
  **normal** tier the same way, then the local model. A per-minute limit cools one model down for
  the time the API names (Gemini RetryInfo, Groq "try again in", Retry-After) and the next model is
  used; a used-up daily quota (Gemini `...PerDay...` / "limit: 0", Groq TPD/RPD) skips the model for
  the run; OpenRouter's account-wide `free-models-per-day` skips all its free models; a request too
  large for one model's per-minute cap, or a blocked prompt, skips that model for that call only; a
  rejected key or a stopped Ollama skips the provider.
- **Wait before falling back locally.** When every hosted model is cooling down, a call waits up to
  `max_wait_s` (300 s) for the first one to free up. The daily run has hours; quality matters more.
- **Local model:** `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf` (Gemma 4 E4B, 4.5B effective
  parameters, official QAT q4_0, Apache-2.0, ungated, 5.15 GB; V via HF API in round 1) replaces
  the 12B: it is only the last resort, so speed beats polish. `scripts/windows/ensure_ollama.ps1`
  starts Ollama if installed but not running.
- **Integrity of real cases (from the Hessdalen draft):** the concept/script prompts no longer ask
  real cases for an invented "rewatch clue"; concepts are angles on the documented case, never a
  theory as premise; lines with times/numbers must cite a claim; 2/3 of lines must cite claims;
  `archive_entry` is set deterministically to "RECONSTRUCTION - <case> (<year>)"; a story whose
  fact-check could not run is marked `draft_with_problems`; unhedged theory lines get a hedge
  appended as a last resort. Fiction: a rewrite may not add failed checks (round 6's rewrite dropped
  5 scenes to 3); film devices have no on-screen overlay.
- **More free providers (research 2026-10-18; V = read on an official page that day, U = not):**
  added Mistral free mode (`mistral-large-4-0`, `mistral-medium-3-5`, `mistral-small-latest`, ids V;
  limits only in the console; prompts may be used for training, opt-out available, V; commercial use
  on free mode U), Cloudflare Workers AI (10,000 Neurons/day shared by all models, V; OpenAI-compatible
  endpoint with the account id in the URL, V), Z.ai GLM flash models (free per the pricing page, V;
  ids `glm-4.7-flash`/`glm-4.5-flash` U), and Gemma 4 on the Gemini key (`gemma-4-31b-it`,
  `gemma-4-26b-a4b-it`, free of charge, V; JSON mode U, so the provider falls back to prompt-only
  JSON). NVIDIA NIM is configured but kept out of the chain: its free API access is a "trial service
  ... for prototyping" (V). Excluded: SambaNova (free plan now requires buying credits, V), Cohere
  (trial keys are for testing/proofs of concept, V), Hugging Face Inference Providers (no monthly
  credits for free users any more, V), Alibaba Model Studio and SiliconFlow (one-time credits, V).
  Not checked (documentation sites unreachable from the authoring sandbox): OVHcloud, Together,
  Fireworks, Hyperbolic, Nebius, Chutes, Vercel AI Gateway, Ollama Cloud, ModelScope.

## M2 round 8 (2026-10-18): what round 7 showed

- Gemini's free tier is **20 requests per day per model** (QuotaFailure
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier=20`); `gemini-2.5-pro` is not on the free tier
  (404). Daily quotas reset at midnight US Pacific time. Line and scene repairs (~40% of calls) now
  start on the normal tier so the best models' requests go to concepts, scripts, critiques and the
  fact-check.
- A whole case can be a legend (Green Children of Woolpit: every claim tagged legend). Hedging each
  line produced "according to legend" on nearly every line. Now a legend is framed once (the first
  line that tells it); theories and disputed points are still hedged every time. The critic is told
  hedges are requirements. The deterministic last-resort hedge is a one-word ", reportedly."
- The same story mixed phone, camcorder and 16mm inside a 12th-century scene. Real cases now keep
  one device look before cameras existed, at most two device types after.
- A video prompt asked for "a digital readout flashes 07:45": clock times, readouts and nameplates
  now count as readable-text risks.
- Mistral and Cloudflare keys answered 403 without the reason being kept; 403 now disables only that
  model unless the API's message is about the key/token, and the message is shown.

## M2 round 9 (2026-10-18): what round 8 showed

- Mistral free mode: `mistral-large-4-0` is "not available in your subscription tier" (removed);
  medium/small answered "Rate limit exceeded" on every call, even minutes apart (account-side
  limit; check the console's limits page). Cloudflare: the token works; `@cf/moonshotai/kimi-k2.5`
  now redirects to K2.6, which is paid only (removed); `@cf/openai/gpt-oss-120b` works but spends
  output tokens on reasoning (critique/rating budgets raised to 2500/2000, critiques capped at 6
  issues). OpenRouter: only nemotron-3-super answered in rounds 7-8; it is now first.
  `gemini-*-pro` (404 on the free tier) is no longer listed.
- The Dyatlov draft showed the dead hikers and a close-up of a head injury. Real victims are never
  shown: video prompts naming bodies, remains, skulls, wounds or victims are rewritten to show the
  place, objects or traces; injuries are mentioned only in general terms.
- Rewrites now send an outline of the previous draft (no video prompts/sound cues) so the request
  fits Groq's 8k tokens-per-minute cap ("request too large" skipped the best writer in round 8).
- The banned-word fallback drops whole sentences instead of single words ("renders the scene in,
  muted gray").
- run_this.ps1 names each story's category (round 8's "story-fiction" step picked a real case at
  random) and shows when Gemini's daily quotas reset.

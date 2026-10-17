"""Builders for ComfyUI API-format workflows.

Node class names and input keys match the pinned versions in config/runtime.yaml
(ComfyUI v0.41.3 core nodes + city96/ComfyUI-GGUF); they were read from that source, and
ComfyClient.validate() re-checks them against the live server before every submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .client import Workflow


@dataclass
class ClipJob:
    prompt: str
    negative: str
    seed: int
    width: int
    height: int
    frames: int
    fps: float
    steps: int
    cfg: float
    sampler: str
    filename_prefix: str
    scheduler: str = "simple"
    shift: float | None = None
    sigmas: str | None = None           # LTXV only: manual sigma schedule instead of LTXVScheduler
    tiling: dict = field(default_factory=lambda: {
        "tile_size": 256, "overlap": 64, "temporal_size": 32, "temporal_overlap": 8})


@dataclass
class ModelFiles:
    diffusion: str                      # file name inside models/diffusion_models
    text_encoder: str                   # file name inside models/text_encoders
    vae: str                            # file name inside models/vae


def _diffusion_loader(filename: str) -> dict:
    if filename.endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": filename}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": filename, "weight_dtype": "default"}}


def _text_encoder_loader(filename: str, clip_type: str) -> dict:
    if filename.endswith(".gguf"):
        return {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": filename, "type": clip_type}}
    return {"class_type": "CLIPLoader", "inputs": {"clip_name": filename, "type": clip_type}}


def _decode_and_save(job: ClipJob, samples: list, vae: list) -> dict:
    return {
        "decode": {"class_type": "VAEDecodeTiled",
                   "inputs": {"samples": samples, "vae": vae, **job.tiling}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": job.filename_prefix}},
    }


def ltxv_t2v(job: ClipJob, files: ModelFiles) -> Workflow:
    """LTX-Video text-to-video: SamplerCustom with LTXVScheduler (or manual distilled sigmas)."""
    if job.sigmas:
        sigmas = {"class_type": "ManualSigmas", "inputs": {"sigmas": job.sigmas}}
    else:
        sigmas = {"class_type": "LTXVScheduler",
                  "inputs": {"steps": job.steps, "max_shift": 2.05, "base_shift": 0.95,
                             "stretch": True, "terminal": 0.1, "latent": ["latent", 0]}}
    return {
        "model": _diffusion_loader(files.diffusion),
        "clip": _text_encoder_loader(files.text_encoder, "ltxv"),
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": files.vae}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": job.prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": job.negative, "clip": ["clip", 0]}},
        "cond": {"class_type": "LTXVConditioning",
                 "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": float(job.fps)}},
        "latent": {"class_type": "EmptyLTXVLatentVideo",
                   "inputs": {"width": job.width, "height": job.height,
                              "length": job.frames, "batch_size": 1}},
        "sigmas": sigmas,
        "sampler_select": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": job.sampler}},
        "sample": {"class_type": "SamplerCustom",
                   "inputs": {"model": ["model", 0], "add_noise": True, "noise_seed": job.seed,
                              "cfg": job.cfg, "positive": ["cond", 0], "negative": ["cond", 1],
                              "sampler": ["sampler_select", 0], "sigmas": ["sigmas", 0],
                              "latent_image": ["latent", 0]}},
        **_decode_and_save(job, ["sample", 0], ["vae", 0]),
    }


def _wan_common(job: ClipJob, files: ModelFiles, latent: dict) -> Workflow:
    return {
        "model": _diffusion_loader(files.diffusion),
        "model_sampling": {"class_type": "ModelSamplingSD3",
                           "inputs": {"model": ["model", 0], "shift": job.shift if job.shift is not None else 8.0}},
        "clip": _text_encoder_loader(files.text_encoder, "wan"),
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": files.vae}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": job.prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": job.negative, "clip": ["clip", 0]}},
        "latent": latent,
        "sample": {"class_type": "KSampler",
                   "inputs": {"model": ["model_sampling", 0], "seed": job.seed, "steps": job.steps,
                              "cfg": job.cfg, "sampler_name": job.sampler, "scheduler": job.scheduler,
                              "positive": ["pos", 0], "negative": ["neg", 0],
                              "latent_image": ["latent", 0], "denoise": 1.0}},
        **_decode_and_save(job, ["sample", 0], ["vae", 0]),
    }


def wan21_t2v(job: ClipJob, files: ModelFiles) -> Workflow:
    """Wan 2.1 text-to-video (1.3B). Uses the Hunyuan-style empty video latent, as in ComfyUI's template."""
    latent = {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": job.width, "height": job.height, "length": job.frames, "batch_size": 1}}
    return _wan_common(job, files, latent)


def wan22_ti2v(job: ClipJob, files: ModelFiles) -> Workflow:
    """Wan 2.2 TI2V 5B in text-to-video mode (no start image)."""
    latent = {"class_type": "Wan22ImageToVideoLatent",
              "inputs": {"vae": ["vae", 0], "width": job.width, "height": job.height,
                         "length": job.frames, "batch_size": 1}}
    return _wan_common(job, files, latent)


BUILDERS = {
    "ltxv_t2v": ltxv_t2v,
    "wan21_t2v": wan21_t2v,
    "wan22_ti2v": wan22_ti2v,
}


def build(workflow_name: str, job: ClipJob, files: ModelFiles) -> Workflow:
    try:
        builder = BUILDERS[workflow_name]
    except KeyError:
        raise KeyError(f"unknown workflow {workflow_name!r}; choose from {sorted(BUILDERS)}") from None
    return builder(job, files)

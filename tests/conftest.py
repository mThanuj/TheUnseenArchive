"""Shared fixtures. Tests are CPU-only: no GPU, ffmpeg, network or model files needed."""

from __future__ import annotations

import pytest

from unseen.config import load_bench, load_models, load_runtime

INT = ["INT", {"default": 0}]
FLOAT = ["FLOAT", {"default": 0.0}]
BOOL = ["BOOLEAN", {"default": True}]
STRING = ["STRING", {"default": ""}]
SAMPLERS = ["euler", "euler_ancestral", "uni_pc", "dpmpp_2m"]
SCHEDULERS = ["simple", "normal", "karras", "beta"]
CLIP_TYPES = ["stable_diffusion", "sd3", "mochi", "ltxv", "wan", "hunyuan_image", "cogvideox"]


@pytest.fixture(scope="session")
def manifest():
    return load_models()


@pytest.fixture(scope="session")
def bench_cfg():
    return load_bench()


@pytest.fixture(scope="session")
def runtime_cfg():
    return load_runtime()


@pytest.fixture(scope="session")
def object_info(manifest):
    """Subset of ComfyUI v0.41.3 + ComfyUI-GGUF /object_info for the nodes we use.

    Input names/required-ness were transcribed from the pinned source; combo lists contain
    every model file from config/models.yaml as if all were installed.
    """
    by_dest = {"diffusion_models": [], "text_encoders": [], "vae": []}
    for f in manifest.files.values():
        by_dest[f.dest].append(f.filename)
    req = lambda d: {"input": {"required": d}}  # noqa: E731
    return {
        "UnetLoaderGGUF": req({"unet_name": [by_dest["diffusion_models"]]}),
        "UNETLoader": req({"unet_name": [by_dest["diffusion_models"]],
                           "weight_dtype": [["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {}]}),
        "CLIPLoaderGGUF": req({"clip_name": [by_dest["text_encoders"]], "type": [CLIP_TYPES]}),
        "CLIPLoader": {"input": {"required": {"clip_name": [by_dest["text_encoders"]], "type": [CLIP_TYPES]},
                                 "optional": {"device": [["default", "cpu"], {}]}}},
        "VAELoader": req({"vae_name": [by_dest["vae"]]}),
        "CLIPTextEncode": req({"text": STRING, "clip": ["CLIP", {}]}),
        "LTXVConditioning": req({"positive": ["CONDITIONING", {}], "negative": ["CONDITIONING", {}],
                                 "frame_rate": FLOAT}),
        "EmptyLTXVLatentVideo": req({"width": INT, "height": INT, "length": INT, "batch_size": INT}),
        "LTXVScheduler": {"input": {"required": {"steps": INT, "max_shift": FLOAT, "base_shift": FLOAT,
                                                 "stretch": BOOL, "terminal": FLOAT},
                                    "optional": {"latent": ["LATENT", {}]}}},
        "ManualSigmas": req({"sigmas": STRING}),
        "KSamplerSelect": req({"sampler_name": ["COMBO", {"options": SAMPLERS}]}),
        "SamplerCustom": req({"model": ["MODEL", {}], "add_noise": BOOL, "noise_seed": INT, "cfg": FLOAT,
                              "positive": ["CONDITIONING", {}], "negative": ["CONDITIONING", {}],
                              "sampler": ["SAMPLER", {}], "sigmas": ["SIGMAS", {}],
                              "latent_image": ["LATENT", {}]}),
        "VAEDecodeTiled": req({"samples": ["LATENT", {}], "vae": ["VAE", {}], "tile_size": INT,
                               "overlap": INT, "temporal_size": INT, "temporal_overlap": INT}),
        "SaveImage": req({"images": ["IMAGE", {}], "filename_prefix": STRING}),
        "ModelSamplingSD3": req({"model": ["MODEL", {}], "shift": FLOAT}),
        "EmptyHunyuanLatentVideo": req({"width": INT, "height": INT, "length": INT, "batch_size": INT}),
        "Wan22ImageToVideoLatent": {"input": {"required": {"vae": ["VAE", {}], "width": INT, "height": INT,
                                                           "length": INT, "batch_size": INT},
                                              "optional": {"start_image": ["IMAGE", {}]}}},
        "KSampler": req({"model": ["MODEL", {}], "seed": INT, "steps": INT, "cfg": FLOAT,
                         "sampler_name": [SAMPLERS], "scheduler": [SCHEDULERS],
                         "positive": ["CONDITIONING", {}], "negative": ["CONDITIONING", {}],
                         "latent_image": ["LATENT", {}], "denoise": FLOAT}),
    }

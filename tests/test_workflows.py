import copy

import pytest

from unseen.bench.video_bench import check_constraints
from unseen.comfy import workflows
from unseen.comfy.client import is_link, validate_workflow


def _files(manifest, model):
    return workflows.ModelFiles(**{role: manifest.files[key].filename for role, key in model.files.items()})


def _job(model, variant, preset, prompt="a corridor", seed=1):
    return workflows.ClipJob(
        prompt=prompt, negative="static", seed=seed, width=preset.width, height=preset.height,
        frames=preset.frames, fps=model.fps, steps=model.steps, cfg=model.cfg,
        sampler=model.sampler, scheduler=model.scheduler, shift=model.shift,
        sigmas=variant.get("sigmas"), filename_prefix="bench/test/frame")


@pytest.mark.parametrize("model_key", ["ltxv", "wan21", "wan22"])
def test_every_variant_validates_against_schema(model_key, manifest, bench_cfg, object_info):
    model = bench_cfg.models[model_key]
    for variant in model.variants:
        wf = workflows.build(model.workflow, _job(model, variant, model.quick), _files(manifest, model))
        errors, warnings = validate_workflow(wf, object_info)
        assert errors == [], errors
        assert warnings == [], warnings
        for node in wf.values():
            for value in node["inputs"].values():
                if is_link(value):
                    assert value[0] in wf


def test_ltxv_variants_switch_sigma_source(manifest, bench_cfg):
    model = bench_cfg.models["ltxv"]
    kinds = set()
    for variant in model.variants:
        wf = workflows.build(model.workflow, _job(model, variant, model.quick), _files(manifest, model))
        kinds.add(wf["sigmas"]["class_type"])
        assert wf["sample"]["inputs"]["cfg"] == 1.0
        assert wf["cond"]["inputs"]["frame_rate"] == model.fps
    assert kinds == {"LTXVScheduler", "ManualSigmas"}


def test_loader_choice_follows_file_type(manifest, bench_cfg):
    ltxv = bench_cfg.models["ltxv"]
    wan21 = bench_cfg.models["wan21"]
    wf_l = workflows.build(ltxv.workflow, _job(ltxv, ltxv.variants[0], ltxv.quick), _files(manifest, ltxv))
    wf_w = workflows.build(wan21.workflow, _job(wan21, wan21.variants[0], wan21.quick), _files(manifest, wan21))
    assert wf_l["model"]["class_type"] == "UnetLoaderGGUF"     # .gguf
    assert wf_w["model"]["class_type"] == "UNETLoader"         # .safetensors
    assert wf_l["clip"]["inputs"]["type"] == "ltxv"
    assert wf_w["clip"]["inputs"]["type"] == "wan"


def test_validator_reports_real_problems(manifest, bench_cfg, object_info):
    model = bench_cfg.models["wan21"]
    good = workflows.build(model.workflow, _job(model, model.variants[0], model.quick), _files(manifest, model))

    missing_model = copy.deepcopy(good)
    missing_model["model"]["inputs"]["unet_name"] = "not_downloaded.safetensors"
    errors, _ = validate_workflow(missing_model, object_info)
    assert any("not_downloaded" in e for e in errors)

    unknown_node = copy.deepcopy(good)
    unknown_node["model"]["class_type"] = "UnetLoaderGGUF_v99"
    errors, _ = validate_workflow(unknown_node, object_info)
    assert any("unknown node type" in e for e in errors)

    missing_input = copy.deepcopy(good)
    del missing_input["sample"]["inputs"]["steps"]
    errors, _ = validate_workflow(missing_input, object_info)
    assert any("'steps'" in e for e in errors)

    dangling = copy.deepcopy(good)
    dangling["sample"]["inputs"]["model"] = ["nope", 0]
    errors, _ = validate_workflow(dangling, object_info)
    assert any("missing node" in e for e in errors)

    bad_combo = copy.deepcopy(good)
    bad_combo["sample"]["inputs"]["sampler_name"] = "magic"
    errors, _ = validate_workflow(bad_combo, object_info)
    assert any("magic" in e for e in errors)


def test_all_presets_satisfy_model_constraints(bench_cfg):
    for name, model in bench_cfg.models.items():
        for preset in [model.quick, *model.ladder, *model.floor]:
            assert check_constraints(model, preset) == [], f"{name}: {preset.label}"


def test_vae_tiling_respects_node_limits(bench_cfg):
    t = bench_cfg.vae_tiling
    assert t.tile_size >= 64 and t.tile_size % 32 == 0
    assert t.overlap % 32 == 0
    assert t.temporal_size >= 8 and t.temporal_size % 4 == 0
    assert t.temporal_overlap >= 4 and t.temporal_overlap % 4 == 0

import re

from unseen.config import MODEL_DESTS


def test_manifest_files_are_well_formed(manifest):
    for key, f in manifest.files.items():
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256), key
        assert f.size > 0, key
        assert f.dest in MODEL_DESTS, key
        assert f.license in manifest.licenses, key


def test_manifest_sets_reference_known_files(manifest):
    for name, keys in manifest.sets.items():
        assert keys, name
        for k in keys:
            assert k in manifest.files, f"set {name} references unknown file {k}"


def test_bench_models_reference_known_files_and_roles(manifest, bench_cfg):
    for name, model in bench_cfg.models.items():
        assert set(model.files) == {"diffusion", "text_encoder", "vae"}, name
        for role, key in model.files.items():
            assert key in manifest.files, f"{name}.{role} -> {key}"
        expected = {"diffusion": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}
        for role, key in model.files.items():
            assert manifest.files[key].dest == expected[role], f"{name}.{role}"
        # every bench model must be downloadable as one set
        assert any(set(model.files.values()) <= set(keys) for keys in manifest.sets.values()), name


def test_runtime_pins_are_full_commits(runtime_cfg):
    assert re.fullmatch(r"[0-9a-f]{40}", runtime_cfg.comfyui.commit)
    for node in runtime_cfg.custom_nodes:
        assert re.fullmatch(r"[0-9a-f]{40}", node.commit), node.name
    assert set(runtime_cfg.comfyui.memory_modes) >= {"dynamic", "classic_lowvram"}


def test_bench_memory_modes_exist_in_runtime(bench_cfg, runtime_cfg):
    for mode in bench_cfg.memory_modes + bench_cfg.fallback_memory_modes:
        assert mode in runtime_cfg.comfyui.memory_modes, mode


def test_negative_prompt_survives_utf8_loading(bench_cfg):
    # The Wan negative prompt contains Chinese; Windows' default cp1252 would mangle it.
    assert "静态" in bench_cfg.negative_prompt


def test_dotenv_loader(tmp_path, monkeypatch):
    from unseen.config import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_TEST=bar\nEMPTY_TEST=\nQUOTED_TEST=\"x y\"\nKEEP_TEST=new\n", encoding="utf-8")
    monkeypatch.delenv("FOO_TEST", raising=False)
    monkeypatch.delenv("EMPTY_TEST", raising=False)
    monkeypatch.delenv("QUOTED_TEST", raising=False)
    monkeypatch.setenv("KEEP_TEST", "old")
    load_dotenv(env)
    import os
    assert os.environ["FOO_TEST"] == "bar"
    assert "EMPTY_TEST" not in os.environ
    assert os.environ["QUOTED_TEST"] == "x y"
    assert os.environ["KEEP_TEST"] == "old"

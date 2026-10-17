import json

import pytest

from unseen.cloud import kaggle as kg


def test_parse_status_variants():
    assert kg.parse_status('me/unseen-archive-runner has status "KernelWorkerStatus.COMPLETE"') == ("complete", "")
    assert kg.parse_status('me/x has status "running"')[0] == "running"
    assert kg.parse_status('me/x has status "KernelWorkerStatus.QUEUED"')[0] == "queued"
    assert kg.parse_status('me/x has status "cancelAcknowledged"')[0] == "cancelled"
    state, msg = kg.parse_status('me/x has status "KernelWorkerStatus.ERROR"\nFailure message: "OOM in cell"')
    assert state == "error" and msg == "OOM in cell"


def test_missing_private_kernel_reads_as_none(monkeypatch):
    # Real message from the first `kaggle-status` on the target machine (kernel not created yet)
    msg = ("`kaggle kernels status u/unseen-archive-runner` failed (exit 1): Cannot access kernel "
           "'u/unseen-archive-runner' (Permission 'kernels.get' was denied). The most likely cause "
           "is a wrong kernel slug.")

    def fake(args, timeout=600):
        raise kg.KaggleError(msg)
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setattr(kg, "_kaggle", fake)
    assert kg.status()[0] == "none"


def test_https_repo_url():
    assert kg.https_repo_url("git@github.com:mThanuj/TheUnseenArchive.git") == \
        "https://github.com/mThanuj/TheUnseenArchive.git"
    assert kg.https_repo_url("https://github.com/mThanuj/TheUnseenArchive.git\n") == \
        "https://github.com/mThanuj/TheUnseenArchive.git"


def _job(**over):
    base = dict(model="ltxv13b_q8", model_set="cloud_ltxv13b_q8",
                repo_url="https://github.com/mThanuj/TheUnseenArchive.git", commit="a" * 40,
                phases="quality", preset="704x1216x97")
    base.update(over)
    return kg.KaggleJob(**base)


def test_runner_renders_to_valid_python_with_the_job():
    src = kg.render_runner(_job())
    compile(src, "runner.py", "exec")           # syntactically valid on Kaggle's Python
    assert "__JOB_JSON__" not in src
    start = src.index('json.loads(r"""') + len('json.loads(r"""')
    payload = json.loads(src[start:src.index('""")', start)])
    assert payload["model"] == "ltxv13b_q8" and payload["commit"] == "a" * 40


def test_bundle_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    folder = kg.write_bundle(_job(), "NvidiaTeslaT4", tmp_path / "b")
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/unseen-archive-runner"
    # Kaggle requires the slug to equal the slugified title
    assert meta["title"].lower().replace(" ", "-") == "unseen-archive-runner"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["kernel_type"] == "script" and meta["code_file"] == "runner.py"
    assert (folder / "runner.py").is_file()


def test_placeholders_are_treated_as_missing(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "your-kaggle-username")
    with pytest.raises(kg.KaggleError):
        kg.username()


def test_model_set_for_cloud_models(bench_cfg, manifest):
    assert kg.model_set_for("ltxv13b_q8", bench_cfg, manifest) == "cloud_ltxv13b_q8"
    assert kg.model_set_for("wan22_5b", bench_cfg, manifest) == "cloud_wan22_5b"
    assert kg.model_set_for("ltxv", bench_cfg, manifest) == "ltxv"


def test_torch_index_auto_resolution(runtime_cfg):
    from unseen.setup_runtime import resolve_torch_index
    assert resolve_torch_index(runtime_cfg, "13.4").endswith("/cu130")
    assert resolve_torch_index(runtime_cfg, "12.8").endswith("/cu128")
    assert resolve_torch_index(runtime_cfg, "12.9").endswith("/cu128")
    assert resolve_torch_index(runtime_cfg, "12.6").endswith("/cu126")
    with pytest.raises(RuntimeError):
        resolve_torch_index(runtime_cfg, "11.8")

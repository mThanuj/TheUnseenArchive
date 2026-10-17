from unseen.bench.report import write_report
from unseen.bench.video_bench import evaluate_gate, timing_breakdown
from unseen.comfy.client import classify_status
from unseen.config import Gate

GATE = Gate(max_minutes_per_clip=20, min_clip_seconds=2.0, min_local_motion_p98=0.3,
            max_duplicate_frame_ratio=0.5, spill_threshold_mb=768)

WF = {
    "clip": {"class_type": "CLIPLoaderGGUF"},
    "pos": {"class_type": "CLIPTextEncode"},
    "sample": {"class_type": "KSampler"},
    "decode": {"class_type": "VAEDecodeTiled"},
    "save": {"class_type": "SaveImage"},
}


def test_timing_breakdown_from_events():
    events = [
        (101.0, "executing", {"node": "clip"}),
        (103.0, "executing", {"node": "pos"}),
        (110.0, "executing", {"node": "sample"}),
        (130.0, "progress", {"value": 1, "max": 5, "node": "sample"}),
        (140.0, "progress", {"value": 2, "max": 5, "node": "sample"}),
        (170.0, "progress", {"value": 5, "max": 5, "node": "sample"}),
        (171.0, "executing", {"node": "decode"}),
        (200.0, "progress", {"value": 4, "max": 8, "node": "decode"}),  # tile progress, ignored
        (231.0, "executing", {"node": "save"}),
        (233.0, "executing", {"node": None}),
    ]
    t = timing_breakdown(events, WF, t_submit=100.0, t_done=234.0)
    assert t["queue_to_first_node_s"] == 1.0
    assert t["stages_s"]["load"] == 2.0
    assert t["stages_s"]["text_encode"] == 7.0
    assert t["stages_s"]["sampling"] == 61.0
    assert t["stages_s"]["vae_decode"] == 60.0
    assert t["s_per_step"] == 10.0
    assert t["steps"] == 5
    assert t["sampler_warmup_s"] == 10.0   # 20 s to first tick minus one 10 s step


def _ok_result(**over):
    r = {"status": "ok", "wall_s": 300.0,
         "resources": {"shared_gpu": {"available": True, "delta_mb": 100}},
         "output": {"duration_s": 2.5},
         "motion": {"local_motion_p98": 0.8, "duplicate_frame_ratio": 0.0}}
    r.update(over)
    return r


def test_gate_pass_and_failure_reasons():
    assert evaluate_gate(_ok_result(), GATE)["passed"]

    slow = evaluate_gate(_ok_result(wall_s=25 * 60), GATE)
    assert not slow["passed"] and not slow["within_time"]

    spill = evaluate_gate(_ok_result(resources={"shared_gpu": {"available": True, "delta_mb": 2000}}), GATE)
    assert not spill["passed"] and not spill["within_time"]

    static = evaluate_gate(_ok_result(motion={"local_motion_p98": 0.01, "duplicate_frame_ratio": 0.9}), GATE)
    assert not static["passed"] and static["within_time"] and not static["motion_ok"]

    oom = evaluate_gate({"status": "oom"}, GATE)
    assert not oom["passed"] and not oom["within_time"]

    unmeasured = evaluate_gate(_ok_result(resources={"shared_gpu": {"available": False}}), GATE)
    assert unmeasured["passed"] and unmeasured["notes"]


def test_classify_status_detects_oom():
    status = {"status_str": "error", "completed": False, "messages": [
        ["execution_start", {}],
        ["execution_error", {"node_id": "sample", "node_type": "KSampler",
                             "exception_type": "torch.OutOfMemoryError",
                             "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB"}],
    ]}
    kind, msg = classify_status(status)
    assert kind == "oom" and "KSampler" in msg

    other = {"status_str": "error", "messages": [["execution_error", {"exception_message": "boom"}]]}
    assert classify_status(other)[0] == "error"

    interrupted = {"status_str": "error", "messages": [["execution_interrupted", {}]]}
    assert classify_status(interrupted)[0] == "timeout"


def test_compare_page_lists_finished_clips(tmp_path):
    import json
    from unseen.bench.compare import write_compare
    for run, model, ok in (("20261017-1000_ltxv", "ltxv", True), ("20261017-1100_wan21", "wan21", False)):
        d = tmp_path / run
        d.mkdir()
        case = {"case_id": "quality_official_cctv_figure", "prompt_id": "cctv_figure",
                "status": "ok" if ok else "oom", "preset_label": "512x768x49f",
                "output": {"file": "quality_official_cctv_figure.mp4"} if ok else {},
                "gate": {"passed": ok}}
        (d / "results.json").write_text(json.dumps({"model_key": model, "model_label": model,
                                                    "cases": [case]}), encoding="utf-8")
    html = write_compare(tmp_path).read_text(encoding="utf-8")
    assert "20261017-1000_ltxv/quality_official_cctv_figure/quality_official_cctv_figure.mp4" in html
    assert "cctv_figure" in html
    assert html.count("<video") == 1   # the failed wan21 case has no clip


def test_report_renders_partial_results(tmp_path):
    results = {
        "model_label": "Test model", "started_at": "now", "env": {}, "gate": GATE.model_dump(),
        "selection": {"memory_mode": "dynamic", "preset": {"width": 512, "height": 768, "frames": 49}},
        "verdict": {"gate_passed": False, "cases_passed": 0, "cases_run": 2},
        "cases": [
            {"case_id": "probe_dynamic", "status": "oom", "error": "CUDA out of memory",
             "gate": {"passed": False, "reasons": ["status=oom"]}, "preset_label": "512x768x49f"},
            {"case_id": "quick", "status": "ok", "wall_s": 61.0, "preset_label": "512x768x49f",
             "timing": {"stages_s": {"text_encode": 4.0, "vae_decode": 9.0}, "s_per_step": 5.5},
             "resources": {"vram": {"available": True, "peak_mb": 3900, "total_mb": 4096},
                           "shared_gpu": {"available": False}},
             "output": {"duration_s": 2.04, "fps": 24.0},
             "motion": {"local_motion_p98": 0.9, "duplicate_frame_ratio": 0.0},
             "gate": {"passed": True, "reasons": []}, "contact_sheet": "quick/contact_sheet.jpg",
             "prompt_id": "flashlight_corridor"},
        ],
    }
    out = write_report(results, tmp_path / "report.md")
    text = out.read_text(encoding="utf-8")
    assert "512x768x49f" in text and "status=oom" in text and "contact_sheet.jpg" in text

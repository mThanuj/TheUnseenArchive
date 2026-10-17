"""Milestone 1 benchmark: generate real AI video clips on this machine and measure everything.

Phases (configured in config/bench.yaml):
  probe   - quick preset once per memory mode, each in a fresh ComfyUI process; the fastest mode
            that finishes without OOM/spill is selected
  ladder  - escalate resolution/length with that mode until a case fails or exceeds the time gate
  quality - largest passing preset x every variant x every prompt in one warm server, for review
Results are written after every case, so a crash or Ctrl+C still leaves a usable report.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx
import psutil

from .. import media
from ..comfy import workflows
from ..comfy.client import ComfyClient, ComfyError
from ..comfy.server import ComfyServer, ComfyStartupError
from ..config import (BenchConfig, BenchModel, BenchPrompt, Gate, ModelsManifest, Preset,
                      RuntimeConfig, load_bench, load_models, load_runtime)
from ..fsutil import move_with_retry
from ..models import is_verified
from .metrics import contact_sheet, load_frames, motion_metrics
from .monitor import ResourceMonitor
from .report import write_report

Log = Callable[[str], None]

CATEGORY = {
    "UnetLoaderGGUF": "load", "UNETLoader": "load", "CLIPLoaderGGUF": "load",
    "CLIPLoader": "load", "VAELoader": "load",
    "CLIPTextEncode": "text_encode",
    "KSampler": "sampling", "SamplerCustom": "sampling",
    "VAEDecodeTiled": "vae_decode", "VAEDecode": "vae_decode",
    "SaveImage": "save",
}
SAMPLERS = {"KSampler", "SamplerCustom"}


@dataclass
class Case:
    case_id: str
    phase: str
    variant: dict
    memory_mode: str
    preset: Preset
    prompt: BenchPrompt


# ----------------------------------------------------------------- pure helpers

def check_constraints(model: BenchModel, preset: Preset) -> list[str]:
    c = model.constraints
    errors = []
    if preset.width % c.dim_multiple or preset.height % c.dim_multiple:
        errors.append(f"{preset.label}: width/height must be multiples of {c.dim_multiple}")
    if preset.frames < 1 or (preset.frames - 1) % c.frame_step:
        errors.append(f"{preset.label}: frames must be {c.frame_step}n+1")
    return errors


def timing_breakdown(events: list[tuple[float, str, dict]], workflow: dict,
                     t_submit: float, t_done: float) -> dict:
    """Seconds per stage from ComfyUI 'executing'/'progress' WebSocket events."""
    starts = [(t, d.get("node")) for t, typ, d in events if typ == "executing"]
    stages: dict[str, float] = {}
    sampler_start = None
    for i, (t, node) in enumerate(starts):
        if node is None:
            continue
        t_end = starts[i + 1][0] if i + 1 < len(starts) else t_done
        ctype = workflow.get(str(node), {}).get("class_type", "?")
        cat = CATEGORY.get(ctype, "other")
        stages[cat] = round(stages.get(cat, 0.0) + (t_end - t), 2)
        if ctype in SAMPLERS and sampler_start is None:
            sampler_start = t

    progress = [(t, d.get("value"), d.get("max")) for t, typ, d in events
                if typ == "progress" and workflow.get(str(d.get("node")), {}).get("class_type") in SAMPLERS]
    out: dict[str, Any] = {"stages_s": stages, "queue_to_first_node_s": None,
                           "s_per_step": None, "sampler_warmup_s": None, "steps": None}
    if starts:
        out["queue_to_first_node_s"] = round(starts[0][0] - t_submit, 2)
    if len(progress) >= 2:
        (t_a, v_a, mx), (t_b, v_b, _) = progress[0], progress[-1]
        if isinstance(v_a, int) and isinstance(v_b, int) and v_b > v_a:
            sps = (t_b - t_a) / (v_b - v_a)
            out["s_per_step"] = round(sps, 2)
            out["steps"] = mx
            if sampler_start is not None:
                # time from sampler start to first progress tick = model load/transfer + one step
                out["sampler_warmup_s"] = round(max(0.0, t_a - sampler_start - sps), 2)
    return out


def spilled(result: dict, gate: Gate) -> bool | None:
    shared = (result.get("resources") or {}).get("shared_gpu") or {}
    if not shared.get("available"):
        return None
    return shared.get("delta_mb", 0) > gate.spill_threshold_mb


def evaluate_gate(result: dict, gate: Gate) -> dict:
    reasons: list[str] = []
    notes: list[str] = []
    ok = result.get("status") == "ok"
    if not ok:
        reasons.append(f"status={result.get('status')}")
    within_time = ok and result.get("wall_s", 1e9) / 60 <= gate.max_minutes_per_clip
    if ok and not within_time:
        reasons.append(f"took {result['wall_s'] / 60:.1f} min > {gate.max_minutes_per_clip} min")
    sp = spilled(result, gate)
    if sp:
        reasons.append("VRAM spilled into shared system memory")
    elif sp is None:
        notes.append("shared-memory spill not measured")
    out = result.get("output") or {}
    if ok and out.get("duration_s", 0) < gate.min_clip_seconds:
        reasons.append(f"clip {out.get('duration_s')}s < {gate.min_clip_seconds}s")
    motion = result.get("motion") or {}
    motion_ok = (motion.get("local_motion_p98", 0) >= gate.min_local_motion_p98
                 and motion.get("duplicate_frame_ratio", 1) <= gate.max_duplicate_frame_ratio)
    if ok and not motion_ok:
        reasons.append("motion below provisional threshold (check the clip visually)")
    return {"passed": ok and not reasons, "within_time": bool(within_time) and not sp,
            "motion_ok": ok and motion_ok, "reasons": reasons, "notes": notes}


def collect_env(rt: RuntimeConfig) -> dict:
    env: dict[str, Any] = {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
        "comfyui_pinned": f"{rt.comfyui.ref} ({rt.comfyui.commit[:10]})",
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        env["gpu"] = {
            "name": name.decode() if isinstance(name, bytes) else name,
            "vram_total_mb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 1024**2),
            "driver": (lambda v: v.decode() if isinstance(v, bytes) else v)(pynvml.nvmlSystemGetDriverVersion()),
            "driver_cuda": f"{cuda // 1000}.{(cuda % 1000) // 10}",
        }
    except Exception as exc:
        env["gpu"] = {"error": repr(exc)}
    return env


# ------------------------------------------------------------------- benchmark

class VideoBench:
    def __init__(self, model_key: str, log: Log = print,
                 rt: RuntimeConfig | None = None, manifest: ModelsManifest | None = None,
                 cfg: BenchConfig | None = None):
        self.rt = rt or load_runtime()
        self.manifest = manifest or load_models()
        self.cfg = cfg or load_bench()
        if model_key not in self.cfg.models:
            raise KeyError(f"unknown model {model_key!r}; choose from {sorted(self.cfg.models)}")
        self.model_key = model_key
        self.model = self.cfg.models[model_key]
        self.log = log
        self.run_tag = f"{datetime.now():%Y%m%d-%H%M%S}_{model_key}"
        self.out_dir = self.rt.path("bench_dir") / self.run_tag
        self.comfy_log = self.out_dir / "comfyui.log"
        self.files = workflows.ModelFiles(
            diffusion=self.manifest.files[self.model.files["diffusion"]].filename,
            text_encoder=self.manifest.files[self.model.files["text_encoder"]].filename,
            vae=self.manifest.files[self.model.files["vae"]].filename,
        )
        self.results: dict[str, Any] = {
            "schema": 1, "model_key": model_key, "model_label": self.model.label,
            "started_at": datetime.now().isoformat(timespec="seconds"), "finished_at": None,
            "gate": self.cfg.gate.model_dump(), "env": {}, "comfyui": None,
            "cases": [], "selection": {}, "verdict": {},
        }

    # ------------------------------------------------------------ bookkeeping
    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "results.json", "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, indent=2, ensure_ascii=False)
        write_report(self.results, self.out_dir / "report.md")

    def preflight(self) -> None:
        problems = []
        for role, key in self.model.files.items():
            f = self.manifest.files[key]
            if not is_verified(self.manifest.local_path(self.rt, key), f.sha256, f.size):
                problems.append(f"model file for {role} ({f.filename}) is missing or unverified")
        if problems:
            sets = [s for s, keys in self.manifest.sets.items()
                    if set(self.model.files.values()) <= set(keys)]
            hint = f"run: uv run unseen fetch-models --set {sets[0] if sets else self.model_key}"
            raise RuntimeError("; ".join(problems) + f". {hint}")
        for preset in [self.model.quick, *self.model.ladder, *self.model.floor]:
            problems += check_constraints(self.model, preset)
        if problems:
            raise RuntimeError("invalid presets in config/bench.yaml: " + "; ".join(problems))
        media.require("ffmpeg")
        media.require("ffprobe")

    # --------------------------------------------------------------- one case
    def _job(self, case: Case) -> workflows.ClipJob:
        m = self.model
        return workflows.ClipJob(
            prompt=case.prompt.text, negative=m.negative_prompt or self.cfg.negative_prompt,
            seed=case.prompt.seed,
            width=case.preset.width, height=case.preset.height, frames=case.preset.frames,
            fps=m.fps, steps=m.steps, cfg=m.cfg, sampler=case.variant.get("sampler", m.sampler),
            scheduler=m.scheduler,
            shift=m.shift, sigmas=case.variant.get("sigmas"),
            filename_prefix=f"bench/{self.run_tag}/{case.case_id}/frame",
            tiling=self.cfg.vae_tiling.model_dump(),
        )

    def _stub(self, case: Case) -> dict:
        return {"case_id": case.case_id, "phase": case.phase, "memory_mode": case.memory_mode,
                "variant": case.variant.get("name"), "preset": case.preset.model_dump(),
                "preset_label": case.preset.label, "prompt_id": case.prompt.id,
                "status": "pending"}

    def run_case(self, server: ComfyServer, client: ComfyClient, case: Case) -> dict:
        result = self._stub(case)
        case_dir = self.out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        wf = workflows.build(self.model.workflow, self._job(case), self.files)
        with open(case_dir / "workflow_api.json", "w", encoding="utf-8") as fh:
            json.dump(wf, fh, indent=2, ensure_ascii=False)

        errors, warnings = client.validate(wf)
        if warnings:
            result["validation_warnings"] = warnings
        if errors:
            result.update(status="invalid", error="\n".join(errors))
            result["gate"] = evaluate_gate(result, self.cfg.gate)
            return result

        recorder = client.recorder()
        recorder.start()
        recorder.wait_connected(10)
        monitor = ResourceMonitor(server.pid).start()
        entry = None
        t0 = time.monotonic()
        try:
            prompt_id = client.submit(wf)
            entry = client.wait(prompt_id, self.cfg.case_timeout_minutes * 60, alive=server.alive)
            result["status"] = "ok"
        except ComfyError as exc:
            result.update(status=exc.kind, error=str(exc))
        except httpx.HTTPError as exc:  # server stopped responding mid-run
            result.update(status="error", error=f"HTTP error talking to ComfyUI: {exc!r}")
        finally:
            t1 = time.monotonic()
            result["resources"] = monitor.stop()
            recorder.stop()
        result["wall_s"] = round(t1 - t0, 2)
        result["timing"] = timing_breakdown(recorder.events, wf, t0, t1)
        if recorder.error:
            result["timing"]["recorder_error"] = recorder.error

        if entry is not None:
            try:
                self._collect_outputs(server, entry, case_dir, result)
            except Exception as exc:  # keep the measurements even if post-processing fails
                result.update(status="postprocess_error", error=repr(exc))
        if result["status"] != "ok":
            result["log_tail"] = server.log_tail(80)
        result["gate"] = evaluate_gate(result, self.cfg.gate)
        return result

    def _collect_outputs(self, server: ComfyServer, entry: dict, case_dir: Path, result: dict) -> None:
        images = ((entry.get("outputs") or {}).get("save") or {}).get("images") or []
        if not images:
            raise RuntimeError("ComfyUI reported success but returned no frames")
        frames_dir = case_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out_root = server.work_dir("output")
        paths = []
        for i, img in enumerate(images, 1):
            src = out_root / (img.get("subfolder") or "") / img["filename"]
            dst = frames_dir / f"frame_{i:05d}.png"
            move_with_retry(src, dst)
            paths.append(dst)
        mp4 = media.encode_frames(frames_dir, self.model.fps, case_dir / f"{case_dir.name}.mp4")
        result["output"] = media.probe(mp4)
        result["output"]["file"] = mp4.name
        frames = load_frames(paths)
        result["motion"] = motion_metrics(frames)
        contact_sheet(frames, case_dir / "contact_sheet.jpg")
        result["contact_sheet"] = f"{case_dir.name}/contact_sheet.jpg"

    # ---------------------------------------------------------------- sessions
    def _session(self, mode: str, cases: list[Case],
                 stop_when: Callable[[dict], bool] | None = None) -> list[dict]:
        """Run cases sequentially in one ComfyUI process, restarting it after crashes/OOMs."""
        out: list[dict] = []
        server: ComfyServer | None = None
        client: ComfyClient | None = None
        try:
            for case in cases:
                if server is None or not server.alive():
                    if server is not None:
                        server.stop()
                    server = ComfyServer(self.rt, mode, self.comfy_log)
                    self.log(f"  starting ComfyUI (memory mode: {mode}) ...")
                    try:
                        server.start()
                    except ComfyStartupError as exc:
                        r = self._stub(case)
                        r.update(status="startup_failed", error=str(exc))
                        r["gate"] = evaluate_gate(r, self.cfg.gate)
                        self._record(r)
                        out.append(r)
                        server = None
                        break
                    client = ComfyClient(server.base_url)
                    if self.results["comfyui"] is None:
                        try:
                            stats = client.system_stats()
                            stats.get("system", {}).pop("argv", None)  # contains local paths
                            self.results["comfyui"] = stats
                        except Exception as exc:
                            self.results["comfyui"] = {"error": repr(exc)}
                self.log(f"  [{case.case_id}] {case.preset.label}, variant={case.variant.get('name')}, "
                         f"prompt={case.prompt.id} ... (can take many minutes)")
                assert client is not None
                r = self.run_case(server, client, case)
                self._record(r)
                out.append(r)
                if r["status"] in ("oom", "crash", "timeout"):
                    client.close()
                    server.stop()
                    server, client = None, None
                if stop_when and stop_when(r):
                    break
        finally:
            if client is not None:
                client.close()
            if server is not None:
                server.stop()
        return out

    def _record(self, r: dict) -> None:
        self.results["cases"].append(r)
        self.save()
        timing = r.get("timing") or {}
        vram = ((r.get("resources") or {}).get("vram") or {}).get("peak_mb")
        motion = (r.get("motion") or {}).get("local_motion_p98")
        verdict = "PASS" if r["gate"]["passed"] else "FAIL: " + "; ".join(r["gate"]["reasons"])
        self.log(f"    -> {r['status']} in {r.get('wall_s', 0) / 60:.1f} min, "
                 f"s/step={timing.get('s_per_step')}, peak VRAM={vram} MB, "
                 f"local motion p98={motion} | {verdict}")
        if r.get("error"):
            self.log(f"    error: {r['error'][:500]}")

    # ------------------------------------------------------------------ phases
    def _case(self, case_id: str, phase: str, mode: str, preset: Preset,
              prompt: BenchPrompt | None = None, variant: dict | None = None) -> Case:
        return Case(case_id, phase, variant or self.model.variants[0], mode, preset,
                    prompt or self.cfg.prompts[0])

    def phase_probe(self, modes: list[str]) -> dict | None:
        self.log(f"Phase probe: {self.model.quick.label} in memory modes {modes}")
        results = [self._session(m, [self._case(f"probe_{m}", "probe", m, self.model.quick)])[0]
                   for m in modes]
        clean = [r for r in results if r["status"] == "ok" and not spilled(r, self.cfg.gate)]
        if not clean:
            return None
        best = min(clean, key=lambda r: r["wall_s"])
        self.results["selection"]["memory_mode"] = best["memory_mode"]
        self.log(f"  selected memory mode: {best['memory_mode']} ({best['wall_s'] / 60:.1f} min)")
        return best

    def phase_ladder(self, mode: str, probe_best: dict | None, quick_failed: bool) -> Preset | None:
        presets = self.model.floor if quick_failed else self.model.ladder
        self.log(f"Phase ladder ({mode}): {[p.label for p in presets]}")
        measured = [probe_best] if probe_best else []
        cases = [self._case(f"ladder_{i}_{p.label}", "ladder", mode, p)
                 for i, p in enumerate(presets) if not (probe_best and p == self.model.quick)]
        measured += self._session(mode, cases, stop_when=lambda r: not r["gate"]["within_time"])
        usable = [r for r in measured if r and r["gate"]["within_time"]]
        if not usable:
            return None
        best = max(usable, key=lambda r: r["preset"]["width"] * r["preset"]["height"] * r["preset"]["frames"])
        preset = Preset.model_validate(best["preset"])
        self.results["selection"]["preset"] = preset.model_dump()
        self.log(f"  largest preset within the time gate: {preset.label}")
        return preset

    def phase_quality(self, mode: str, preset: Preset) -> None:
        self.log(f"Phase quality ({mode}, {preset.label}): "
                 f"{len(self.model.variants)} variant(s) x {len(self.cfg.prompts)} prompts")
        cases = [self._case(f"quality_{v['name']}_{p.id}", "quality", mode, preset, p, v)
                 for v in self.model.variants for p in self.cfg.prompts]
        self._session(mode, cases)

    def _verdict(self) -> None:
        cases = self.results["cases"]
        passed = [c for c in cases if c.get("gate", {}).get("passed")]
        self.results["verdict"] = {
            "cases_run": len(cases),
            "cases_passed": len(passed),
            "gate_passed": bool(passed),
            "note": ("Automatic checks only. Watch the clips/contact sheets: the final call on "
                     "visual quality and believable motion is a human one."),
        }

    def run(self, phases: list[str], quick: bool = False, modes: list[str] | None = None,
            preset: Preset | None = None) -> Path:
        """`preset` overrides the quick preset (with quick=True) or the quality-phase preset."""
        self.preflight()
        if preset is not None and (problems := check_constraints(self.model, preset)):
            raise RuntimeError("invalid --preset: " + "; ".join(problems))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results["env"] = collect_env(self.rt)
        self.log(f"Benchmark {self.model.label} -> {self.out_dir}")
        modes = modes or self.cfg.memory_modes
        try:
            if quick:
                p = preset or self.model.quick
                self.log(f"Quick run: one clip, {p.label}, memory mode {modes[0]}")
                self._session(modes[0], [self._case("quick", "quick", modes[0], p)])
                return self.out_dir
            probe_best = None
            mode = modes[0]
            quick_failed = False
            if "probe" in phases:
                probe_best = self.phase_probe(modes)
                if probe_best is None and self.cfg.fallback_memory_modes:
                    probe_best = self.phase_probe(self.cfg.fallback_memory_modes)
                if probe_best is not None:
                    mode = probe_best["memory_mode"]
                # a quick clip that works but blows the time gate means: go smaller, not bigger
                quick_failed = probe_best is None or not probe_best["gate"]["within_time"]
            chosen = None if quick_failed else self.model.quick
            if "ladder" in phases:
                chosen = self.phase_ladder(mode, probe_best, quick_failed)
            if preset is not None:
                chosen = preset
            if "quality" in phases:
                if chosen is None:
                    self.log("No preset finished within the time gate; skipping the quality phase.")
                else:
                    self.phase_quality(mode, chosen)
            return self.out_dir
        finally:
            self._verdict()
            self.results["finished_at"] = datetime.now().isoformat(timespec="seconds")
            self.save()
            self.log(f"Report: {self.out_dir / 'report.md'}")

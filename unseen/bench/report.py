"""Render benchmark results as a Markdown report that is easy to read and to paste back."""

from __future__ import annotations

from pathlib import Path


def _fmt(v, suffix: str = "", digits: int = 1) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}{suffix}"
    return f"{v}{suffix}"


def _row(c: dict) -> str:
    t = c.get("timing") or {}
    st = t.get("stages_s") or {}
    res = c.get("resources") or {}
    vram = res.get("vram") or {}
    shared = res.get("shared_gpu") or {}
    out = c.get("output") or {}
    mo = c.get("motion") or {}
    gate = c.get("gate") or {}
    wall = c.get("wall_s")
    cells = [
        c["case_id"], c.get("memory_mode", "-"), c.get("variant") or "-", c.get("preset_label", "-"),
        c.get("status", "-"),
        _fmt(wall / 60 if isinstance(wall, (int, float)) else None, " min"),
        _fmt(st.get("text_encode"), " s"),
        _fmt(t.get("sampler_warmup_s"), " s"),
        _fmt(t.get("s_per_step"), " s", 2),
        _fmt(st.get("vae_decode"), " s"),
        f"{_fmt(vram.get('peak_mb'))} / {_fmt(vram.get('total_mb'))}" if vram.get("available") else "n/a",
        _fmt(shared.get("delta_mb"), " MB") if shared.get("available") else "n/a",
        f"{_fmt(res.get('comfy_rss_peak_mb'))} / {_fmt(res.get('sys_ram_used_peak_mb'))}",
        f"{_fmt(out.get('duration_s'), ' s', 2)} @ {_fmt(out.get('fps'))}" if out else "-",
        _fmt(mo.get("local_motion_p98"), "", 2),
        _fmt(mo.get("duplicate_frame_ratio"), "", 2),
        "PASS" if gate.get("passed") else "FAIL",
    ]
    return "| " + " | ".join(str(x) for x in cells) + " |"


def write_report(results: dict, path: Path) -> Path:
    env = results.get("env") or {}
    gpu = env.get("gpu") or {}
    comfy = (results.get("comfyui") or {}).get("system") or {}
    gate = results.get("gate") or {}
    verdict = results.get("verdict") or {}
    sel = results.get("selection") or {}
    p = sel.get("preset")
    preset_label = f"{p['width']}x{p['height']}x{p['frames']}f" if p else "-"

    lines = [
        f"# Video benchmark: {results.get('model_label')}",
        "",
        f"- Run: `{path.parent.name}` (started {results.get('started_at')}, finished {results.get('finished_at') or 'in progress'})",
        f"- GPU: {gpu.get('name', '?')}, {gpu.get('vram_total_mb', '?')} MB VRAM, driver {gpu.get('driver', '?')} (CUDA {gpu.get('driver_cuda', '?')})",
        f"- CPU: {env.get('cpu', '?')} ({env.get('cpu_cores', '?')} cores / {env.get('cpu_threads', '?')} threads), RAM {env.get('ram_total_gb', '?')} GB, {env.get('os', '?')}",
        f"- ComfyUI {comfy.get('comfyui_version', '?')} (pinned {env.get('comfyui_pinned', '?')}), PyTorch {comfy.get('pytorch_version', '?')}, Python {comfy.get('python_version', '?')}",
        "",
        "## Verdict",
        "",
        f"- Automatic gate: **{'PASS' if verdict.get('gate_passed') else 'FAIL'}** "
        f"({verdict.get('cases_passed', 0)}/{verdict.get('cases_run', 0)} cases passed)",
        f"- Selected memory mode: {sel.get('memory_mode', '-')}; largest preset within time gate: {preset_label}",
        f"- Gate: <= {gate.get('max_minutes_per_clip')} min/clip, >= {gate.get('min_clip_seconds')} s, "
        f"local motion p98 >= {gate.get('min_local_motion_p98')} (provisional), "
        f"duplicate frames <= {gate.get('max_duplicate_frame_ratio')}, no VRAM spill",
        f"- {verdict.get('note', '')}",
        "",
        "## Cases",
        "",
        "| case | memory mode | variant | preset | status | wall | text enc | sampler warmup | s/step | VAE decode | peak VRAM / total (MB) | shared GPU mem growth | RAM peak ComfyUI / system (MB) | clip | local motion p98 | dup ratio | gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_row(c) for c in results.get("cases") or []]

    failures = [c for c in results.get("cases") or [] if not (c.get("gate") or {}).get("passed")]
    if failures:
        lines += ["", "## Why cases failed", ""]
        for c in failures:
            reasons = "; ".join((c.get("gate") or {}).get("reasons") or []) or c.get("status", "?")
            lines.append(f"- **{c['case_id']}**: {reasons}")
            if c.get("error"):
                lines.append(f"  - error: `{c['error'][:400].replace(chr(10), ' ')}`")

    sheets = [c for c in results.get("cases") or [] if c.get("contact_sheet")]
    if sheets:
        lines += ["", "## Contact sheets", ""]
        for c in sheets:
            lines.append(f"### {c['case_id']} ({c.get('prompt_id')})")
            lines.append(f"![{c['case_id']}]({c['contact_sheet']})")
            lines.append("")

    lines += [
        "",
        "## Notes",
        "",
        "- `sampler warmup` = time from sampler start to the first step minus one step: mostly moving weights to the GPU.",
        "- `shared GPU mem growth` is usually ComfyUI keeping offloaded weights in pinned system RAM "
        "(expected). It only counts as a spill when dedicated VRAM is also full; then set NVIDIA Control "
        "Panel > Manage 3D settings > Program Settings > ComfyUI's python.exe > "
        "CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback.",
        "- Clips (`*.mp4`) and frames are git-ignored; review them locally in each case folder.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

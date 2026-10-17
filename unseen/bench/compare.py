"""`unseen compare`: one HTML page that plays every benchmark clip side by side, grouped by prompt.

Open bench_results/compare.html in a browser. It only references files inside bench_results/,
so nothing is copied or uploaded.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


def _load_runs(bench_dir: Path) -> list[tuple[Path, dict]]:
    runs = []
    for res in sorted(bench_dir.glob("*/results.json")):
        try:
            runs.append((res.parent, json.loads(res.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return runs


def _caption(run: dict, case: dict) -> str:
    t = case.get("timing") or {}
    mo = case.get("motion") or {}
    vram = ((case.get("resources") or {}).get("vram") or {}).get("peak_mb")
    wall = case.get("wall_s")
    gpu = ((run.get("env") or {}).get("gpu") or {}).get("name", "?")
    parts = [
        f"<b>{html.escape(run.get('model_label', run.get('model_key', '?')))}</b>",
        html.escape(f"on {gpu}"),
        html.escape(f"variant {case.get('variant')} | {case.get('preset_label')} | {case.get('memory_mode')}"),
        html.escape(f"{wall / 60:.1f} min" if isinstance(wall, (int, float)) else "-")
        + html.escape(f" | {t.get('s_per_step')} s/step | VRAM {vram} MB"),
        html.escape(f"local motion p98 {mo.get('local_motion_p98')} | gate "
                    f"{'PASS' if (case.get('gate') or {}).get('passed') else 'FAIL'}"),
        f"<span class=dim>{html.escape(case.get('case_id', ''))}</span>",
    ]
    return "<br>".join(parts)


def write_compare(bench_dir: Path, models: list[str] | None = None, latest_only: bool = True) -> Path:
    runs = _load_runs(bench_dir)
    if models:
        runs = [(d, r) for d, r in runs if r.get("model_key") in models]
    if latest_only:  # newest run per model key (run dirs sort by timestamp)
        latest: dict[str, tuple[Path, dict]] = {}
        for d, r in runs:
            latest[r.get("model_key", d.name)] = (d, r)
        runs = list(latest.values())

    by_prompt: dict[str, list[str]] = {}
    for run_dir, run in runs:
        for case in run.get("cases") or []:
            out = case.get("output") or {}
            if case.get("status") != "ok" or not out.get("file"):
                continue
            rel = f"{run_dir.name}/{case['case_id']}/{out['file']}"
            cell = (f"<figure><video src='{html.escape(rel)}' controls loop muted playsinline "
                    f"preload=metadata></video><figcaption>{_caption(run, case)}</figcaption></figure>")
            by_prompt.setdefault(case.get("prompt_id", "?"), []).append(cell)

    sections = []
    for prompt_id, cells in sorted(by_prompt.items()):
        sections.append(f"<h2>{html.escape(prompt_id)}</h2><div class=grid>{''.join(cells)}</div>")
    body = "".join(sections) or "<p>No finished clips found. Run <code>uv run unseen bench-video ...</code> first.</p>"
    page = f"""<!doctype html><html><head><meta charset=utf-8><title>Unseen Archive - clip comparison</title>
<style>
body{{background:#111;color:#ddd;font:14px/1.4 system-ui,sans-serif;margin:20px}}
h1{{font-weight:600}} h2{{margin-top:32px;border-bottom:1px solid #333;padding-bottom:4px}}
.grid{{display:flex;flex-wrap:wrap;gap:16px}}
figure{{margin:0;width:300px}} video{{width:300px;background:#000;border:1px solid #333}}
figcaption{{font-size:12px;margin-top:4px}} .dim{{color:#777}}
</style></head><body>
<h1>Clip comparison</h1>
<p class=dim>Latest run per model{', filtered' if models else ''}. Clips play muted and loop; compare motion, coherence and how believable each looks.
Raw model output: grain/VHS/timestamps are added later in post-production.</p>
{body}
<script>
// play all videos in sync on demand
document.addEventListener('keydown', e => {{ if (e.key === ' ') {{ e.preventDefault();
  const v=[...document.querySelectorAll('video')]; const play=v.some(x=>x.paused);
  v.forEach(x=>{{x.currentTime=0; play?x.play():x.pause();}}); }} }});
</script>
<p class=dim>Tip: press Space to restart and play all clips together.</p>
</body></html>"""
    out = bench_dir / "compare.html"
    out.write_text(page, encoding="utf-8")
    return out

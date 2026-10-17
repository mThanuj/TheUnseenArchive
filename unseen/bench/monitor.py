"""Background sampling of VRAM, shared GPU memory and system RAM during one generation.

Dedicated VRAM comes from NVML (device-wide, so the desktop's own usage is included; the
baseline before the job is recorded too). On Windows the NVIDIA driver can silently spill to
"shared GPU memory" (system RAM) instead of failing with out-of-memory, which makes generation
10-50x slower; that is detected heuristically from the "GPU Adapter Memory" performance counters
via typeperf. Every metric is best-effort: anything unavailable is reported as such, never guessed.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import threading
import time

import psutil

MB = 1024 * 1024


class _NvmlSampler:
    def __init__(self) -> None:
        self.available = False
        self.reason = ""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.total_mb = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total / MB
            self.available = True
        except Exception as exc:  # no NVIDIA driver / library
            self.reason = repr(exc)

    def sample(self) -> tuple[float, float] | None:
        if not self.available:
            return None
        try:
            used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / MB
            util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            return used, util
        except Exception:
            return None


_COUNTER_RE = re.compile(r"\((?P<inst>[^()]*)\)\\(?P<kind>Dedicated|Shared) Usage$", re.IGNORECASE)


class SharedGpuMemorySampler(threading.Thread):
    """Streams per-adapter Dedicated/Shared usage from `typeperf` (Windows only)."""

    COUNTERS = [r"\GPU Adapter Memory(*)\Dedicated Usage", r"\GPU Adapter Memory(*)\Shared Usage"]

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.first: dict[tuple[str, str], float] = {}
        self.peak: dict[tuple[str, str], float] = {}
        self.error: str | None = None
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ["typeperf", *self.COUNTERS, "-si", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.error = f"typeperf unavailable: {exc!r}"
            return
        columns: list[tuple[str, str] | None] | None = None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line.startswith('"'):
                if "error" in line.lower() and not self.peak:
                    self.error = line
                continue
            row = next(csv.reader([line]))
            if columns is None:
                columns = []
                for name in row[1:]:
                    m = _COUNTER_RE.search(name)
                    columns.append((m["inst"], m["kind"].lower()) if m else None)
                continue
            for col, raw in zip(columns, row[1:]):
                if col is None:
                    continue
                try:
                    txt = raw.strip()
                    if "," in txt and "." not in txt:  # decimal comma locales
                        txt = txt.replace(",", ".")
                    value = float(txt) / MB
                except ValueError:
                    continue
                self.first.setdefault(col, value)
                self.peak[col] = max(value, self.peak.get(col, value))

    def stop(self) -> dict:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.join(timeout=5)
        dedicated = {inst: v for (inst, kind), v in self.peak.items() if kind == "dedicated"}
        if not dedicated:
            return {"available": False, "reason": self.error or "no counter samples (non-English Windows?)"}
        # The discrete NVIDIA adapter is the one with by far the most dedicated memory in use.
        adapter = max(dedicated, key=dedicated.get)
        key = (adapter, "shared")
        base = self.first.get(key, 0.0)
        peak = self.peak.get(key, base)
        return {"available": True, "adapter": adapter, "baseline_mb": round(base),
                "peak_mb": round(peak), "delta_mb": round(peak - base)}


class ResourceMonitor:
    def __init__(self, pid: int | None, interval_s: float = 0.5):
        self.pid = pid
        self.interval_s = interval_s
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._nvml = _NvmlSampler()
        self._shared = SharedGpuMemorySampler() if os.name == "nt" else None
        self.vram_baseline_mb: float | None = None
        self.vram_peak_mb = 0.0
        self.util: list[float] = []
        self.rss_peak_mb = 0.0
        self.sys_used_peak_mb = 0.0
        self.sys_avail_min_mb = float("inf")

    def start(self) -> "ResourceMonitor":
        first = self._nvml.sample()
        if first:
            self.vram_baseline_mb = first[0]
        if self._shared:
            self._shared.start()
        self._thread.start()
        return self

    def _proc_rss_mb(self) -> float:
        if not self.pid:
            return 0.0
        try:
            proc = psutil.Process(self.pid)
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            return total / MB
        except psutil.Error:
            return 0.0

    def _loop(self) -> None:
        while not self._halt.is_set():
            s = self._nvml.sample()
            if s:
                self.vram_peak_mb = max(self.vram_peak_mb, s[0])
                self.util.append(s[1])
            self.rss_peak_mb = max(self.rss_peak_mb, self._proc_rss_mb())
            vm = psutil.virtual_memory()
            self.sys_used_peak_mb = max(self.sys_used_peak_mb, (vm.total - vm.available) / MB)
            self.sys_avail_min_mb = min(self.sys_avail_min_mb, vm.available / MB)
            time.sleep(self.interval_s)

    def stop(self) -> dict:
        self._halt.set()
        self._thread.join(timeout=5)
        out: dict = {
            "vram": ({"available": True, "total_mb": round(self._nvml.total_mb),
                      "baseline_mb": round(self.vram_baseline_mb or 0),
                      "peak_mb": round(self.vram_peak_mb),
                      "gpu_util_mean": round(sum(self.util) / len(self.util), 1) if self.util else None,
                      "gpu_util_max": max(self.util) if self.util else None}
                     if self._nvml.available else {"available": False, "reason": self._nvml.reason}),
            "comfy_rss_peak_mb": round(self.rss_peak_mb),
            "sys_ram_total_mb": round(psutil.virtual_memory().total / MB),
            "sys_ram_used_peak_mb": round(self.sys_used_peak_mb),
            "sys_ram_available_min_mb": (round(self.sys_avail_min_mb)
                                         if self.sys_avail_min_mb != float("inf") else None),
        }
        out["shared_gpu"] = (self._shared.stop() if self._shared
                             else {"available": False, "reason": "only measured on Windows"})
        return out

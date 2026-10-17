"""Typed access to the YAML files in config/.

All files are read as UTF-8 explicitly: Windows defaults to cp1252, which would corrupt the
non-ASCII negative prompt in bench.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing variables."""
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------- runtime.yaml

class ComfyUISettings(BaseModel):
    repo: str
    ref: str
    commit: str
    python: str
    torch_index_url: str
    min_driver_cuda: str
    host: str = "127.0.0.1"
    port: int = 8188
    startup_timeout_s: int = 300
    base_args: list[str] = Field(default_factory=list)
    memory_modes: dict[str, list[str]]


class CustomNode(BaseModel):
    name: str
    repo: str
    commit: str


class RuntimePaths(BaseModel):
    comfyui_dir: str
    comfy_venv: str
    comfy_work: str
    models_dir: str
    bench_dir: str


class RuntimeConfig(BaseModel):
    comfyui: ComfyUISettings
    custom_nodes: list[CustomNode]
    paths: RuntimePaths

    def path(self, name: str) -> Path:
        return (REPO_ROOT / getattr(self.paths, name)).resolve()

    def comfy_python(self) -> Path:
        venv = self.path("comfy_venv")
        if os.name == "nt":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"


# ----------------------------------------------------------------- models.yaml

MODEL_DESTS = ("diffusion_models", "text_encoders", "vae")


class LicenseInfo(BaseModel):
    name: str
    url: str
    commercial_use: str
    obligations: str = ""


class ModelFile(BaseModel):
    repo: str
    path: str
    dest: str
    size: int
    sha256: str
    license: str

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]


class ModelsManifest(BaseModel):
    licenses: dict[str, LicenseInfo]
    files: dict[str, ModelFile]
    sets: dict[str, list[str]]

    def local_path(self, rt: RuntimeConfig, key: str) -> Path:
        f = self.files[key]
        return rt.path("models_dir") / f.dest / f.filename


# ------------------------------------------------------------------ bench.yaml

class Preset(BaseModel):
    width: int
    height: int
    frames: int

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}x{self.frames}f"


class Constraints(BaseModel):
    dim_multiple: int
    frame_step: int


class BenchModel(BaseModel):
    label: str
    workflow: str
    files: dict[str, str]
    fps: float
    steps: int
    cfg: float
    sampler: str
    scheduler: str = "simple"
    shift: float | None = None
    negative_prompt: str | None = None      # overrides BenchConfig.negative_prompt
    constraints: Constraints
    variants: list[dict[str, Any]]
    quick: Preset
    ladder: list[Preset]
    floor: list[Preset] = Field(default_factory=list)


class BenchPrompt(BaseModel):
    id: str
    seed: int
    text: str


class Gate(BaseModel):
    max_minutes_per_clip: float
    min_clip_seconds: float
    min_local_motion_p98: float
    max_duplicate_frame_ratio: float
    spill_threshold_mb: float


class VaeTiling(BaseModel):
    tile_size: int
    overlap: int
    temporal_size: int
    temporal_overlap: int


class BenchConfig(BaseModel):
    gate: Gate
    case_timeout_minutes: float
    memory_modes: list[str]
    fallback_memory_modes: list[str] = Field(default_factory=list)
    negative_prompt: str
    prompts: list[BenchPrompt]
    models: dict[str, BenchModel]
    vae_tiling: VaeTiling


def load_runtime(path: Path | None = None) -> RuntimeConfig:
    return RuntimeConfig.model_validate(load_yaml(path or CONFIG_DIR / "runtime.yaml"))


def load_models(path: Path | None = None) -> ModelsManifest:
    return ModelsManifest.model_validate(load_yaml(path or CONFIG_DIR / "models.yaml"))


def load_bench(path: Path | None = None) -> BenchConfig:
    return BenchConfig.model_validate(load_yaml(path or CONFIG_DIR / "bench.yaml"))

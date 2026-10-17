"""Anti-repetition: semantic similarity of a new concept to every past story.

Uses fastembed's ONNX build of BAAI/bge-small-en-v1.5 (MIT, ~67 MB, CPU only). The model is
downloaded on first use into cache/fastembed/.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import REPO_ROOT

MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self) -> None:
        from fastembed import TextEmbedding  # heavy import, only when needed
        self._model = TextEmbedding(model_name=MODEL, cache_dir=str(REPO_ROOT / "cache" / "fastembed"))

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def concept_text(title: str, logline: str, twist: str) -> str:
    return f"{title}. {logline} Twist: {twist}"


def max_similarity(vec: np.ndarray, history: list[np.ndarray]) -> float:
    if not history:
        return 0.0
    mat = np.stack(history)
    return float(np.max(mat @ vec))

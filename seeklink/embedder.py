"""Embedding wrapper — lazy-loaded FastEmbed for document/query encoding."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np


def _default_cache_dir() -> Path:
    # On macOS the system default $TMPDIR is under /var/folders/.../T/, which
    # launchd periodically purges — breaking the ONNX blob symlinks. Pin the
    # fastembed cache to a persistent user directory instead.
    override = os.environ.get("SEEKLINK_FASTEMBED_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "fastembed"


class Embedder:
    """Lazy-loaded FastEmbed wrapper for jina-embeddings-v2-base-zh.

    Thread-safe: model loads once on first use via double-checked locking.
    """

    MODEL_NAME = "jinaai/jina-embeddings-v2-base-zh"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from fastembed import TextEmbedding

            cache_dir = _default_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.MODEL_NAME,
                cache_dir=str(cache_dir),
            )

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        """Embed document texts with 'passage: ' prefix. Returns float32 bytes."""
        self._ensure_model()
        embeddings = list(self._model.passage_embed(texts))
        return [e.astype(np.float32).tobytes() for e in embeddings]

    def embed_query(self, text: str) -> bytes:
        """Embed a single query with 'query: ' prefix. Returns float32 bytes."""
        self._ensure_model()
        embeddings = list(self._model.query_embed(text))
        return embeddings[0].astype(np.float32).tobytes()

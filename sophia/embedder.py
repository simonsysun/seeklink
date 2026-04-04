"""Embedding wrapper — lazy-loaded FastEmbed for document/query encoding."""

from __future__ import annotations

import threading

import numpy as np


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

            self._model = TextEmbedding(model_name=self.MODEL_NAME)

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

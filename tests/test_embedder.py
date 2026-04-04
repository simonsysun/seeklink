"""Tests for sophia.embedder — requires model download, session-scoped fixture."""

from __future__ import annotations

import numpy as np
import pytest

from sophia.embedder import Embedder

FLOAT32_DIM = 768
FLOAT32_BYTES = FLOAT32_DIM * 4  # 3072


@pytest.fixture(scope="session")
def embedder():
    """Session-scoped embedder — model loads once for all tests."""
    return Embedder()


class TestEmbedder:
    def test_lazy_loading(self):
        """Model should be None before first use."""
        e = Embedder()
        assert e._model is None

    def test_embed_documents_returns_bytes(self, embedder: Embedder):
        result = embedder.embed_documents(["Hello world."])
        assert len(result) == 1
        assert isinstance(result[0], bytes)
        assert len(result[0]) == FLOAT32_BYTES

    def test_embed_documents_batch(self, embedder: Embedder):
        texts = ["First text.", "Second text.", "Third text."]
        result = embedder.embed_documents(texts)
        assert len(result) == 3
        for emb in result:
            assert len(emb) == FLOAT32_BYTES

    def test_embed_query_returns_bytes(self, embedder: Embedder):
        result = embedder.embed_query("search query")
        assert isinstance(result, bytes)
        assert len(result) == FLOAT32_BYTES

    def test_document_and_query_both_work(self, embedder: Embedder):
        """Both embed_documents and embed_query produce valid 768d vectors."""
        text = "Machine learning fundamentals"
        doc_emb = embedder.embed_documents([text])[0]
        query_emb = embedder.embed_query(text)
        assert len(doc_emb) == FLOAT32_BYTES
        assert len(query_emb) == FLOAT32_BYTES

    def test_similar_texts_closer(self, embedder: Embedder):
        """Semantically similar texts should have higher cosine similarity."""
        embs = embedder.embed_documents([
            "Machine learning and artificial intelligence",
            "Deep learning neural networks",
            "Cooking recipes for Italian pasta",
        ])

        def cosine_sim(a: bytes, b: bytes) -> float:
            va = np.frombuffer(a, dtype=np.float32)
            vb = np.frombuffer(b, dtype=np.float32)
            return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

        sim_ml = cosine_sim(embs[0], embs[1])  # ML topics
        sim_diff = cosine_sim(embs[0], embs[2])  # ML vs cooking
        assert sim_ml > sim_diff

    def test_chinese_text(self, embedder: Embedder):
        result = embedder.embed_documents(["深度学习使用反向传播算法训练神经网络"])
        assert len(result) == 1
        assert len(result[0]) == FLOAT32_BYTES

    def test_model_loaded_after_use(self, embedder: Embedder):
        """After first embed call, model should be loaded."""
        assert embedder._model is not None

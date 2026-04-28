"""Tests for the MLX reranker wrapper without loading real MLX models."""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

import seeklink.reranker as reranker_mod
from seeklink.reranker import Reranker


@pytest.fixture
def fake_mlx(monkeypatch):
    mlx_module = types.ModuleType("mlx")
    core_module = types.ModuleType("mlx.core")
    core_module.array = lambda value: np.array(value, dtype=np.int64)
    core_module.eval = lambda *args, **kwargs: None
    mlx_module.core = core_module
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", core_module)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"yes": 1, "no": 2}[token]

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]

    def encode(self, text: str, return_tensors=None) -> list[int]:
        assert return_tensors is None
        if "Document: " not in text:
            return [1] * len(text)
        passage = text.split("Document: ", 1)[1].split("<think>", 1)[0]
        marker = max(1, len(passage))
        return [1] * (3 + len(passage)) + [marker]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return "x" * len(tokens)


class RecordingModel:
    def __init__(self, *, fail_all: bool = False):
        self.fail_all = fail_all
        self.shapes: list[tuple[int, int]] = []

    def __call__(self, input_ids):
        arr = np.asarray(input_ids)
        self.shapes.append(tuple(arr.shape))
        if self.fail_all:
            raise RuntimeError("fake model failure")

        logits = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
        for row_index, row in enumerate(arr):
            non_padding = np.flatnonzero(row != 0)
            last_real = int(non_padding[-1])
            marker = float(row[last_real])
            logits[row_index, last_real, 1] = marker
            logits[row_index, last_real, 2] = 0.0
            if last_real != arr.shape[1] - 1:
                logits[row_index, -1, 1] = -100.0
                logits[row_index, -1, 2] = 100.0
        return logits


def _ready_reranker(model: RecordingModel) -> Reranker:
    reranker = Reranker()
    reranker._model = model
    reranker._tokenizer = FakeTokenizer()
    reranker._token_yes = 1
    reranker._token_no = 2
    return reranker


def _sigmoid(value: float) -> float:
    return math.exp(value) / (math.exp(value) + 1.0)


def test_rerank_caps_long_passages_before_scoring(fake_mlx, monkeypatch):
    monkeypatch.setattr(reranker_mod, "_MAX_PASSAGE_TOKENS", 2)
    model = RecordingModel()
    reranker = _ready_reranker(model)

    scores = reranker.rerank("query", ["abcdef"])

    assert scores == pytest.approx([_sigmoid(2)])
    assert model.shapes == [(1, 6)]


def test_rerank_keeps_short_passages_intact(fake_mlx, monkeypatch):
    monkeypatch.setattr(reranker_mod, "_MAX_PASSAGE_TOKENS", 10)
    model = RecordingModel()
    reranker = _ready_reranker(model)

    scores = reranker.rerank("query", ["abc"])

    assert scores == pytest.approx([_sigmoid(3)])
    assert model.shapes == [(1, 7)]


def test_rerank_returns_none_when_inference_fails(fake_mlx):
    reranker = _ready_reranker(RecordingModel(fail_all=True))

    assert reranker.rerank("query", ["passage"]) is None

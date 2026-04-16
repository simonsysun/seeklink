"""Reranker — Qwen3-Reranker-0.6B via MLX on Apple Silicon.

Cross-encoder reranking is a precision booster applied after first-stage
4-channel RRF retrieval. For each (query, passage) pair in the top
candidates, the reranker model reads query and passage together with
full cross-attention and outputs a relevance score. This is more
accurate than bi-encoder vector similarity (which encodes query and
passage independently), at the cost of ~60ms per pair.

Implementation uses MLX (Apple's native ML framework) which runs on
Metal GPU, achieving ~1.2s for 20 pairs on M3 Air. The model uses a
yes/no prompt format per Qwen3-Reranker's official usage guide:
the model outputs logits for 'yes' and 'no' tokens, and we convert
the yes-probability to a relevance score.

Default model: mlx-community/Qwen3-Reranker-0.6B-mxfp8 (~700MB, 100+
languages including Chinese, Apache 2.0). Override via
SEEKLINK_RERANKER_MODEL env var. Set to empty string to disable.
"""

from __future__ import annotations

import logging
import math
import os
import threading

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"
_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
)


class Reranker:
    """MLX-based Qwen3 reranker. Lazy-loaded, thread-safe.

    Failures during model load or inference downgrade gracefully — the
    reranker self-disables and callers get None back so they can fall
    back to the first-stage ordering instead of crashing the search.
    """

    MODEL_NAME = os.environ.get("SEEKLINK_RERANKER_MODEL", _DEFAULT_MODEL)

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._token_yes: int | None = None
        self._token_no: int | None = None
        self._lock = threading.Lock()
        self._disabled = self.MODEL_NAME == ""

    @property
    def disabled(self) -> bool:
        return self._disabled

    def _ensure_model(self) -> None:
        if self._disabled or self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import mlx_lm

                self._model, self._tokenizer = mlx_lm.load(self.MODEL_NAME)
                self._token_yes = self._tokenizer.convert_tokens_to_ids("yes")
                self._token_no = self._tokenizer.convert_tokens_to_ids("no")
            except Exception as e:
                logger.warning(
                    "Reranker load failed (%s): %s — reranking disabled",
                    self.MODEL_NAME,
                    e,
                )
                self._disabled = True

    def _score_one(self, query: str, passage: str) -> float:
        """Score a single (query, passage) pair. Returns 0-1 probability."""
        import mlx.core as mx

        prompt = (
            f"Instruct: {_DEFAULT_INSTRUCTION}\n"
            f"Query: {query}\n"
            f"Document: {passage}"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text += "<think>\n"

        tokens = self._tokenizer.encode(text, return_tensors=None)
        if isinstance(tokens, list):
            input_ids = mx.array([tokens])
        else:
            input_ids = mx.array(tokens)
            if input_ids.ndim == 1:
                input_ids = input_ids[None]

        logits = self._model(input_ids)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits)

        yes_s = last_logits[self._token_yes].item()
        no_s = last_logits[self._token_no].item()
        return math.exp(yes_s) / (math.exp(yes_s) + math.exp(no_s))

    def rerank(
        self, query: str, passages: list[str]
    ) -> list[float] | None:
        """Score each passage against the query.

        Returns a list of floats (higher = more relevant) in the same
        order as the input passages. Returns None if the reranker is
        disabled or inference fails — callers should fall back to the
        first-stage ordering in that case.
        """
        if not passages:
            return []
        self._ensure_model()
        if self._disabled or self._model is None:
            return None
        try:
            return [self._score_one(query, p) for p in passages]
        except Exception as e:
            logger.warning(
                "Reranker inference failed: %s — falling back to first-stage",
                e,
            )
            return None

"""Reranker — Qwen3-Reranker-0.6B via MLX on Apple Silicon.

Cross-encoder reranking is a precision booster applied after first-stage
4-channel RRF retrieval. For each (query, passage) pair in the top
candidates, the reranker model reads query and passage together with
full cross-attention and outputs a relevance score. This is more
accurate than bi-encoder vector similarity (which encodes query and
passage independently), at the cost of ~60ms per pair.

Implementation uses MLX (Apple's native ML framework) which runs on
Metal GPU. Candidate passages are capped before scoring so the reranker
does not spend most of its time on long chunks whose opening text is
already enough for relevance judgments. The model uses a yes/no prompt
format per Qwen3-Reranker's official usage guide:
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
_MAX_PASSAGE_TOKENS = 200
_SCORING_MODE_ENV = "SEEKLINK_RERANK_SCORING"
_SCORING_AUTO = "auto"
_SCORING_CLS_HEAD = "cls_head"
_SCORING_LEGACY = "legacy"
_VALID_SCORING_MODES = {_SCORING_AUTO, _SCORING_CLS_HEAD, _SCORING_LEGACY}


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
        self._cls_weight = None
        self._lock = threading.Lock()
        self._disabled = self.MODEL_NAME == ""
        self._scoring_mode = (
            os.environ.get(_SCORING_MODE_ENV, _SCORING_AUTO)
            .strip()
            .casefold()
            .replace("-", "_")
        )
        if self._scoring_mode not in _VALID_SCORING_MODES:
            logger.warning(
                "Unsupported %s=%r; using %s",
                _SCORING_MODE_ENV,
                self._scoring_mode,
                _SCORING_AUTO,
            )
            self._scoring_mode = _SCORING_AUTO

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
                if self._scoring_mode != _SCORING_LEGACY:
                    self._prepare_cls_head()
            except Exception as e:
                logger.warning(
                    "Reranker load failed (%s): %s — reranking disabled",
                    self.MODEL_NAME,
                    e,
                )
                self._disabled = True

    def _prepare_cls_head(self) -> None:
        """Prepare a two-token classifier head when the MLX model supports it."""
        try:
            import mlx.core as mx

            if self._model is None or self._token_yes is None or self._token_no is None:
                return
            body = getattr(self._model, "model", None)
            embed_tokens = getattr(body, "embed_tokens", None)
            if body is None or embed_tokens is None:
                if self._scoring_mode == _SCORING_CLS_HEAD:
                    logger.warning(
                        "Reranker cls_head scoring unavailable; using legacy logits"
                    )
                return
            yes_no = embed_tokens(mx.array([self._token_yes, self._token_no]))
            self._cls_weight = yes_no[0] - yes_no[1]
            mx.eval(self._cls_weight)
        except Exception as e:
            self._cls_weight = None
            if self._scoring_mode == _SCORING_CLS_HEAD:
                logger.warning(
                    "Reranker cls_head scoring failed to initialize (%s); "
                    "using legacy logits",
                    e,
                )

    def _token_list(self, text: str) -> list[int]:
        """Tokenize text into a flat Python list."""
        tokens = self._tokenizer.encode(text, return_tensors=None)
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        return list(tokens)

    def _truncate_passage(self, passage: str) -> str:
        """Cap passage text used by the reranker; display text remains full."""
        tokens = self._token_list(passage)
        if len(tokens) <= _MAX_PASSAGE_TOKENS:
            return passage

        head = tokens[:_MAX_PASSAGE_TOKENS]
        decode = getattr(self._tokenizer, "decode", None)
        if decode is not None:
            try:
                return decode(head, skip_special_tokens=True)
            except TypeError:
                return decode(head)
            except Exception:
                logger.debug("Reranker passage decode failed; using char fallback")

        # Conservative fallback for unusual tokenizers without decode().
        return passage[:1200]

    def _pair_tokens(self, query: str, passage: str) -> list[int]:
        passage = self._truncate_passage(passage)
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

        tokens = self._token_list(text)
        return tokens

    def _score_one_cls_head(self, tokens: list[int]) -> float | None:
        """Score via final hidden state dot (yes_embedding - no_embedding)."""
        if self._cls_weight is None:
            return None

        import mlx.core as mx

        input_ids = mx.array([tokens])
        hidden = self._model.model(input_ids)
        last_h = hidden[0, -1, :]
        logit = mx.sum(last_h * self._cls_weight)
        mx.eval(logit)
        value = logit.item()
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _score_one_legacy(self, tokens: list[int]) -> float:
        """Score via full-vocabulary yes/no token logits."""
        import mlx.core as mx

        input_ids = mx.array([tokens])
        logits = self._model(input_ids)
        last_logits = logits[0, -1, :]
        mx.eval(last_logits)

        yes_s = last_logits[self._token_yes].item()
        no_s = last_logits[self._token_no].item()
        max_s = max(yes_s, no_s)
        yes_e = math.exp(yes_s - max_s)
        no_e = math.exp(no_s - max_s)
        return yes_e / (yes_e + no_e)

    def _score_one(self, query: str, passage: str) -> float:
        """Score a single (query, passage) pair. Returns 0-1 probability."""
        tokens = self._pair_tokens(query, passage)
        if self._scoring_mode != _SCORING_LEGACY:
            score = self._score_one_cls_head(tokens)
            if score is not None:
                return score
        return self._score_one_legacy(tokens)

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

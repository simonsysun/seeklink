"""Retrieval metrics for SeekLink's blind-test framework."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _expected_set(expected: Sequence[str]) -> set[str]:
    return {path for path in expected if path}


def hit_ranks(
    hits: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
) -> list[int]:
    """Return 1-indexed ranks where expected paths appear in the top k.

    Duplicate hit paths count only once, at their first occurrence.
    """
    if k <= 0:
        return []
    expected_paths = _expected_set(expected)
    if not expected_paths:
        return []

    seen: set[str] = set()
    ranks: list[int] = []
    for rank, path in enumerate(hits[:k], start=1):
        if path in seen:
            continue
        seen.add(path)
        if path in expected_paths:
            ranks.append(rank)
    return ranks


def recall_at_k(
    hits: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
) -> float:
    """Fraction of expected paths found in top k."""
    expected_paths = _expected_set(expected)
    if not expected_paths or k <= 0:
        return 0.0
    found = {hits[rank - 1] for rank in hit_ranks(hits, expected, k=k)}
    return len(found) / len(expected_paths)


def reciprocal_rank(
    hits: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
) -> float:
    """MRR contribution for one query: reciprocal rank of first hit."""
    ranks = hit_ranks(hits, expected, k=k)
    if not ranks:
        return 0.0
    return 1.0 / ranks[0]


def precision_at_k(
    hits: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
) -> float:
    """Precision@k using k as denominator, so missing results are penalized."""
    if k <= 0:
        return 0.0
    return len(hit_ranks(hits, expected, k=k)) / k


def average_precision_at_k(
    hits: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
) -> float:
    """Average precision at k for one query.

    Duplicate hit paths count only at first occurrence. The denominator is
    ``min(number of relevant paths, k)``, matching common AP@k practice.
    """
    expected_paths = _expected_set(expected)
    if not expected_paths or k <= 0:
        return 0.0

    seen: set[str] = set()
    relevant_seen = 0
    precision_sum = 0.0
    for rank, path in enumerate(hits[:k], start=1):
        if path in seen:
            continue
        seen.add(path)
        if path in expected_paths:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    denominator = min(len(expected_paths), k)
    return precision_sum / denominator if denominator else 0.0


def dcg_at_k(gains: Sequence[float], *, k: int = 10) -> float:
    """Discounted cumulative gain for graded relevance values."""
    if k <= 0:
        return 0.0
    total = 0.0
    for rank, gain in enumerate(gains[:k], start=1):
        if gain <= 0:
            continue
        total += (math.pow(2.0, gain) - 1.0) / math.log2(rank + 1)
    return total


def ndcg_at_k(
    hits: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
    relevance: Mapping[str, float] | None = None,
) -> float:
    """nDCG@k for binary or graded relevance.

    If ``relevance`` is omitted, every expected path has grade 1. If it is
    provided, expected paths missing from the mapping also receive grade 1.
    Extra paths in the mapping are allowed and count as relevant labels.
    """
    if k <= 0:
        return 0.0

    grades: dict[str, float] = {path: 1.0 for path in expected if path}
    if relevance is not None:
        grades.update({path: float(grade) for path, grade in relevance.items()})

    if not grades:
        return 0.0

    seen: set[str] = set()
    gains: list[float] = []
    for path in hits[:k]:
        if path in seen:
            gains.append(0.0)
            continue
        seen.add(path)
        gains.append(grades.get(path, 0.0))

    ideal_gains = sorted((g for g in grades.values() if g > 0), reverse=True)
    ideal = dcg_at_k(ideal_gains, k=k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(gains, k=k) / ideal


def last_expected_rank(
    hits: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
) -> int | None:
    """Rank of the lowest-ranked expected hit in top k, or None on miss."""
    ranks = hit_ranks(hits, expected, k=k)
    return ranks[-1] if ranks else None

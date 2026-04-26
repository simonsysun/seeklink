"""Tests for blind-test retrieval metrics."""

from __future__ import annotations

import pytest

from tests.blind.metrics import (
    average_precision_at_k,
    dcg_at_k,
    hit_ranks,
    last_expected_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestHitRanks:
    def test_returns_one_indexed_ranks(self):
        hits = ["a.md", "b.md", "c.md", "d.md"]
        assert hit_ranks(hits, ["c.md", "a.md"], k=4) == [1, 3]

    def test_respects_k(self):
        assert hit_ranks(["a.md", "b.md"], ["b.md"], k=1) == []

    def test_duplicate_hit_counts_once(self):
        hits = ["a.md", "a.md", "b.md"]
        assert hit_ranks(hits, ["a.md", "b.md"], k=3) == [1, 3]


class TestRecallAtK:
    def test_fraction_of_expected_paths_found(self):
        hits = ["a.md", "b.md", "c.md"]
        assert recall_at_k(hits, ["a.md", "c.md", "missing.md"], k=3) == pytest.approx(2 / 3)

    def test_empty_expected_is_zero(self):
        assert recall_at_k(["a.md"], [], k=10) == 0.0

    def test_non_positive_k_is_zero(self):
        assert recall_at_k(["a.md"], ["a.md"], k=0) == 0.0


class TestReciprocalRank:
    def test_first_expected_hit(self):
        hits = ["a.md", "b.md", "c.md"]
        assert reciprocal_rank(hits, ["c.md", "b.md"], k=3) == pytest.approx(0.5)

    def test_miss_is_zero(self):
        assert reciprocal_rank(["a.md"], ["b.md"], k=10) == 0.0


class TestPrecisionAtK:
    def test_uses_k_as_denominator(self):
        hits = ["a.md", "b.md"]
        assert precision_at_k(hits, ["a.md"], k=5) == pytest.approx(0.2)

    def test_duplicate_hit_does_not_double_count(self):
        hits = ["a.md", "a.md", "b.md"]
        assert precision_at_k(hits, ["a.md", "b.md"], k=3) == pytest.approx(2 / 3)


class TestAveragePrecisionAtK:
    def test_average_precision_for_multiple_hits(self):
        hits = ["a.md", "x.md", "b.md", "c.md"]
        expected = ["a.md", "b.md", "c.md"]
        # Precision at relevant ranks: 1/1, 2/3, 3/4.
        assert average_precision_at_k(hits, expected, k=4) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)

    def test_denominator_caps_at_k(self):
        hits = ["a.md"]
        expected = ["a.md", "b.md", "c.md"]
        assert average_precision_at_k(hits, expected, k=1) == pytest.approx(1.0)

    def test_miss_is_zero(self):
        assert average_precision_at_k(["x.md"], ["a.md"], k=10) == 0.0


class TestNdcgAtK:
    def test_binary_ndcg_perfect_is_one(self):
        hits = ["a.md", "b.md"]
        assert ndcg_at_k(hits, ["a.md", "b.md"], k=2) == pytest.approx(1.0)

    def test_binary_ndcg_penalizes_late_hits(self):
        hits = ["x.md", "a.md"]
        score = ndcg_at_k(hits, ["a.md"], k=2)
        assert 0.0 < score < 1.0

    def test_graded_relevance(self):
        hits = ["support.md", "answer.md", "related.md"]
        relevance = {"answer.md": 3, "support.md": 2, "related.md": 1}
        score = ndcg_at_k(hits, [], k=3, relevance=relevance)
        assert 0.0 < score < 1.0

    def test_extra_relevance_labels_count(self):
        hits = ["answer.md"]
        assert ndcg_at_k(hits, [], k=1, relevance={"answer.md": 3}) == pytest.approx(1.0)


class TestDcgAtK:
    def test_dcg_uses_exponential_gain(self):
        assert dcg_at_k([3], k=1) == pytest.approx(7.0)

    def test_non_positive_k_is_zero(self):
        assert dcg_at_k([3], k=0) == 0.0


class TestLastExpectedRank:
    def test_returns_lowest_ranked_hit(self):
        hits = ["a.md", "x.md", "b.md"]
        assert last_expected_rank(hits, ["a.md", "b.md"], k=3) == 3

    def test_returns_none_on_miss(self):
        assert last_expected_rank(["x.md"], ["a.md"], k=10) is None

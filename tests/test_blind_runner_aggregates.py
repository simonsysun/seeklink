"""Tests for blind-test runner aggregation helpers."""

from __future__ import annotations

import pytest

from tests.blind.run import ResultRow, aggregate_by_tag, aggregate_rows


def _row(
    *,
    query: str,
    tags: list[str],
    recall: float,
    mrr: float,
    ndcg: float,
    p5: float,
    ap: float,
    latency: float,
) -> ResultRow:
    return ResultRow(
        query=query,
        config="A",
        tags=tags,
        hits=[],
        titles=[],
        snippets=[],
        scores=[],
        latency_ms=latency,
        reranker_active=True,
        recall_at_10=recall,
        mrr=mrr,
        precision_at_5=p5,
        average_precision_at_10=ap,
        ndcg_at_10=ndcg,
        last_expected_rank=None,
    )


class TestAggregateRows:
    def test_empty_rows_returns_zero_metrics(self):
        aggregate = aggregate_rows([])

        assert aggregate["n_queries"] == 0
        assert aggregate["mean_recall_at_10"] == 0.0
        assert aggregate["mean_mrr"] == 0.0
        assert aggregate["mean_precision_at_5"] == 0.0
        assert aggregate["mean_average_precision_at_10"] == 0.0
        assert aggregate["mean_ndcg_at_10"] == 0.0
        assert aggregate["mean_latency_ms"] == 0.0
        assert aggregate["p50_latency_ms"] == 0.0
        assert aggregate["p95_latency_ms"] == 0.0

    def test_mean_metrics_and_latency_percentiles(self):
        rows = [
            _row(
                query="q1",
                tags=["cjk"],
                recall=1.0,
                mrr=1.0,
                ndcg=1.0,
                p5=0.2,
                ap=1.0,
                latency=100.0,
            ),
            _row(
                query="q2",
                tags=["english"],
                recall=0.5,
                mrr=0.5,
                ndcg=0.75,
                p5=0.4,
                ap=0.5,
                latency=200.0,
            ),
            _row(
                query="q3",
                tags=["english"],
                recall=0.0,
                mrr=0.0,
                ndcg=0.0,
                p5=0.0,
                ap=0.0,
                latency=300.0,
            ),
        ]

        aggregate = aggregate_rows(rows)

        assert aggregate["n_queries"] == 3
        assert aggregate["mean_recall_at_10"] == pytest.approx(0.5)
        assert aggregate["mean_mrr"] == pytest.approx(0.5)
        assert aggregate["mean_ndcg_at_10"] == pytest.approx(0.5833333333)
        assert aggregate["mean_precision_at_5"] == pytest.approx(0.2)
        assert aggregate["mean_average_precision_at_10"] == pytest.approx(0.5)
        assert aggregate["mean_latency_ms"] == pytest.approx(200.0)
        assert aggregate["p50_latency_ms"] == pytest.approx(200.0)
        assert aggregate["p95_latency_ms"] == pytest.approx(300.0)


class TestAggregateByTag:
    def test_groups_rows_by_each_tag(self):
        rows = [
            _row(
                query="q1",
                tags=["cjk", "short"],
                recall=1.0,
                mrr=1.0,
                ndcg=1.0,
                p5=0.2,
                ap=1.0,
                latency=100.0,
            ),
            _row(
                query="q2",
                tags=["english", "short"],
                recall=0.0,
                mrr=0.0,
                ndcg=0.0,
                p5=0.0,
                ap=0.0,
                latency=300.0,
            ),
        ]

        by_tag = aggregate_by_tag(rows)

        assert list(by_tag) == ["cjk", "english", "short"]
        assert by_tag["cjk"]["n_queries"] == 1
        assert by_tag["english"]["n_queries"] == 1
        assert by_tag["short"]["n_queries"] == 2
        assert by_tag["short"]["mean_recall_at_10"] == pytest.approx(0.5)
        assert by_tag["short"]["p95_latency_ms"] == pytest.approx(300.0)

    def test_rows_without_tags_are_omitted(self):
        rows = [
            _row(
                query="q1",
                tags=[],
                recall=1.0,
                mrr=1.0,
                ndcg=1.0,
                p5=0.2,
                ap=1.0,
                latency=100.0,
            )
        ]

        assert aggregate_by_tag(rows) == {}

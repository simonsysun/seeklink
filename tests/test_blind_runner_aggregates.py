"""Tests for blind-test runner aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.blind import run as blind_run
from tests.blind.run import (
    ResultRow,
    RunnerState,
    aggregate_by_tag,
    aggregate_rows,
    resolved_rerank_k_counts,
)


class TestLoadQueries:
    def test_load_queries_defaults_expected_paths_to_grade_three(self, tmp_path: Path):
        path = tmp_path / "queries.yaml"
        path.write_text(
            """
- query: memory
  expected_paths:
    - notes/answer.md
  tags: [english]
""",
            encoding="utf-8",
        )

        specs = blind_run.load_queries(path)

        assert specs[0].relevance == {"notes/answer.md": 3.0}

    def test_load_queries_accepts_graded_relevance_labels(self, tmp_path: Path):
        path = tmp_path / "queries.yaml"
        path.write_text(
            """
- query: transformer retrieval
  expected_paths:
    - notes/attention.md
  relevance:
    notes/attention.md: 3
    notes/agent-memory.md: 2
    notes/rrf.md: 1
  tags: [english]
""",
            encoding="utf-8",
        )

        specs = blind_run.load_queries(path)

        assert specs[0].relevance == {
            "notes/attention.md": 3.0,
            "notes/agent-memory.md": 2.0,
            "notes/rrf.md": 1.0,
        }

    def test_result_row_uses_graded_relevance_for_ndcg(self):
        spec = blind_run.QuerySpec(
            query="transformer retrieval",
            intent=None,
            expected_paths=["notes/answer.md"],
            relevance={
                "notes/answer.md": 3.0,
                "notes/support.md": 2.0,
            },
            tags=["english"],
            expansion=None,
        )

        row = blind_run._result_row(
            spec=spec,
            config="A",
            hits=["notes/support.md", "notes/answer.md"],
            titles=[None, None],
            snippets=["", ""],
            scores=[1.0, 0.5],
            latency_ms=10.0,
            reranker_active=True,
            rerank_k=5,
        )

        assert row.mrr == pytest.approx(0.5)
        assert row.ndcg_at_10 > row.mrr


class TestFirstStagePayload:
    def test_records_expected_and_hit_channel_ranks(self):
        diagnostics = blind_run.SearchDiagnostics()
        diagnostics.candidate_count = 3
        diagnostics.bm25_ranks = {1: 2}
        diagnostics.vector_ranks = {1: 5, 2: 1}
        diagnostics.title_ranks = {2: 1}
        diagnostics.indegree_ranks = {1: 1, 2: 2, 3: 3}
        diagnostics.first_stage_ranked_source_ids = [2, 1, 3]
        diagnostics.rerank_candidate_source_ids = [2, 1]

        class Source:
            def __init__(self, source_id: int):
                self.id = source_id

        class FakeDb:
            def get_source_by_path(self, path: str):
                return {
                    "answer.md": Source(1),
                    "missing.md": None,
                }[path]

        payload = blind_run._first_stage_payload(
            db=FakeDb(),
            diagnostics=diagnostics,
            expected_paths=["answer.md", "missing.md"],
            results=[
                blind_run.SearchResult(
                    source_id=2,
                    chunk_id=20,
                    path="hit.md",
                    title="Hit",
                    content="",
                    score=1.0,
                    indegree=0,
                )
            ],
        )

        assert payload["candidate_count"] == 3
        assert payload["channel_source_counts"] == {
            "bm25": 1,
            "vector": 2,
            "title": 1,
            "indegree": 3,
        }
        assert payload["expected_path_ranks"] == {
            "answer.md": {
                "bm25": 2,
                "vector": 5,
                "title": None,
                "indegree": 1,
                "rrf": 2,
                "rerank_candidate": 2,
            },
            "missing.md": None,
        }
        assert payload["hit_channel_ranks"] == [
            {
                "path": "hit.md",
                "bm25": None,
                "vector": 1,
                "title": 1,
                "indegree": 2,
                "rrf": 1,
                "rerank_candidate": 1,
            }
        ]


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
        relevance={},
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

    def test_resolved_rerank_k_counts(self):
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
            ),
            _row(
                query="q2",
                tags=[],
                recall=1.0,
                mrr=1.0,
                ndcg=1.0,
                p5=0.2,
                ap=1.0,
                latency=100.0,
            ),
            _row(
                query="q3",
                tags=[],
                recall=1.0,
                mrr=1.0,
                ndcg=1.0,
                p5=0.2,
                ap=1.0,
                latency=100.0,
            ),
        ]
        rows[0].resolved_rerank_k = 5
        rows[1].resolved_rerank_k = 20
        rows[2].resolved_rerank_k = 5

        assert resolved_rerank_k_counts(rows) == {"5": 2, "20": 1}


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


class TestRerankOptions:
    def test_parser_supports_rerank_k_and_no_rerank_alias(self):
        parser = blind_run.build_parser()

        args = parser.parse_args(
            [
                "--config",
                "A",
                "--queries",
                "queries.yaml",
                "--vault",
                "vault",
                "--out",
                "out.json",
                "--rerank-k",
                "7",
                "--no-rerank",
            ]
        )

        assert args.rerank_k == 7
        assert args.no_reranker is True

    def test_parser_supports_auto_rerank_k(self):
        parser = blind_run.build_parser()

        args = parser.parse_args(
            [
                "--config",
                "A",
                "--queries",
                "queries.yaml",
                "--vault",
                "vault",
                "--out",
                "out.json",
                "--rerank-k",
                "auto",
            ]
        )

        assert args.rerank_k == "auto"

    def test_legacy_no_reranker_alias_still_works(self):
        parser = blind_run.build_parser()

        args = parser.parse_args(
            [
                "--config",
                "A",
                "--queries",
                "queries.yaml",
                "--vault",
                "vault",
                "--out",
                "out.json",
                "--no-reranker",
            ]
        )

        assert args.no_reranker is True

    def test_search_with_state_passes_rerank_k(self, monkeypatch):
        captured: dict = {}

        class FakeReranker:
            disabled = False

        def fake_search(db, embedder, query, **kwargs):
            captured["db"] = db
            captured["embedder"] = embedder
            captured["query"] = query
            captured["top_k"] = kwargs["top_k"]
            captured["reranker"] = kwargs["reranker"]
            captured["rerank_k"] = kwargs["rerank_k"]
            return []

        fake_db = object()
        fake_embedder = object()
        fake_reranker = FakeReranker()
        state = RunnerState(
            db=fake_db,
            embedder=fake_embedder,
            reranker=fake_reranker,  # type: ignore[arg-type]
            rerank_k=7,
            vault=Path("/tmp/vault"),
        )
        monkeypatch.setattr(blind_run, "search", fake_search)

        assert blind_run._search_with_state(state, "memory") == []
        assert captured == {
            "db": fake_db,
            "embedder": fake_embedder,
            "query": "memory",
            "top_k": 10,
            "reranker": fake_reranker,
            "rerank_k": 7,
        }

    def test_search_with_state_passes_auto_rerank_k(self, monkeypatch):
        captured: dict = {}

        class FakeReranker:
            disabled = False

        def fake_search(db, embedder, query, **kwargs):
            captured["rerank_k"] = kwargs["rerank_k"]
            return []

        state = RunnerState(
            db=object(),
            embedder=object(),
            reranker=FakeReranker(),  # type: ignore[arg-type]
            rerank_k="auto",
            vault=Path("/tmp/vault"),
        )
        monkeypatch.setattr(blind_run, "search", fake_search)

        assert blind_run._search_with_state(state, "memory") == []
        assert captured == {"rerank_k": "auto"}

    def test_search_with_state_passes_diagnostics(self, monkeypatch):
        captured: dict = {}

        class FakeReranker:
            disabled = False

        def fake_search(db, embedder, query, **kwargs):
            captured["diagnostics"] = kwargs["diagnostics"]
            return []

        diagnostics = blind_run.SearchDiagnostics()
        state = RunnerState(
            db=object(),
            embedder=object(),
            reranker=FakeReranker(),  # type: ignore[arg-type]
            rerank_k="auto",
            vault=Path("/tmp/vault"),
        )
        monkeypatch.setattr(blind_run, "search", fake_search)

        assert blind_run._search_with_state(
            state,
            "memory",
            diagnostics=diagnostics,
        ) == []
        assert captured == {"diagnostics": diagnostics}

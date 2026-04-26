"""Blind test runner for query expansion evaluation.

Usage:
    python tests/blind/run.py --config A --queries tests/blind/queries.yaml \\
        --vault /path/to/vault --out tests/blind/results/A.json

Three configs:
    A: baseline = current seeklink daemon path (search + reranker)
    B: Qwen3-0.6B expansion (requires seeklink v0.4+; raises until implemented)
    C: hand-crafted expansion (RRF-fused; upper bound, not simple max)

See docs/blind-test.md for the framework spec and acceptance criteria.

Requires: PyYAML (dev dependency). Install with
    uv pip install --group dev pyyaml
or add to pyproject.toml [dependency-groups.dev] before running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seeklink.app import init_app
from seeklink.reranker import Reranker
from seeklink.search import RerankK, SearchDiagnostics, SearchResult, search

try:
    from .metrics import (
        average_precision_at_k,
        last_expected_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
    )
except ImportError:  # pragma: no cover - supports `python tests/blind/run.py`
    from metrics import (
        average_precision_at_k,
        last_expected_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
    )


@dataclass(frozen=True, slots=True)
class QuerySpec:
    query: str
    intent: str | None
    expected_paths: list[str]
    relevance: dict[str, float]
    tags: list[str]
    expansion: list[str] | None


def _parse_relevance(raw: object, expected_paths: list[str]) -> dict[str, float]:
    relevance = {path: 3.0 for path in expected_paths}
    if raw is None:
        return relevance
    if not isinstance(raw, dict):
        raise ValueError("relevance must be a mapping of path -> grade")

    for path, grade in raw.items():
        if not isinstance(path, str) or not path:
            raise ValueError("relevance paths must be non-empty strings")
        try:
            numeric_grade = float(grade)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"relevance grade for {path!r} must be numeric"
            ) from e
        if numeric_grade < 0:
            raise ValueError(f"relevance grade for {path!r} must be >= 0")
        relevance[path] = numeric_grade
    return relevance


def _parse_rerank_k(raw: str) -> RerankK:
    if raw == "auto":
        return raw
    try:
        value = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--rerank-k must be a positive integer or 'auto'"
        ) from e
    if value < 1:
        raise argparse.ArgumentTypeError("--rerank-k must be >= 1")
    return value


@dataclass
class ResultRow:
    """Everything the framework's acceptance criteria need at evaluation time.

    Paths are needed for Recall@K / MRR. Snippets are needed for human blind
    scoring (you read them without knowing the config). Latency is per-query
    wall clock of the full search call chain, measured at runner level —
    model load time is excluded by initializing once in main().
    """
    query: str
    config: str
    tags: list[str]
    hits: list[str]
    titles: list[str | None]
    snippets: list[str]
    scores: list[float]
    relevance: dict[str, float]
    latency_ms: float
    reranker_active: bool
    recall_at_10: float
    mrr: float
    precision_at_5: float
    average_precision_at_10: float
    ndcg_at_10: float
    last_expected_rank: int | None
    rerank_k: int | str = 0
    resolved_rerank_k: int | None = None
    rerank_k_reason: str | None = None
    expansions_used: list[str] = field(default_factory=list)


def load_queries(path: Path) -> list[QuerySpec]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    specs: list[QuerySpec] = []
    for i, r in enumerate(raw):
        if "query" not in r or "expected_paths" not in r:
            raise ValueError(
                f"queries.yaml entry {i}: missing required field "
                f"('query' and 'expected_paths' are mandatory)"
            )
        expected_paths = list(r["expected_paths"])
        specs.append(
            QuerySpec(
                query=r["query"],
                intent=r.get("intent"),
                expected_paths=expected_paths,
                relevance=_parse_relevance(r.get("relevance"), expected_paths),
                tags=list(r.get("tags", [])),
                expansion=list(r["expansion"]) if r.get("expansion") else None,
            )
        )
    return specs


def _extract(results: list[SearchResult]) -> tuple[
    list[str], list[str | None], list[str], list[float]
]:
    return (
        [r.path for r in results],
        [r.title for r in results],
        [(r.content or "")[:200] for r in results],
        [r.score for r in results],
    )


def _result_row(
    *,
    spec: QuerySpec,
    config: str,
    hits: list[str],
    titles: list[str | None],
    snippets: list[str],
    scores: list[float],
    latency_ms: float,
    reranker_active: bool,
    rerank_k: int | str,
    resolved_rerank_k: int | None = None,
    rerank_k_reason: str | None = None,
    expansions_used: list[str] | None = None,
) -> ResultRow:
    """Build a ResultRow and compute all per-query metrics in one place."""
    return ResultRow(
        query=spec.query,
        config=config,
        tags=list(spec.tags),
        hits=hits,
        titles=titles,
        snippets=snippets,
        scores=scores,
        relevance=dict(spec.relevance),
        latency_ms=latency_ms,
        reranker_active=reranker_active,
        rerank_k=rerank_k if reranker_active else 0,
        recall_at_10=recall_at_k(hits, spec.expected_paths),
        mrr=reciprocal_rank(hits, spec.expected_paths),
        precision_at_5=precision_at_k(hits, spec.expected_paths, k=5),
        average_precision_at_10=average_precision_at_k(
            hits, spec.expected_paths, k=10
        ),
        ndcg_at_10=ndcg_at_k(
            hits, spec.expected_paths, k=10, relevance=spec.relevance
        ),
        last_expected_rank=last_expected_rank(hits, spec.expected_paths, k=10),
        resolved_rerank_k=resolved_rerank_k if reranker_active else 0,
        rerank_k_reason=rerank_k_reason if reranker_active else None,
        expansions_used=list(expansions_used or []),
    )


def _rrf_fuse_paths(
    per_query_results: list[list[SearchResult]],
    k_rrf: int = 60,
) -> list[SearchResult]:
    """Fuse multiple SearchResult lists (from different queries) by rank.

    Used by Config C to aggregate results from the original query + each
    hand-crafted expansion. Rank-fusion, not score-fusion, because scores
    from different query runs are not calibrated to each other.

    Returns up to 10 results, ordered by RRF score descending, with the
    SearchResult taken from whichever run first surfaced that path (so
    snippets/titles come from a real retrieval, not synthesized).
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for results in per_query_results:
        for rank0, r in enumerate(results):
            scores[r.path] = scores.get(r.path, 0.0) + 1.0 / (k_rrf + rank0 + 1)
            first_seen.setdefault(r.path, r)
    ranked_paths = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)[:10]
    return [
        # Replace score with the fused RRF score so callers can see the
        # merged ranking signal, but keep every other field intact.
        SearchResult(
            source_id=first_seen[p].source_id,
            chunk_id=first_seen[p].chunk_id,
            path=p,
            title=first_seen[p].title,
            content=first_seen[p].content,
            score=scores[p],
            indegree=first_seen[p].indegree,
        )
        for p in ranked_paths
    ]


@dataclass
class RunnerState:
    """Initialized once per invocation. Carries the shared db/embedder/reranker.

    Running `init_app` per query would make every latency measurement include
    model load time (1-2s on cold cache), which is not what the user
    experiences in the daemon path where the daemon holds these in memory.
    """
    db: object
    embedder: object
    reranker: Reranker | None
    rerank_k: RerankK
    vault: Path

    @property
    def reranker_active(self) -> bool:
        return self.reranker is not None and not self.reranker.disabled

    @property
    def active_rerank_k(self) -> int | str:
        return self.rerank_k if self.reranker_active else 0

    def close(self) -> None:
        try:
            self.db.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def init_state(vault: Path, with_reranker: bool, rerank_k: RerankK) -> RunnerState:
    db, embedder, resolved_vault = init_app(vault)
    # Trigger lazy load up-front so the first measured query excludes
    # embedder model/cache startup, matching the resident daemon path.
    try:
        embedder.embed_query("warmup")
    except Exception:
        pass
    # Trigger jieba/FTS tokenizer setup before the measured query loop.
    try:
        db.search_fts("warmup", limit=1)
        db.search_fts_sources("warmup", limit=1)
    except Exception:
        pass

    reranker: Reranker | None = None
    if with_reranker:
        reranker = Reranker()
        # Trigger lazy load up-front so the FIRST real query's latency is
        # not artificially inflated by model warmup.
        try:
            reranker.rerank("warmup", ["warmup passage"])
        except Exception:
            pass
    return RunnerState(
        db=db,
        embedder=embedder,
        reranker=reranker,
        rerank_k=rerank_k,
        vault=resolved_vault,
    )


def _search_with_state(
    state: RunnerState,
    query: str,
    diagnostics: SearchDiagnostics | None = None,
) -> list[SearchResult]:
    return search(
        state.db,  # type: ignore[arg-type]
        state.embedder,  # type: ignore[arg-type]
        query,
        top_k=10,
        reranker=state.reranker,
        rerank_k=state.rerank_k,
        diagnostics=diagnostics,
    )


def run_config_a(spec: QuerySpec, state: RunnerState) -> ResultRow:
    """Baseline: product behavior = search + reranker (matches daemon path)."""
    diagnostics = SearchDiagnostics()
    t0 = time.perf_counter()
    results = _search_with_state(state, spec.query, diagnostics=diagnostics)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    hits, titles, snippets, scores = _extract(results)
    return _result_row(
        spec=spec,
        config="A",
        hits=hits,
        titles=titles,
        snippets=snippets,
        scores=scores,
        latency_ms=latency_ms,
        reranker_active=state.reranker_active,
        rerank_k=state.active_rerank_k,
        resolved_rerank_k=diagnostics.resolved_rerank_k,
        rerank_k_reason=diagnostics.rerank_k_reason,
    )


def run_config_b(spec: QuerySpec, state: RunnerState) -> ResultRow:
    """v0.4 candidate: Qwen3-0.6B expansion + seeklink. Not yet shipped.

    Wiring sketch for when v0.4 arrives:

        from seeklink.expansion import expand_query     # future module
        expansions = expand_query(spec.query, intent=spec.intent)
        per_q = [_search_with_state(state, q) for q in [spec.query, *expansions]]
        fused = _rrf_fuse_paths(per_q)
        # ... wrap into ResultRow, record `expansions_used=expansions` ...
    """
    raise NotImplementedError(
        "Config B requires seeklink v0.4 query expansion (not yet shipped). "
        "Re-run the blind test after v0.4 lands."
    )


def run_config_c(spec: QuerySpec, state: RunnerState) -> ResultRow:
    """Upper bound: run each hand-crafted expansion, fuse via RRF (rank-based).

    Rank-fusion (not score-fusion) because scores across different query
    runs are not calibrated to each other. `_rrf_fuse_paths` implements a
    standard RRF over the lists.
    """
    if not spec.expansion:
        raise ValueError(
            f"query={spec.query!r}: 'expansion:' field is empty. "
            "Config C requires hand-crafted alternates in queries.yaml."
        )
    all_queries = [spec.query, *spec.expansion]
    t0 = time.perf_counter()
    per_q = [_search_with_state(state, q) for q in all_queries]
    fused = _rrf_fuse_paths(per_q)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    hits, titles, snippets, scores = _extract(fused)
    return _result_row(
        spec=spec,
        config="C",
        hits=hits,
        titles=titles,
        snippets=snippets,
        scores=scores,
        latency_ms=latency_ms,
        reranker_active=state.reranker_active,
        rerank_k=state.active_rerank_k,
        expansions_used=list(spec.expansion),
    )


CONFIG_RUNNERS = {"A": run_config_a, "B": run_config_b, "C": run_config_c}


_AGGREGATE_FIELDS = (
    "recall_at_10",
    "mrr",
    "precision_at_5",
    "average_precision_at_10",
    "ndcg_at_10",
    "latency_ms",
)


def _percentile(values: list[float], quantile: float) -> float:
    """Return a sorted-pick percentile suitable for small blind-test runs."""
    if not values:
        return 0.0
    if quantile <= 0:
        return sorted(values)[0]
    if quantile >= 1:
        return sorted(values)[-1]
    ordered = sorted(values)
    index = int(quantile * len(ordered))
    index = min(index, len(ordered) - 1)
    return ordered[index]


def aggregate_rows(rows: list[ResultRow]) -> dict[str, float | int]:
    """Aggregate metrics across result rows."""
    aggregate: dict[str, float | int] = {"n_queries": len(rows)}
    for field_name in _AGGREGATE_FIELDS:
        values = [float(getattr(row, field_name)) for row in rows]
        aggregate[f"mean_{field_name}"] = sum(values) / len(values) if values else 0.0

    latencies = [row.latency_ms for row in rows]
    aggregate["p50_latency_ms"] = _percentile(latencies, 0.50)
    aggregate["p95_latency_ms"] = _percentile(latencies, 0.95)
    return aggregate


def resolved_rerank_k_counts(rows: list[ResultRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.resolved_rerank_k is None:
            continue
        key = str(row.resolved_rerank_k)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def aggregate_by_tag(rows: list[ResultRow]) -> dict[str, dict[str, float | int]]:
    """Aggregate metrics for each QuerySpec tag.

    A query can belong to multiple tags, so per-tag counts intentionally do
    not sum to total query count.
    """
    grouped: dict[str, list[ResultRow]] = {}
    for row in rows:
        for tag in row.tags:
            grouped.setdefault(tag, []).append(row)
    return {
        tag: aggregate_rows(tag_rows)
        for tag, tag_rows in sorted(grouped.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["A", "B", "C"], required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--rerank-k",
        type=_parse_rerank_k,
        default=20,
        help=(
            "Number of first-stage candidates passed to the reranker "
            "or 'auto' for query-sensitive routing (default: 20). "
            "Use with config A/C latency sweeps."
        ),
    )
    parser.add_argument(
        "--no-rerank",
        "--no-reranker",
        dest="no_reranker",
        action="store_true",
        help="Run without reranker. Useful for isolating blending effects. "
             "Do NOT use for the official baseline — product path has reranker on.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve()
    specs = load_queries(args.queries)
    runner = CONFIG_RUNNERS[args.config]

    state = init_state(
        vault,
        with_reranker=not args.no_reranker,
        rerank_k=args.rerank_k,
    )
    try:
        records: list[ResultRow] = []
        rerank_label = (
            f"rerank_k={state.active_rerank_k}"
            if state.reranker_active
            else "rerank=off"
        )
        for spec in specs:
            try:
                row = runner(spec, state)
            except NotImplementedError as e:
                print(f"ABORT: {e}", file=sys.stderr)
                sys.exit(2)
            except Exception as e:
                print(f"FAIL query={spec.query!r}: {e}", file=sys.stderr)
                continue
            records.append(row)
            print(
                f"{row.config} {rerank_label} {spec.query!r:<40} "
                f"recall@10={row.recall_at_10:.2f} "
                f"mrr={row.mrr:.2f} ndcg@10={row.ndcg_at_10:.2f} "
                f"lat={row.latency_ms:.0f}ms "
                f"resolved_k={row.resolved_rerank_k}",
                file=sys.stderr,
            )

        aggregate = aggregate_rows(records)
        by_tag = aggregate_by_tag(records)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": args.config,
                    "queries_file": str(args.queries),
                    "vault": str(vault),
                    "n_queries": len(records),
                    "with_reranker": state.reranker_active,
                    "reranking": {
                        "enabled": state.reranker_active,
                        "rerank_k": state.active_rerank_k,
                        "resolved_k_counts": resolved_rerank_k_counts(records),
                    },
                    "aggregate": aggregate,
                    "by_tag": by_tag,
                    "results": [asdict(r) for r in records],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(
            f"\nconfig={args.config} n={len(records)} "
            f"mean_recall@10={aggregate['mean_recall_at_10']:.3f} "
            f"mean_mrr={aggregate['mean_mrr']:.3f} "
            f"mean_ndcg@10={aggregate['mean_ndcg_at_10']:.3f} "
            f"mean_lat={aggregate['mean_latency_ms']:.0f}ms "
            f"p50_lat={aggregate['p50_latency_ms']:.0f}ms "
            f"p95_lat={aggregate['p95_latency_ms']:.0f}ms "
            f"→ {args.out}",
            file=sys.stderr,
        )
    finally:
        state.close()


if __name__ == "__main__":
    main()

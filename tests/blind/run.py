"""Blind test runner for query expansion evaluation (v2, post-review).

Usage:
    python tests/blind/run.py --config A --queries tests/blind/queries.yaml \\
        --vault ~/Rhizome --out tests/blind/results/A.json

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

from seeklink.app import init_app
from seeklink.reranker import Reranker
from seeklink.search import SearchResult, search


@dataclass(frozen=True, slots=True)
class QuerySpec:
    query: str
    intent: str | None
    expected_paths: list[str]
    tags: list[str]
    expansion: list[str] | None


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
    hits: list[str]
    titles: list[str | None]
    snippets: list[str]
    scores: list[float]
    latency_ms: float
    reranker_active: bool
    recall_at_10: float
    mrr: float
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
        specs.append(
            QuerySpec(
                query=r["query"],
                intent=r.get("intent"),
                expected_paths=list(r["expected_paths"]),
                tags=list(r.get("tags", [])),
                expansion=list(r["expansion"]) if r.get("expansion") else None,
            )
        )
    return specs


def recall_at_k(hits: list[str], expected: list[str], k: int = 10) -> float:
    if not expected:
        return 0.0
    top = set(hits[:k])
    return len(top & set(expected)) / len(expected)


def mrr(hits: list[str], expected: list[str], k: int = 10) -> float:
    expected_set = set(expected)
    for i, path in enumerate(hits[:k], start=1):
        if path in expected_set:
            return 1.0 / i
    return 0.0


def _extract(results: list[SearchResult]) -> tuple[
    list[str], list[str | None], list[str], list[float]
]:
    return (
        [r.path for r in results],
        [r.title for r in results],
        [(r.content or "")[:200] for r in results],
        [r.score for r in results],
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
    vault: Path

    def close(self) -> None:
        try:
            self.db.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def init_state(vault: Path, with_reranker: bool) -> RunnerState:
    db, embedder, resolved_vault = init_app(vault)
    reranker: Reranker | None = None
    if with_reranker:
        reranker = Reranker()
        # Trigger lazy load up-front so the FIRST real query's latency is
        # not artificially inflated by model warmup.
        try:
            reranker.rerank("warmup", ["warmup passage"])
        except Exception:
            pass
    return RunnerState(db=db, embedder=embedder, reranker=reranker, vault=resolved_vault)


def _search_with_state(state: RunnerState, query: str) -> list[SearchResult]:
    return search(
        state.db,  # type: ignore[arg-type]
        state.embedder,  # type: ignore[arg-type]
        query,
        top_k=10,
        reranker=state.reranker,
    )


def run_config_a(spec: QuerySpec, state: RunnerState) -> ResultRow:
    """Baseline: product behavior = search + reranker (matches daemon path)."""
    t0 = time.perf_counter()
    results = _search_with_state(state, spec.query)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    hits, titles, snippets, scores = _extract(results)
    return ResultRow(
        query=spec.query,
        config="A",
        hits=hits,
        titles=titles,
        snippets=snippets,
        scores=scores,
        latency_ms=latency_ms,
        reranker_active=state.reranker is not None and not state.reranker.disabled,
        recall_at_10=recall_at_k(hits, spec.expected_paths),
        mrr=mrr(hits, spec.expected_paths),
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
    return ResultRow(
        query=spec.query,
        config="C",
        hits=hits,
        titles=titles,
        snippets=snippets,
        scores=scores,
        latency_ms=latency_ms,
        reranker_active=state.reranker is not None and not state.reranker.disabled,
        recall_at_10=recall_at_k(hits, spec.expected_paths),
        mrr=mrr(hits, spec.expected_paths),
        expansions_used=list(spec.expansion),
    )


CONFIG_RUNNERS = {"A": run_config_a, "B": run_config_b, "C": run_config_c}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["A", "B", "C"], required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Run without reranker. Useful for isolating blending effects. "
             "Do NOT use for the official baseline — product path has reranker on.",
    )
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve()
    specs = load_queries(args.queries)
    runner = CONFIG_RUNNERS[args.config]

    state = init_state(vault, with_reranker=not args.no_reranker)
    try:
        records: list[ResultRow] = []
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
                f"{row.config} {spec.query!r:<40} "
                f"recall@10={row.recall_at_10:.2f} mrr={row.mrr:.2f} "
                f"lat={row.latency_ms:.0f}ms",
                file=sys.stderr,
            )

        def _agg(field_name: str) -> float:
            vals = [getattr(r, field_name) for r in records]
            return sum(vals) / len(vals) if vals else 0.0

        # Simple p95 (sorted-pick) — good enough for 20-30 queries.
        latencies = sorted(r.latency_ms for r in records)
        p95 = latencies[int(0.95 * len(latencies))] if latencies else 0.0

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": args.config,
                    "queries_file": str(args.queries),
                    "vault": str(vault),
                    "n_queries": len(records),
                    "with_reranker": state.reranker is not None and not state.reranker.disabled,
                    "aggregate": {
                        "mean_recall_at_10": _agg("recall_at_10"),
                        "mean_mrr": _agg("mrr"),
                        "mean_latency_ms": _agg("latency_ms"),
                        "p95_latency_ms": p95,
                    },
                    "results": [asdict(r) for r in records],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(
            f"\nconfig={args.config} n={len(records)} "
            f"mean_recall@10={_agg('recall_at_10'):.3f} "
            f"mean_mrr={_agg('mrr'):.3f} "
            f"mean_lat={_agg('latency_ms'):.0f}ms "
            f"p95_lat={p95:.0f}ms "
            f"→ {args.out}",
            file=sys.stderr,
        )
    finally:
        state.close()


if __name__ == "__main__":
    main()

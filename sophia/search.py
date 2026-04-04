"""Search engine — three-channel RRF fusion with optional graph expansion."""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass

from sophia.db import Database
from sophia.embedder import Embedder
from sophia.models import Chunk, Source

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    source_id: int
    chunk_id: int
    path: str
    title: str | None
    content: str
    score: float
    indegree: int


def search(
    db: Database,
    embedder: Embedder,
    query: str,
    *,
    top_k: int = 10,
    expand: bool = False,
    bm25_weight: float = 1.0,
    vec_weight: float = 1.0,
    indegree_weight: float = 0.3,
    path_prefix: str | None = None,
) -> list[SearchResult]:
    """Search the knowledge base using three-channel RRF fusion.

    Channels: BM25 (keyword), vector (semantic), indegree (quality prior).
    Optional graph expansion follows wiki-links from top results.

    If path_prefix is given, only return results whose source path starts
    with the prefix (e.g. "agent_memory/decisions/" for namespace filtering).
    """
    if not query.strip() or top_k <= 0:
        return []

    # Channel 1: BM25
    bm25_results = _safe_fts(db, query, limit=50)
    bm25_best = _best_chunk_per_source(bm25_results)
    # Rank by BM25 score (FTS5 rank is negative; more negative = better)
    bm25_ranked = sorted(bm25_best.keys(), key=lambda sid: bm25_best[sid][1])
    bm25_ranks = {sid: i + 1 for i, sid in enumerate(bm25_ranked)}

    # Channel 2: Vector (use larger k when expansion is requested)
    vec_limit = 200 if expand else 50
    query_emb = _safe_embed(embedder, query)
    if query_emb is not None:
        vec_results = db.search_vec(query_emb, k=vec_limit)
    else:
        vec_results = []

    # Batch-fetch chunks to get source_id, filtering invalid distances
    chunk_ids = [cid for cid, dist in vec_results if isinstance(dist, (int, float)) and math.isfinite(dist)]
    valid_distances = {cid: dist for cid, dist in vec_results if isinstance(dist, (int, float)) and math.isfinite(dist)}
    chunks_by_id = db.get_chunks_by_ids(chunk_ids)

    vec_chunks_with_dist: list[tuple[Chunk, float]] = [
        (chunks_by_id[cid], valid_distances[cid])
        for cid in chunk_ids
        if cid in chunks_by_id
    ]
    vec_best = _best_chunk_per_source(vec_chunks_with_dist)
    # Rank by distance (ascending = better)
    vec_ranked = sorted(vec_best.keys(), key=lambda sid: vec_best[sid][1])
    vec_ranks = {sid: i + 1 for i, sid in enumerate(vec_ranked)}

    # Candidate pool: union of BM25 and vector source IDs
    candidate_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())
    if not candidate_ids:
        return []

    # Batch-fetch sources for indegree (and path_prefix filtering)
    sources = db.get_sources_by_ids(list(candidate_ids))

    # Namespace filtering: restrict to sources matching path_prefix
    if path_prefix:
        filtered = {sid for sid in candidate_ids if sid in sources and sources[sid].path.startswith(path_prefix)}
        if not filtered:
            return []
        candidate_ids = filtered
        bm25_ranks = {k: v for k, v in bm25_ranks.items() if k in filtered}
        vec_ranks = {k: v for k, v in vec_ranks.items() if k in filtered}

    # Channel 3: Indegree (rank candidates by indegree descending)
    indeg_ranked = sorted(
        candidate_ids,
        key=lambda sid: sources[sid].indegree if sid in sources else 0,
        reverse=True,
    )
    indeg_ranks = {sid: i + 1 for i, sid in enumerate(indeg_ranked)}

    # RRF fusion
    scores = _rrf_fuse(
        bm25_ranks, vec_ranks, indeg_ranks,
        weights=(bm25_weight, vec_weight, indegree_weight),
    )

    # Sort by score descending, take top_k
    ranked = sorted(scores.keys(), key=lambda sid: scores[sid], reverse=True)
    ranked = ranked[:top_k]

    # Pick best chunk for each source (prefer BM25 chunk, fall back to vec)
    best_chunks: dict[int, Chunk] = {}
    for sid in ranked:
        if sid in bm25_best:
            best_chunks[sid] = bm25_best[sid][0]
        elif sid in vec_best:
            best_chunks[sid] = vec_best[sid][0]

    # Build initial results
    results = [
        SearchResult(
            source_id=sid,
            chunk_id=best_chunks[sid].id,
            path=sources[sid].path,
            title=sources[sid].title,
            content=best_chunks[sid].content,
            score=scores[sid],
            indegree=sources[sid].indegree,
        )
        for sid in ranked
        if sid in best_chunks and sid in sources
    ]

    # Graph expansion (reuses vec_results from the larger k=200 search)
    if expand and results and query_emb is not None:
        expansion_source_ids = [r.source_id for r in results[:3]]
        existing_ids = {r.source_id for r in results}
        neighbor_ids = _get_neighbor_source_ids(db, expansion_source_ids)
        new_neighbors = neighbor_ids - existing_ids

        if new_neighbors:
            expanded = _score_expansion_candidates(
                db, vec_results, new_neighbors, sources, vec_weight,
            )
            results.extend(expanded)
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[:top_k]

    return results


def _safe_fts(db: Database, query: str, limit: int) -> list[tuple[Chunk, float]]:
    """Run FTS search, returning empty list on query syntax errors."""
    try:
        return db.search_fts(query, limit=limit)
    except sqlite3.OperationalError:
        return []
    except Exception:
        logger.warning("Unexpected FTS error for query %r", query, exc_info=True)
        return []


def _safe_embed(embedder: Embedder, query: str) -> bytes | None:
    """Embed a query, returning None on failure (falls back to BM25-only)."""
    try:
        return embedder.embed_query(query)
    except Exception:
        logger.warning("Embedding failed for query %r", query, exc_info=True)
        return None


def _best_chunk_per_source(
    chunks_with_scores: list[tuple[Chunk, float]],
) -> dict[int, tuple[Chunk, float]]:
    """Group chunks by source_id, keep best (lowest score) per source."""
    best: dict[int, tuple[Chunk, float]] = {}
    for chunk, score in chunks_with_scores:
        if chunk.source_id not in best or score < best[chunk.source_id][1]:
            best[chunk.source_id] = (chunk, score)
    return best


def _rrf_fuse(
    bm25_ranks: dict[int, int],
    vec_ranks: dict[int, int],
    indeg_ranks: dict[int, int],
    weights: tuple[float, float, float],
    k: int = 60,
) -> dict[int, float]:
    """Compute RRF scores for candidates across three channels."""
    bm25_w, vec_w, indeg_w = weights
    all_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())
    scores: dict[int, float] = {}
    for sid in all_ids:
        score = 0.0
        if sid in bm25_ranks:
            score += bm25_w / (k + bm25_ranks[sid])
        if sid in vec_ranks:
            score += vec_w / (k + vec_ranks[sid])
        if sid in indeg_ranks:
            score += indeg_w / (k + indeg_ranks[sid])
        scores[sid] = score
    return scores


def _get_neighbor_source_ids(db: Database, source_ids: list[int]) -> set[int]:
    """Collect 1-hop neighbor source IDs via wiki-links (outgoing + incoming)."""
    neighbors: set[int] = set()
    for sid in source_ids:
        for link in db.get_links_from(sid):
            if link.target_note_id is not None:
                neighbors.add(link.target_note_id)
        for link in db.get_links_to(sid):
            neighbors.add(link.source_note_id)
    return neighbors


def _score_expansion_candidates(
    db: Database,
    vec_results: list[tuple[int, float]],
    neighbor_ids: set[int],
    sources_cache: dict[int, Source],
    vec_weight: float,
    discount: float = 0.7,
) -> list[SearchResult]:
    """Score neighbor sources using pre-computed vec results with expansion discount.

    Expansion uses rank-based RRF scoring (consistent with main search) rather
    than raw cosine similarity, with a 0.7× discount to keep expanded results
    below direct matches.
    """
    # Filter vec results to neighbor chunks, batch-fetch
    neighbor_chunk_ids = [cid for cid, dist in vec_results if isinstance(dist, (int, float)) and math.isfinite(dist)]
    chunks_by_id = db.get_chunks_by_ids(neighbor_chunk_ids)

    neighbor_chunks: dict[int, list[tuple[Chunk, float]]] = {}
    for cid, dist in vec_results:
        if not (isinstance(dist, (int, float)) and math.isfinite(dist)):
            continue
        chunk = chunks_by_id.get(cid)
        if chunk is not None and chunk.source_id in neighbor_ids:
            neighbor_chunks.setdefault(chunk.source_id, []).append(
                (chunk, dist)
            )

    # Pick best chunk per source, rank by distance
    best_per_source: list[tuple[int, Chunk, float]] = []
    for sid, chunk_dists in neighbor_chunks.items():
        chunk_dists.sort(key=lambda x: x[1])
        best_chunk, best_dist = chunk_dists[0]
        best_per_source.append((sid, best_chunk, best_dist))

    best_per_source.sort(key=lambda x: x[2])

    # Batch-fetch any missing sources
    missing_ids = [sid for sid, _, _ in best_per_source if sid not in sources_cache]
    if missing_ids:
        new_sources = db.get_sources_by_ids(missing_ids)
        sources_cache.update(new_sources)

    results: list[SearchResult] = []
    for rank, (sid, chunk, dist) in enumerate(best_per_source, start=1):
        src = sources_cache.get(sid)
        if src is None:
            continue

        # Discounted RRF-style score
        score = discount * vec_weight / (60 + rank)

        results.append(SearchResult(
            source_id=sid,
            chunk_id=chunk.id,
            path=src.path,
            title=src.title,
            content=chunk.content,
            score=score,
            indegree=src.indegree,
        ))

    return results

"""Search engine — four-channel RRF fusion with optional cross-encoder reranking and graph expansion."""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seeklink.db import Database
from seeklink.embedder import Embedder
from seeklink.ingest import FRONTMATTER_RE
from seeklink.models import Chunk, Source

if TYPE_CHECKING:
    from seeklink.reranker import Reranker

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
    # 1-indexed inclusive line range of the best chunk in the on-disk file.
    # 0 means "not yet computed" or "not applicable" (e.g. title-only match).
    # Populated by compute_lines_for_results() after the main ranking pass.
    line_start: int = 0
    line_end: int = 0


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
    title_weight: float = 1.5,
    path_prefix: str | None = None,
    tags: list[str] | None = None,
    folder: str | None = None,
    reranker: "Reranker | None" = None,
    rerank_k: int = 20,
    vault_root: Path | None = None,
) -> list[SearchResult]:
    """Search the knowledge base using four-channel RRF fusion.

    Channels: BM25 (keyword chunks), vector (semantic), indegree (quality
    prior), title/alias (source-level FTS5).
    Optional graph expansion follows wiki-links from top results.

    Note on title_weight (default 1.5): rank-1 title match contributes
    1.5/61 ≈ 0.025, which is roughly on par with BM25+vector combined at
    rank 1 each (~0.032). This keeps title as a meaningful boost for
    alias lookups (e.g., [[fsrs-basics]]) without systematically
    dominating content matches from artifacts that lack title metadata
    (logs, journal entries, cards). Agents can override via CLI
    --title-weight flag per query: raise for "find the definitive article"
    queries, lower (toward 0) for "surface raw moments" queries.

    Filtering (applied as post-filter on candidate pool):
    - tags: only include sources with ALL specified tags
    - folder: only include sources whose path starts with this prefix
    - path_prefix: alias for folder (legacy)
    """
    if not query.strip() or top_k <= 0:
        return []

    # Merge folder into path_prefix
    effective_prefix = folder or path_prefix
    # Ensure folder prefix ends with / for directory boundary
    if effective_prefix and not effective_prefix.endswith("/"):
        effective_prefix += "/"

    # When filtering, increase vec limit to reduce post-filter recall loss
    has_filter = bool(tags or effective_prefix)

    # Channel 1: BM25 (chunk-level)
    bm25_results = _safe_fts(db, query, limit=50)
    bm25_best = _best_chunk_per_source(bm25_results)
    bm25_ranked = sorted(bm25_best.keys(), key=lambda sid: bm25_best[sid][1])
    bm25_ranks = {sid: i + 1 for i, sid in enumerate(bm25_ranked)}

    # Channel 2: Vector (use larger k when expansion or filtering requested)
    vec_limit = 200 if (expand or has_filter) else 50
    query_emb = _safe_embed(embedder, query)
    if query_emb is not None:
        vec_results = db.search_vec(query_emb, k=vec_limit)
    else:
        vec_results = []

    chunk_ids = [cid for cid, dist in vec_results if isinstance(dist, (int, float)) and math.isfinite(dist)]
    valid_distances = {cid: dist for cid, dist in vec_results if isinstance(dist, (int, float)) and math.isfinite(dist)}
    chunks_by_id = db.get_chunks_by_ids(chunk_ids)

    vec_chunks_with_dist: list[tuple[Chunk, float]] = [
        (chunks_by_id[cid], valid_distances[cid])
        for cid in chunk_ids
        if cid in chunks_by_id
    ]
    vec_best = _best_chunk_per_source(vec_chunks_with_dist)
    vec_ranked = sorted(vec_best.keys(), key=lambda sid: vec_best[sid][1])
    vec_ranks = {sid: i + 1 for i, sid in enumerate(vec_ranked)}

    # Channel 4: Title/alias (source-level FTS5)
    title_results = db.search_fts_sources(query, limit=50)
    title_ranked = sorted(
        [sid for sid, _ in title_results],
        key=lambda sid: dict(title_results).get(sid, 0),
    )
    title_ranks = {sid: i + 1 for i, sid in enumerate(title_ranked)}

    # Candidate pool: union of all channel source IDs
    candidate_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys()) | set(title_ranks.keys())
    if not candidate_ids:
        return []

    # Batch-fetch sources
    sources = db.get_sources_by_ids(list(candidate_ids))

    # Post-filter: folder/path_prefix
    if effective_prefix:
        candidate_ids = {
            sid for sid in candidate_ids
            if sid in sources and sources[sid].path.startswith(effective_prefix)
        }
        if not candidate_ids:
            return []

    # Post-filter: tags (source must have ALL specified tags)
    if tags:
        tagged_ids: set[int] | None = None
        for tag in tags:
            tag_sources = {s.id for s in db.get_sources_by_tag(tag)}
            if tagged_ids is None:
                tagged_ids = tag_sources
            else:
                tagged_ids &= tag_sources
        if tagged_ids is None:
            tagged_ids = set()
        candidate_ids &= tagged_ids
        if not candidate_ids:
            return []

    # Apply filter to per-channel ranks
    bm25_ranks = {k: v for k, v in bm25_ranks.items() if k in candidate_ids}
    vec_ranks = {k: v for k, v in vec_ranks.items() if k in candidate_ids}
    title_ranks = {k: v for k, v in title_ranks.items() if k in candidate_ids}

    # Channel 3: Indegree (rank filtered candidates by indegree descending)
    indeg_ranked = sorted(
        candidate_ids,
        key=lambda sid: sources[sid].indegree if sid in sources else 0,
        reverse=True,
    )
    indeg_ranks = {sid: i + 1 for i, sid in enumerate(indeg_ranked)}

    # RRF fusion (4 channels)
    scores = _rrf_fuse(
        [bm25_ranks, vec_ranks, indeg_ranks, title_ranks],
        weights=[bm25_weight, vec_weight, indegree_weight, title_weight],
    )

    # Sort by score descending. When a reranker is provided, fetch a larger
    # candidate pool (rerank_k) so the cross-encoder has meaningful room to
    # reorder. Without a reranker, cut directly to top_k.
    reranking_enabled = reranker is not None and not reranker.disabled
    candidate_k = max(rerank_k, top_k) if reranking_enabled else top_k
    ranked = sorted(scores.keys(), key=lambda sid: scores[sid], reverse=True)
    ranked = ranked[:candidate_k]

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

    # For title-only matches (no chunk), create a result with title as content
    for sid in ranked:
        if sid not in best_chunks and sid in sources:
            src = sources[sid]
            results.append(SearchResult(
                source_id=sid,
                chunk_id=0,
                path=src.path,
                title=src.title,
                content=src.title or src.path,
                score=scores[sid],
                indegree=src.indegree,
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    results = results[:candidate_k]

    # Cross-encoder reranking — v0.3 Item 1.
    #
    # Empirical discovery during v0.3 blind-testing: an unconditional
    # blend (qmd's `1/rank` formula, or even a normalized-RRF-score
    # formula) over-protects rank 1 for the class of queries where the
    # first-stage RRF puts a genuinely BAD candidate at rank 1. The
    # reranker correctly identifies a better candidate at (say) rank 11,
    # but position weighting prevents it from winning. Example from the
    # test corpus: query `把文档切块放进向量库` — `vector-embeddings.md` is at
    # RRF rank 11 but correctly scored #1 by the reranker; any
    # always-on blend keeps `agent-memory-patterns.md` at rank 1.
    #
    # The failure mode we DO want to prevent is "exact title/alias hit
    # wins rank 1 cleanly from the title channel, but the reranker
    # demotes it because the note is short or differently-phrased". In
    # that case the title signal is a strong confidence prior and we
    # want to protect rank 1.
    #
    # Design: TITLE-GATED BLENDING. Apply position-aware blending
    # across the full candidate pool ONLY when the title channel
    # produced a rank-1 match and that match survived into the pool.
    # Otherwise fall back to pre-v0.3 pure reranker replacement so
    # the reranker can correct wrong first-stage ordering.
    #
    #   IF title channel rank 1 is in the rerank candidate pool:
    #       blended_i = alpha_i * (rrf_i / max_rrf) + (1 - alpha_i) * rerank_i
    #       with alpha = 0.60 (rank 1-3), 0.50 (4-10), 0.40 (11+)
    #   ELSE:
    #       blended_i = rerank_i  (pure reranker, v0.2.2 behavior)
    #
    # Why "anywhere in pool", not strictly "pool rank 1":
    # Empirically on the built-in blind test, exact-title queries like
    # `Zettelkasten` sometimes put the title-winning note at pool
    # rank 2 (e.g. indegree boost pushes another note to pool rank 1).
    # Applying Option-B blending to the whole pool still lifts
    # zettelkasten.md back to rank 1 via its high normalized RRF
    # score and reasonable rerank, matching user intent. Gating
    # strictly on "pool rank 1 === title winner" would drop those
    # wins. See tests/blind/results/A_v0.3_optC*.json for the
    # measured impact.
    #
    # Theoretical risk: if a query produces a *weak* title hit whose
    # winner is far down in the pool, this gate still activates blending
    # for all candidates. On the 22-query blind test we never observed
    # this (non-exact-title queries had empty title_ranks and the gate
    # stayed off), but it's a latent corner case worth monitoring.
    # Revisit this gate once we have labeled data that exercises
    # weak-title-match queries.
    #
    # This preserves Zettelkasten / attention / RRF-style exact-hit
    # queries at rank 1 while letting reranker correct poor first-stage
    # ordering for everything else.
    #
    # Reranker failures downgrade gracefully (rerank() returns None →
    # keep first-stage RRF ordering).
    if reranking_enabled and len(results) > 1:
        passages = [r.content for r in results]
        rerank_scores = reranker.rerank(query, passages)
        if rerank_scores is not None and len(rerank_scores) == len(results):
            # Title-channel rank 1 is the strongest "this source was
            # confidently identified by title/alias" signal. If that
            # source is anywhere in the candidate pool, apply position
            # blending; otherwise trust the reranker fully. See the
            # block comment above for the "anywhere in pool" rationale.
            title_rank_1_sid: int | None = None
            for sid, rank in title_ranks.items():
                if rank == 1:
                    title_rank_1_sid = sid
                    break
            candidate_sids = {r.source_id for r in results}
            apply_blend = (
                title_rank_1_sid is not None
                and title_rank_1_sid in candidate_sids
            )

            max_rrf = results[0].score if results[0].score > 0 else 1.0
            blended: list[SearchResult] = []
            for i, r in enumerate(results):
                rerank_s = float(rerank_scores[i])
                if apply_blend:
                    norm_score = r.score / max_rrf
                    rrf_rank = i + 1
                    if rrf_rank <= 3:
                        alpha = 0.60
                    elif rrf_rank <= 10:
                        alpha = 0.50
                    else:
                        alpha = 0.40
                    blended_score = alpha * norm_score + (1.0 - alpha) * rerank_s
                else:
                    # No title-channel confidence → pure reranker override
                    # (pre-v0.3 behavior). Reranker has the full say.
                    blended_score = rerank_s
                blended.append(SearchResult(
                    source_id=r.source_id,
                    chunk_id=r.chunk_id,
                    path=r.path,
                    title=r.title,
                    content=r.content,
                    score=blended_score,
                    indegree=r.indegree,
                ))
            blended.sort(key=lambda r: r.score, reverse=True)
            results = blended

    results = results[:top_k]

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

    # v0.3 Item 2 — populate line_start/line_end on each result.
    # Caller must pass vault_root for the mapping to work; if absent,
    # line fields remain at their default of 0 (backward compatible).
    if vault_root is not None:
        results = compute_lines_for_results(db, vault_root, results)

    return results


# ── Line-range retrieval helpers (v0.3 Item 2) ──────────────────────


def body_offset_to_file_line(full_text: str, body_char_offset: int) -> int:
    """Map a char offset in the frontmatter-stripped body back to a
    1-indexed line number in the on-disk file.

    Invariant: if the file has frontmatter, the frontmatter prefix
    occupies lines 1..N of the file; the body starts at line N+1
    (where N = newlines inside the frontmatter block, inclusive of
    the opening/closing --- lines). If the file no longer has
    frontmatter on disk (user deleted it after indexing), treat the
    full file as body — that is the correct mapping for a
    body-relative offset even then.

    Args:
        full_text: current on-disk file content (after `read_text`
            universal-newline translation, so only `\n`).
        body_char_offset: a 0-indexed character offset inside the
            post-frontmatter body that ingest chunked against.
    """
    m = FRONTMATTER_RE.match(full_text)
    if m:
        frontmatter_len = m.end()
        prefix_lines = full_text.count("\n", 0, frontmatter_len)
    else:
        frontmatter_len = 0
        prefix_lines = 0

    body_abs_offset = frontmatter_len + body_char_offset
    lines_before = full_text.count("\n", frontmatter_len, body_abs_offset)
    return prefix_lines + lines_before + 1


def compute_lines_for_results(
    db: Database,
    vault_root: Path,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Return a new list of SearchResult with line_start/line_end filled in.

    Reads each unique source file from disk once, maps the chunk's
    stored char offsets back to 1-indexed inclusive line numbers.

    Title-only matches (chunk_id == 0) get line_start=line_end=1 (the
    file itself is the match). Missing files get (0, 0) and emit a
    warning to the logger.

    ChunkSpan.char_end is end-exclusive in seeklink's chunker (see
    chunker.py:3), so line_end is computed from
    `max(char_start, char_end - 1)` to get the 1-indexed inclusive
    last line.
    """
    if not results:
        return results

    # Batch-fetch chunks by chunk_id so we avoid N per-row roundtrips.
    chunk_ids = [r.chunk_id for r in results if r.chunk_id != 0]
    chunks_by_id: dict[int, Chunk] = db.get_chunks_by_ids(chunk_ids) if chunk_ids else {}

    # File read cache for the duration of this call (a single query
    # typically surfaces 5-10 distinct paths).
    file_cache: dict[str, str | None] = {}

    def _read(path: str) -> str | None:
        if path in file_cache:
            return file_cache[path]
        try:
            abs_path = vault_root / path
            text = abs_path.read_text(encoding="utf-8")
            file_cache[path] = text
            return text
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Could not read %s for line mapping: %s", path, e)
            file_cache[path] = None
            return None

    out: list[SearchResult] = []
    for r in results:
        # Title-only match: file is the match, line 1 is the best default —
        # BUT only if the file still exists on disk. If the file was
        # deleted after indexing, treat it like any other missing file
        # (keep line_start=line_end=0 so agents don't see a bogus
        # `path:1` that won't resolve to anything useful).
        if r.chunk_id == 0:
            full_text = _read(r.path)
            if full_text is None:
                out.append(r)
            else:
                out.append(SearchResult(
                    source_id=r.source_id,
                    chunk_id=r.chunk_id,
                    path=r.path,
                    title=r.title,
                    content=r.content,
                    score=r.score,
                    indegree=r.indegree,
                    line_start=1,
                    line_end=1,
                ))
            continue

        chunk = chunks_by_id.get(r.chunk_id)
        full_text = _read(r.path)
        if chunk is None or full_text is None or chunk.char_start is None or chunk.char_end is None:
            # Fall back to 0/0 — caller can check and avoid printing lines
            out.append(r)
            continue

        line_start = body_offset_to_file_line(full_text, chunk.char_start)
        # char_end is exclusive; map the last INCLUDED char to get inclusive line_end
        end_inclusive = max(chunk.char_start, chunk.char_end - 1)
        line_end = body_offset_to_file_line(full_text, end_inclusive)

        out.append(SearchResult(
            source_id=r.source_id,
            chunk_id=r.chunk_id,
            path=r.path,
            title=r.title,
            content=r.content,
            score=r.score,
            indegree=r.indegree,
            line_start=line_start,
            line_end=line_end,
        ))

    return out


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
    channel_ranks: list[dict[int, int]],
    weights: list[float],
    k: int = 60,
) -> dict[int, float]:
    """Compute RRF scores for candidates across N channels.

    Each channel contributes weight / (k + rank) for candidates it contains.
    """
    all_ids: set[int] = set()
    for ranks in channel_ranks:
        all_ids |= ranks.keys()

    scores: dict[int, float] = {}
    for sid in all_ids:
        score = 0.0
        for ranks, weight in zip(channel_ranks, weights):
            if sid in ranks:
                score += weight / (k + ranks[sid])
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

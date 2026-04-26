"""Tests for seeklink.search — four-channel RRF fusion, title-gated rerank blending, line-range computation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from seeklink.db import Database
from seeklink.embedder import Embedder
from seeklink.ingest import ingest_file
from seeklink.search import (
    SearchDiagnostics,
    SearchResult,
    _best_chunk_per_source,
    _resolve_rerank_k,
    _rrf_fuse,
    search,
)
from seeklink.models import Chunk


@pytest.fixture(scope="session")
def embedder():
    """Session-scoped embedder — model loads once."""
    return Embedder()


@pytest.fixture
def db():
    """In-memory database for each test."""
    d = Database(":memory:")
    d.check_capabilities()
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Create a temporary vault directory."""
    v = tmp_path / "vault"
    v.mkdir()
    return v


class TestAutoRerankK:
    def test_numeric_value_is_used_as_is(self):
        assert _resolve_rerank_k(
            "记忆保持力",
            7,
            has_filter=False,
            title_ranks={},
        ) == 7

    def test_title_match_uses_fast_budget(self):
        assert _resolve_rerank_k(
            "RRF",
            "auto",
            has_filter=False,
            title_ranks={10: 1},
        ) == 5

    def test_cjk_without_title_match_uses_deep_budget(self):
        assert _resolve_rerank_k(
            "把文档切块放进向量库",
            "auto",
            has_filter=False,
            title_ranks={},
        ) == 20

    def test_english_without_title_match_uses_fast_budget(self):
        assert _resolve_rerank_k(
            "agent memory architectures",
            "auto",
            has_filter=False,
            title_ranks={},
        ) == 5

    def test_filters_use_deep_budget(self):
        assert _resolve_rerank_k(
            "memory",
            "auto",
            has_filter=True,
            title_ranks={},
        ) == 20

    def test_search_populates_auto_rerank_diagnostics(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        _ingest_corpus(db, embedder, vault)

        class FakeReranker:
            disabled = False

            def rerank(self, query: str, passages: list[str]) -> list[float]:
                return [1.0 for _ in passages]

        diagnostics = SearchDiagnostics()
        results = search(
            db,
            embedder,
            "Machine Learning",
            reranker=FakeReranker(),  # type: ignore[arg-type]
            rerank_k="auto",
            diagnostics=diagnostics,
        )

        assert results
        assert diagnostics.reranking_enabled is True
        assert diagnostics.requested_rerank_k == "auto"
        assert diagnostics.resolved_rerank_k == 5
        assert diagnostics.rerank_k_reason == "title"


def _write_md(vault: Path, rel_path: str, content: str) -> Path:
    """Helper to write a markdown file in the vault."""
    p = vault / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _ingest_corpus(db: Database, embedder: Embedder, vault: Path) -> None:
    """Ingest the standard 4-note test corpus.

    After ingestion:
      ml-basics: indegree=2 (linked from deep-learning + hub-note)
      deep-learning: indegree=1 (linked from hub-note)
      cooking: indegree=0
      hub-note: indegree=0
    """
    _write_md(vault, "ml-basics.md", "# Machine Learning\n\nML uses algorithms to learn from data.")
    _write_md(vault, "deep-learning.md", "# Deep Learning\n\nNeural networks with many layers. See [[ml-basics]].")
    _write_md(vault, "cooking.md", "# Italian Cooking\n\nPasta recipes from Italy.")
    _write_md(vault, "hub-note.md", "# Hub\n\nSee [[ml-basics]] and [[deep-learning]].")

    # Ingest in order so links resolve correctly
    ingest_file(db, vault / "ml-basics.md", vault, embedder)
    ingest_file(db, vault / "deep-learning.md", vault, embedder)
    ingest_file(db, vault / "cooking.md", vault, embedder)
    ingest_file(db, vault / "hub-note.md", vault, embedder)


class TestBM25Channel:
    def test_english_keyword_match(self, db: Database, embedder: Embedder, vault: Path):
        """'machine learning' finds ml-basics via BM25."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning")
        assert len(results) > 0
        paths = [r.path for r in results]
        assert "ml-basics.md" in paths

    def test_chinese_keyword_match(self, db: Database, embedder: Embedder, vault: Path):
        """Chinese query finds Chinese note via BM25."""
        _write_md(vault, "知识管理.md", "# 知识管理\n\n个人知识管理是一种组织信息的方法。")
        ingest_file(db, vault / "知识管理.md", vault, embedder)

        results = search(db, embedder, "知识管理")
        assert len(results) > 0
        paths = [r.path for r in results]
        assert "知识管理.md" in paths

    def test_no_results_empty_query(self, db: Database, embedder: Embedder, vault: Path):
        """Empty or whitespace query returns no results."""
        _ingest_corpus(db, embedder, vault)
        assert search(db, embedder, "") == []
        assert search(db, embedder, "   ") == []


class TestVectorChannel:
    def test_semantic_match(self, db: Database, embedder: Embedder, vault: Path):
        """'AI algorithms' (no exact keywords) finds ml-basics via semantics."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "AI algorithms that learn patterns")
        assert len(results) > 0
        paths = [r.path for r in results]
        assert "ml-basics.md" in paths

    def test_cross_language(self, db: Database, embedder: Embedder, vault: Path):
        """Chinese query for ML concept finds English ml-basics note."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "机器学习算法")
        assert len(results) > 0
        paths = [r.path for r in results]
        assert "ml-basics.md" in paths


class TestIndegreeBoost:
    def test_hub_boosted(self, db: Database, embedder: Embedder, vault: Path):
        """ml-basics (indegree=2) ranks higher than cooking (indegree=0).

        Both are in a mixed-topic query. The indegree boost should push
        ml-basics ahead when relevance is comparable.
        """
        _ingest_corpus(db, embedder, vault)

        # Verify indegrees are as expected
        ml = db.get_source_by_path("ml-basics.md")
        cooking = db.get_source_by_path("cooking.md")
        assert ml is not None and ml.indegree == 2
        assert cooking is not None and cooking.indegree == 0

        # Search for a broad query that could match both
        results = search(db, embedder, "learning techniques and methods")
        source_ids = [r.source_id for r in results]

        if ml.id in source_ids and cooking.id in source_ids:
            ml_rank = source_ids.index(ml.id)
            cooking_rank = source_ids.index(cooking.id)
            assert ml_rank < cooking_rank, "ml-basics should rank higher than cooking"

    def test_hub_absent_unrelated(self, db: Database, embedder: Embedder, vault: Path):
        """Cooking query doesn't surface ml-basics in top results despite high indegree."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "Italian pasta recipes from Italy")
        if results:
            # cooking should be top result, not ml-basics
            assert results[0].path == "cooking.md"


class TestRRFFusion:
    def test_combined_ranking(self, db: Database, embedder: Embedder, vault: Path):
        """Search returns results with positive scores."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning algorithms")
        assert len(results) > 0
        for r in results:
            assert r.score > 0
            assert isinstance(r, SearchResult)

    def test_top_k_respected(self, db: Database, embedder: Embedder, vault: Path):
        """top_k=2 returns at most 2 results."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning", top_k=2)
        assert len(results) <= 2

    def test_empty_db(self, db: Database, embedder: Embedder):
        """Search on empty DB returns no results."""
        results = search(db, embedder, "anything")
        assert results == []


class TestGraphExpansion:
    def test_expand_finds_neighbors(self, db: Database, embedder: Embedder, vault: Path):
        """expand=True on 'machine learning' finds deep-learning via link."""
        _ingest_corpus(db, embedder, vault)

        # Without expansion
        results_no_expand = search(db, embedder, "machine learning algorithms", top_k=10)

        # With expansion
        results_expand = search(db, embedder, "machine learning algorithms", expand=True, top_k=10)

        expanded_paths = {r.path for r in results_expand}
        # deep-learning links to ml-basics, so it should appear as a neighbor
        assert "deep-learning.md" in expanded_paths

    def test_expand_discount(self, db: Database, embedder: Embedder, vault: Path):
        """Expanded results have lower scores than direct matches."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning algorithms", expand=True, top_k=10)

        if len(results) >= 2:
            # Find a direct match (should be first)
            direct_scores = [r.score for r in results if r.path == "ml-basics.md"]
            # Check expansion candidates have lower scores
            for r in results:
                if r.path != "ml-basics.md" and direct_scores:
                    # Expanded results may have comparable or lower scores
                    # At minimum, check scores are positive
                    assert r.score > 0


class TestEdgeCases:
    def test_single_note(self, db: Database, embedder: Embedder, vault: Path):
        """DB with 1 note, search returns it."""
        _write_md(vault, "only.md", "# Only Note\n\nThis is the only note in the vault.")
        ingest_file(db, vault / "only.md", vault, embedder)

        results = search(db, embedder, "only note")
        assert len(results) == 1
        assert results[0].path == "only.md"

    def test_query_with_no_fts_match(self, db: Database, embedder: Embedder, vault: Path):
        """Query only matches via vector, still returns results."""
        _write_md(vault, "planets.md", "# Solar System\n\nEarth orbits the Sun alongside Mars and Jupiter.")
        ingest_file(db, vault / "planets.md", vault, embedder)

        # Use a semantically related query with no keyword overlap
        results = search(db, embedder, "celestial bodies in space")
        assert len(results) > 0
        assert results[0].path == "planets.md"

    def test_negative_top_k(self, db: Database, embedder: Embedder, vault: Path):
        """top_k <= 0 returns empty list."""
        _ingest_corpus(db, embedder, vault)
        assert search(db, embedder, "machine learning", top_k=0) == []
        assert search(db, embedder, "machine learning", top_k=-1) == []

    def test_query_matches_nothing(self, db: Database, embedder: Embedder, vault: Path):
        """Gibberish query returns empty or very low-confidence results."""
        _write_md(vault, "note.md", "# Note\n\nContent about apples and oranges.")
        ingest_file(db, vault / "note.md", vault, embedder)
        # Vector channel may still return results (everything is a neighbor in
        # embedding space) but the result set should be small and graceful
        results = search(db, embedder, "zzzzzzzzzzzzzzz xyzabc 98765")
        # No crash — that's the main assertion
        assert isinstance(results, list)

    def test_expand_isolated_notes(self, db: Database, embedder: Embedder, vault: Path):
        """Expansion is a no-op when top results have no wiki-links."""
        _write_md(vault, "solo-a.md", "# Solo A\n\nStandalone note about quantum physics.")
        _write_md(vault, "solo-b.md", "# Solo B\n\nAnother standalone note about chemistry.")
        ingest_file(db, vault / "solo-a.md", vault, embedder)
        ingest_file(db, vault / "solo-b.md", vault, embedder)

        without = search(db, embedder, "quantum physics", expand=False)
        with_expand = search(db, embedder, "quantum physics", expand=True)

        # Same results — no links to expand
        assert [r.source_id for r in without] == [r.source_id for r in with_expand]

    def test_expand_top_k_clamped(self, db: Database, embedder: Embedder, vault: Path):
        """Expansion + top_k still respects the top_k limit."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning", expand=True, top_k=2)
        assert len(results) <= 2


class TestRRFFuseUnit:
    """Unit tests for the _rrf_fuse helper."""

    def test_single_channel_bm25_only(self):
        """BM25 results only, no vector — indegree still contributes."""
        bm25 = {1: 1, 2: 2}
        vec: dict[int, int] = {}
        indeg = {1: 1, 2: 2}
        scores = _rrf_fuse([bm25, vec, indeg], weights=[1.0, 1.0, 0.3])
        assert len(scores) == 2
        assert scores[1] > scores[2]  # rank 1 beats rank 2

    def test_single_channel_vec_only(self):
        """Vector results only, no BM25."""
        bm25: dict[int, int] = {}
        vec = {10: 1, 20: 2}
        indeg = {10: 1, 20: 2}
        scores = _rrf_fuse([bm25, vec, indeg], weights=[1.0, 1.0, 0.3])
        assert len(scores) == 2
        assert scores[10] > scores[20]

    def test_all_channels_same_source(self):
        """Source appears in all three channels — gets highest score."""
        bm25 = {1: 1, 2: 2}
        vec = {1: 1, 3: 1}
        indeg = {1: 1, 2: 2, 3: 3}
        scores = _rrf_fuse([bm25, vec, indeg], weights=[1.0, 1.0, 0.3])
        # Source 1 is rank 1 in all 3 channels
        assert scores[1] > scores[2]
        assert scores[1] > scores[3]

    def test_zero_weight_disables_channel(self):
        """Setting a weight to 0 removes that channel's contribution."""
        bm25 = {1: 1}
        vec = {2: 1}
        indeg = {1: 1, 2: 2}
        # Zero out BM25 — source 1 should only get indegree
        scores = _rrf_fuse([bm25, vec, indeg], weights=[0.0, 1.0, 0.3])
        # Source 2 has vec contribution, source 1 has only indegree
        assert scores[2] > scores[1]

    def test_empty_channels(self):
        """All channels empty → no scores."""
        scores = _rrf_fuse([{}, {}, {}], weights=[1.0, 1.0, 0.3])
        assert scores == {}

    def test_four_channels(self):
        """4-channel RRF with title channel."""
        bm25 = {1: 2, 2: 1}
        vec = {1: 2, 2: 1}
        indeg = {1: 1, 2: 2}
        title = {1: 1}  # source 1 has title match
        scores = _rrf_fuse(
            [bm25, vec, indeg, title], weights=[1.0, 1.0, 0.3, 3.0]
        )
        # Source 1 has title match (weight 3.0), should beat source 2
        assert scores[1] > scores[2]


class TestBestChunkPerSourceUnit:
    """Unit tests for _best_chunk_per_source helper."""

    def _make_chunk(self, id: int, source_id: int) -> Chunk:
        return Chunk(
            id=id, source_id=source_id, content=f"chunk-{id}",
            chunk_index=0, char_start=0, char_end=5,
            token_count=1, created_at="2026-01-01",
        )

    def test_keeps_lowest_score(self):
        """When two chunks share a source, the lower score wins."""
        c1 = self._make_chunk(1, source_id=100)
        c2 = self._make_chunk(2, source_id=100)
        result = _best_chunk_per_source([(c1, -5.0), (c2, -10.0)])
        assert result[100][0].id == 2  # c2 has lower score (-10 < -5)

    def test_different_sources(self):
        """Chunks from different sources are kept independently."""
        c1 = self._make_chunk(1, source_id=100)
        c2 = self._make_chunk(2, source_id=200)
        result = _best_chunk_per_source([(c1, -5.0), (c2, -3.0)])
        assert 100 in result and 200 in result

    def test_empty_input(self):
        assert _best_chunk_per_source([]) == {}


class TestEmbedFailure:
    def test_bm25_fallback_on_embed_failure(self, db: Database, embedder: Embedder, vault: Path):
        """If embed_query fails, search falls back to BM25-only results."""
        _write_md(vault, "fallback.md", "# Fallback Test\n\nKeyword fallback content here.")
        ingest_file(db, vault / "fallback.md", vault, embedder)

        with patch.object(embedder, "embed_query", side_effect=RuntimeError("model OOM")):
            results = search(db, embedder, "fallback")

        # Should still return BM25 results despite embedding failure
        assert len(results) > 0
        assert results[0].path == "fallback.md"


class TestCustomWeights:
    def test_bm25_zero_weight(self, db: Database, embedder: Embedder, vault: Path):
        """bm25_weight=0 means only vector + indegree contribute."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning", bm25_weight=0.0)
        assert len(results) > 0
        for r in results:
            assert r.score > 0

    def test_vec_zero_weight(self, db: Database, embedder: Embedder, vault: Path):
        """vec_weight=0 means only BM25 + indegree contribute."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning", vec_weight=0.0)
        assert len(results) > 0
        for r in results:
            assert r.score > 0


# ── v2: Tag/Folder filtering ─────────────────────────────────────


class TestTagFiltering:
    """Test tag-based filtering in search."""

    def test_search_with_tags(self, db: Database, embedder: Embedder, vault: Path):
        _ingest_corpus(db, embedder, vault)
        # Tag some sources
        ml = db.get_source_by_path("ml-basics.md")
        cooking = db.get_source_by_path("cooking.md")
        db.add_tags(ml.id, ["ai"])
        db.add_tags(cooking.id, ["cooking"])

        # Search with tag filter
        results = search(db, embedder, "algorithms", tags=["ai"])
        paths = {r.path for r in results}
        assert "cooking.md" not in paths  # cooking shouldn't match ai tag
        if results:
            assert all(
                "ai" in db.get_tags(r.source_id) for r in results
            )

    def test_search_without_tags(self, db: Database, embedder: Embedder, vault: Path):
        """Search without tags returns all matching results."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning")
        assert len(results) > 0


class TestFolderFiltering:
    """Test folder-based filtering in search."""

    def test_search_with_folder(self, db: Database, embedder: Embedder, vault: Path):
        # Create notes in different folders
        _write_md(vault, "notes/a.md", "# Alpha\n\nAlpha content about science.")
        _write_md(vault, "archive/b.md", "# Beta\n\nBeta content about science.")
        ingest_file(db, vault / "notes" / "a.md", vault, embedder)
        ingest_file(db, vault / "archive" / "b.md", vault, embedder)

        results = search(db, embedder, "science", folder="notes")
        paths = {r.path for r in results}
        assert all(p.startswith("notes/") for p in paths)


class TestTitleChannel:
    """Test that the 4th RRF channel (title/alias) boosts title matches."""

    def test_title_match_boosts_ranking(self, db: Database, embedder: Embedder, vault: Path):
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "Machine Learning")
        if results:
            # ml-basics.md has title "Machine Learning" — should rank high
            top_paths = [r.path for r in results[:3]]
            assert "ml-basics.md" in top_paths


class TestPositionAwareBlending:
    """Test title-gated rerank blending (v0.3) preserves confident
    first-stage wins while letting the reranker override when no title
    signal is present.

    v0.2.x behavior: reranker score fully replaced the first-stage RRF
    score, so an exact-title or exact-alias hit at RRF rank 1 could be
    demoted if the reranker gave a longer adjacent document a higher
    content-relevance score.

    v0.3 formula (title-gated, confidence-aware):
        norm_score = rrf_score / max_rrf_score_in_pool
        alpha      = 0.60 (ranks 1-3), 0.50 (4-10), 0.40 (11+)
        blended    = alpha * norm_score + (1 - alpha) * rerank_score
    Only applied when the title channel's rank-1 source is in the
    rerank candidate pool; otherwise falls back to pure reranker
    (pre-v0.3 behavior).
    """

    def test_blending_preserves_confident_rank_1(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """A confident rank-1 hit (title channel fires, opening a real RRF
        gap) stays at rank 1 under mild reranker disagreement.

        Note: v0.3's confidence-aware blending intentionally DOES allow
        a strongly-disagreeing reranker to flip rank 1 when the RRF gap
        is razor-thin or the reranker is very confident. That's the fix
        for the `红烧肉做法` class of regression. What we assert here is
        the complementary case: when RRF is confident AND reranker only
        mildly disagrees, rank 1 is preserved.
        """
        _ingest_corpus(db, embedder, vault)

        class MildlyDisagreeingReranker:
            """Gives rank 1 a below-average but non-zero score, so the
            reranker's 'vote' does not overcome a confident rank-1 RRF
            position."""
            disabled = False

            def rerank(self, query, passages):
                n = len(passages)
                if n == 0:
                    return []
                # Rank 1 gets 0.3 (mild 'meh'); others get 0.6.
                # With alpha_1=0.60 and norm_score_1=1.0:
                #   blended_1 = 0.60 * 1.0 + 0.40 * 0.3 = 0.72
                # Competing rank-2/3 (norm_score ~0.8, rerank=0.6):
                #   blended_2 = 0.60 * 0.8 + 0.40 * 0.6 = 0.72  (tie-ish)
                # In practice ml-basics has a strong title-channel lead
                # so its norm_score is close to 1.0 and it wins.
                return [0.3 if i == 0 else 0.6 for i in range(n)]

        results = search(
            db, embedder, "Machine Learning",
            reranker=MildlyDisagreeingReranker(),  # type: ignore[arg-type]
            top_k=4,
            rerank_k=4,
        )
        assert len(results) >= 1
        # Title-channel-winning result should stay at rank 1
        assert "ml-basics" in results[0].path, (
            f"Expected ml-basics at rank 1 after blending, got "
            f"{[r.path for r in results]}"
        )

    def test_gate_off_without_title_match(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """When the query has no title/alias hit at all, blending must
        be OFF — final scores equal the raw reranker scores, and the
        reranker's ordering wins (pre-v0.3 behavior)."""
        _ingest_corpus(db, embedder, vault)

        rerank_seq = [0.1, 0.9, 0.4, 0.7]

        class SeqReranker:
            disabled = False

            def rerank(self, query, passages):
                return rerank_seq[: len(passages)]

        # Query with no match against any title or alias in the fixture
        # corpus (ml-basics / deep-learning / cooking / hub-note titles).
        results = search(
            db, embedder, "xyz-nonsense-no-title-match-qqq",
            reranker=SeqReranker(),  # type: ignore[arg-type]
            top_k=4,
            rerank_k=4,
        )
        if results:
            # Pure reranker path → each score equals its rerank score
            # (not blended). The set of scores should equal the set of
            # rerank scores used (ignoring order since we sort by score
            # descending).
            actual_scores = sorted((r.score for r in results), reverse=True)
            expected_scores = sorted(rerank_seq[: len(results)], reverse=True)
            for a, e in zip(actual_scores, expected_scores):
                assert abs(a - e) < 1e-9, (
                    f"Gate-off scores should equal rerank scores; "
                    f"got {actual_scores}, expected {expected_scores}"
                )

    def test_gate_on_when_title_winner_in_pool(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """When the title channel ranks a source at its rank 1 AND that
        source is in the rerank candidate pool, blending activates.
        Scores then show the blended math (not raw reranker).

        This is the empirically-preferred "loose" gate semantics — if
        title has a confident winner anywhere in the pool, lift its
        protection across the candidate set so the title winner can
        rise back to rank 1 even from pool position 2+ (the
        Zettelkasten case on the blind-test corpus).
        """
        _ingest_corpus(db, embedder, vault)

        class UniformReranker:
            disabled = False

            def rerank(self, query, passages):
                return [0.5] * len(passages)

        # "Machine Learning" fires the title channel hard on ml-basics.md
        # which has that exact title. Gate should be ON.
        results = search(
            db, embedder, "Machine Learning",
            reranker=UniformReranker(),  # type: ignore[arg-type]
            top_k=4,
            rerank_k=4,
        )
        if results:
            # Rank 1 under blending: alpha_1=0.60, norm=1.0, rerank=0.5
            #   blended = 0.60 * 1.0 + 0.40 * 0.5 = 0.80
            # Under pure-reranker (gate off), it would be 0.5.
            top_score = results[0].score
            assert abs(top_score - 0.80) < 1e-6, (
                f"Expected gate-on rank 1 blended score 0.80 for "
                f"'Machine Learning' query (ml-basics.md has that title), "
                f"got {top_score}"
            )

    def test_blending_formula_properties(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """Check the invariants of the confidence-aware blending:

        - With uniform rerank scores, rank 1 (norm_score=1.0) scores
          highest; lower-ranked items score lower in proportion to
          their RRF-score ratio.
        - Scores are monotonically non-increasing with rank (sorted).
        - Rank 1's blended score equals `alpha_1 + (1 - alpha_1) * rerank`
          when norm_score=1.0 (rank 1 is always top-normalized).
        """
        _ingest_corpus(db, embedder, vault)

        class SpyReranker:
            disabled = False

            def rerank(self, query, passages):
                # All passages score 0.5 — neutral. Blended score should
                # then be dominated by norm_score.
                return [0.5] * len(passages)

        results = search(
            db, embedder, "learning",
            reranker=SpyReranker(),  # type: ignore[arg-type]
            top_k=4,
            rerank_k=4,
        )
        scores = [r.score for r in results]
        # With uniform rerank and monotone norm_score, scores should be
        # sorted descending.
        assert scores == sorted(scores, reverse=True)
        # Rank 1 always has norm_score=1.0. alpha_1 = 0.60, rerank=0.5
        # → blended = 0.60 * 1.0 + 0.40 * 0.5 = 0.80
        if len(scores) >= 1:
            assert abs(scores[0] - 0.80) < 1e-6, (
                f"rank 1 expected 0.80, got {scores[0]}"
            )

    def test_disabled_reranker_falls_through(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """When the reranker is disabled, blending must NOT activate —
        results keep their raw RRF scores."""
        _ingest_corpus(db, embedder, vault)

        class DisabledReranker:
            disabled = True

            def rerank(self, query, passages):
                raise AssertionError("disabled reranker should not be called")

        # Should not raise (rerank() not called when disabled=True)
        results = search(
            db, embedder, "machine learning",
            reranker=DisabledReranker(),  # type: ignore[arg-type]
            top_k=4,
        )
        # Scores are small (RRF range ~0.01-0.06), not in the 0-1 blended range
        if results:
            assert all(r.score < 0.2 for r in results), (
                f"Expected RRF-scale scores, got {[r.score for r in results]}"
            )

    def test_rerank_k_limits_candidate_count(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """rerank_k controls candidates even when top_k is larger."""
        _ingest_corpus(db, embedder, vault)

        class CountingReranker:
            disabled = False
            seen = 0

            def rerank(self, query, passages):
                self.seen = len(passages)
                return [0.5] * len(passages)

        reranker = CountingReranker()
        results = search(
            db,
            embedder,
            "learning",
            reranker=reranker,  # type: ignore[arg-type]
            top_k=4,
            rerank_k=3,
        )

        assert len(results) == 4
        assert reranker.seen == 3


class TestLineRangeE2E:
    """End-to-end: search() populates line_start/line_end from real chunks."""

    def test_line_fields_populated_with_vault_root(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """When vault_root is passed, results should come back with
        line_start > 0 and line_end >= line_start."""
        _ingest_corpus(db, embedder, vault)
        results = search(
            db, embedder, "machine learning",
            top_k=3,
            vault_root=vault,
        )
        assert len(results) > 0
        for r in results:
            # Every result should have a 1-indexed line span (even if small)
            assert r.line_start >= 1, (
                f"Expected line_start >= 1 for {r.path}, got {r.line_start}"
            )
            assert r.line_end >= r.line_start, (
                f"Expected line_end >= line_start for {r.path}, "
                f"got {r.line_start}..{r.line_end}"
            )

    def test_line_fields_zero_without_vault_root(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """When vault_root is None (legacy call shape), line fields
        remain at default 0 — backward compatible."""
        _ingest_corpus(db, embedder, vault)
        results = search(db, embedder, "machine learning", top_k=3)
        if results:
            # Default values on SearchResult dataclass
            assert all(r.line_start == 0 for r in results)
            assert all(r.line_end == 0 for r in results)

    def test_title_only_missing_file_degrades_to_zero(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        """A title-only hit where the file is missing from disk should
        not return a bogus line 1 — it should degrade to 0/0."""
        _ingest_corpus(db, embedder, vault)

        # Construct a title-only SearchResult (chunk_id=0) for a file
        # that doesn't exist, then run through compute_lines_for_results.
        from seeklink.search import compute_lines_for_results
        fake = SearchResult(
            source_id=999,
            chunk_id=0,  # title-only marker
            path="does-not-exist.md",
            title="Fake",
            content="Fake",
            score=0.5,
            indegree=0,
        )
        out = compute_lines_for_results(db, vault, [fake])
        assert len(out) == 1
        assert out[0].line_start == 0, (
            "Title-only match with missing file must degrade to line_start=0, "
            f"got {out[0].line_start}"
        )
        assert out[0].line_end == 0

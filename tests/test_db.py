"""Tests for seeklink.db — schema, CRUD, FTS, vec, and link-graph maintenance."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

import seeklink.db as db_module
from seeklink.db import CapabilityError, Database, FTSQuery
from seeklink.models import Chunk, Source, Suggestion, WikiLink


# ── Helpers ──────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())


def _make_source(db: Database, path: str | None = None, title: str | None = None) -> Source:
    """Helper to quickly add a source."""
    return db.add_source(
        uid=_uid(),
        path=path or f"notes/{_uid()[:8]}.md",
        title=title,
    )


def _random_embedding(dim: int = 768) -> bytes:
    """Generate a random float32 embedding as bytes."""
    return np.random.default_rng().standard_normal(dim).astype(np.float32).tobytes()


# ── 1. TestSchemaCreation ────────────────────────────────────────


class TestSchemaCreation:
    """All tables + triggers + indexes exist; schema init is idempotent."""

    def test_tables_exist(self, db: Database):
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        expected = {
            "sources",
            "chunks",
            "wiki_links",
            "suggestions",
            "source_tags",
            "index_metadata",
        }
        assert expected.issubset(tables)

    def test_virtual_tables_exist(self, db: Database):
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "vec_chunks" in tables
        assert "fts_chunks" in tables
        assert "fts_sources" in tables

    def test_vec_table_uses_configured_dimension(self, monkeypatch):
        monkeypatch.setenv("SEEKLINK_EMBEDDING_DIM", "384")
        db = Database(":memory:")
        db.check_capabilities()
        db.init_schema()

        assert db.get_vector_dimension() == 384
        db.close()

    def test_triggers_exist(self, db: Database):
        triggers = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        expected = {
            "chunks_fts_insert",
            "chunks_fts_delete",
            "chunks_fts_update",
            "wikilinks_insert_indegree",
            "wikilinks_delete_indegree",
            "wikilinks_update_indegree_new",
            "wikilinks_update_indegree_old",
            "sources_fts_insert",
            "sources_fts_delete",
            "sources_fts_update",
        }
        assert expected == triggers

    def test_indexes_exist(self, db: Database):
        indexes = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name LIKE 'idx_%'"
            ).fetchall()
        }
        expected = {
            "idx_sources_uid",
            "idx_sources_status",
            "idx_sources_indegree",
            "idx_chunks_source",
            "idx_wikilinks_source",
            "idx_wikilinks_target",
            "idx_wikilinks_target_path",
            "idx_suggestions_status",
            "idx_source_tags_source",
            "idx_source_tags_tag",
        }
        assert expected == indexes

    def test_user_version(self, db: Database):
        version = db.conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == Database.SCHEMA_VERSION

    def test_idempotent(self, db: Database):
        """Calling init_schema twice doesn't error."""
        db.init_schema()  # second call
        version = db.conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == Database.SCHEMA_VERSION

    def test_static_sqlite_build_uses_trigram_fallback(self, monkeypatch):
        monkeypatch.setattr(
            db_module,
            "register_jieba_tokenizer",
            lambda conn: False,
        )
        db = Database(":memory:")
        try:
            assert db._fts_tokenizer == "trigram"
            db.init_schema()
            source = db.add_source(uid=_uid(), path="notes/cjk.md", title="CJK")
            db.add_chunk(source.id, "个人知识管理是一种组织信息的方法。", 0)

            results = db.search_fts("知识管理")

            assert [chunk.source_id for chunk, _ in results] == [source.id]
        finally:
            db.close()


# ── 2. TestCapabilityCheck ───────────────────────────────────────


class TestCapabilityCheck:
    """Rejects bad environment. (On this machine, passes.)"""

    def test_passes_on_good_env(self, db: Database):
        # Already called in fixture; just verify no error
        db.check_capabilities()

    def test_vec_version_works(self, db: Database):
        row = db.conn.execute("SELECT vec_version()").fetchone()
        assert row[0]  # non-empty version string


# ── 3. TestSourceCRUD ────────────────────────────────────────────


class TestSourceCRUD:
    def test_add_and_get(self, db: Database):
        uid = _uid()
        src = db.add_source(uid=uid, path="notes/test.md", title="Test Note")
        assert isinstance(src, Source)
        assert src.uid == uid
        assert src.path == "notes/test.md"
        assert src.title == "Test Note"
        assert src.status == "unprocessed"
        assert src.indegree == 0

    def test_get_by_uid(self, db: Database):
        uid = _uid()
        db.add_source(uid=uid, path="notes/a.md")
        found = db.get_source_by_uid(uid)
        assert found is not None
        assert found.uid == uid

    def test_get_by_path(self, db: Database):
        db.add_source(uid=_uid(), path="notes/b.md")
        found = db.get_source_by_path("notes/b.md")
        assert found is not None
        assert found.path == "notes/b.md"

    def test_list_by_status(self, db: Database):
        db.add_source(uid=_uid(), path="notes/c.md", status="indexed")
        db.add_source(uid=_uid(), path="notes/d.md")
        indexed = db.list_sources(status="indexed")
        unprocessed = db.list_sources(status="unprocessed")
        assert len(indexed) == 1
        assert len(unprocessed) == 1

    def test_rename_preserves_uid(self, db: Database):
        uid = _uid()
        src = db.add_source(uid=uid, path="notes/old.md")
        renamed = db.rename_source(src.id, "notes/new.md")
        assert renamed is not None
        assert renamed.uid == uid  # uid stays the same
        assert renamed.path == "notes/new.md"

    def test_update_source(self, db: Database):
        src = _make_source(db)
        updated = db.update_source(src.id, status="indexed", title="Updated")
        assert updated is not None
        assert updated.status == "indexed"
        assert updated.title == "Updated"

    def test_update_rejects_invalid_columns(self, db: Database):
        """update_source must reject non-whitelisted columns."""
        src = _make_source(db)
        with pytest.raises(ValueError, match="Cannot update columns"):
            db.update_source(src.id, uid="injected")

    def test_delete_cascades(self, db: Database):
        src = _make_source(db)
        chunk = db.add_chunk(src.id, "test content", 0)
        db.upsert_vec(chunk.id, _random_embedding())

        assert db.delete_source(src.id)
        assert db.get_source(src.id) is None
        assert db.get_chunks_by_source(src.id) == []

    def test_delete_cascades_suggestions(self, db: Database):
        """deleting a source must cascade to suggestions."""
        s1 = _make_source(db)
        s2 = _make_source(db)
        db.add_suggestion(s1.id, s2.id, 0.85, "test")

        # Deleting s1 (source_note_id) should not raise IntegrityError
        assert db.delete_source(s1.id)
        assert len(db.get_pending_suggestions()) == 0

    def test_delete_nonexistent(self, db: Database):
        assert not db.delete_source(999)


# ── 4. TestChunkCRUD ─────────────────────────────────────────────


class TestChunkCRUD:
    def test_add_and_get_by_source(self, db: Database):
        src = _make_source(db)
        c1 = db.add_chunk(src.id, "First chunk", 0)
        c2 = db.add_chunk(src.id, "Second chunk", 1)
        chunks = db.get_chunks_by_source(src.id)
        assert len(chunks) == 2
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    def test_fts_trigger_fires(self, db: Database):
        """Insert into chunks → FTS trigger → FTS search finds it."""
        src = _make_source(db)
        db.add_chunk(src.id, "Machine learning fundamentals and neural networks", 0)

        results = db.search_fts("machine learning")
        assert len(results) >= 1
        chunk, rank = results[0]
        assert "machine" in chunk.content.lower()

    def test_fts_chinese(self, db: Database):
        src = _make_source(db)
        db.add_chunk(src.id, "深度学习使用反向传播算法训练神经网络", 0)

        results = db.search_fts("深度学习")
        assert len(results) >= 1

    def test_fts_compound_word(self, db: Database):
        """jieba search mode: '知识管理' matches even if indexed as '知识' + '管理'."""
        src = _make_source(db)
        db.add_chunk(src.id, "个人知识管理系统可以提高学习效率", 0)

        results = db.search_fts("知识管理")
        assert len(results) >= 1

    def test_fts_chinese_question_terms_do_not_block_keyword_match(self, db: Database):
        src = _make_source(db)
        db.add_chunk(src.id, "卵生动物通过产卵进行繁殖。", 0)

        results = db.search_fts("卵生动物有哪些？")

        assert [chunk.source_id for chunk, _ in results] == [src.id]

    def test_prepare_fts_query_marks_chinese_question_terms(self, db: Database):
        question = db.prepare_fts_query("卵生动物有哪些？")
        ordinary = db.prepare_fts_query("知识管理")

        assert question.stripped_cjk_question_terms is True
        assert "有哪些" not in question.query
        assert ordinary.stripped_cjk_question_terms is False

    def test_prepare_fts_query_strips_chinese_questions_for_trigram_fallback(
        self, db: Database
    ):
        db._fts_tokenizer = "trigram"  # type: ignore[attr-defined]

        question = db.prepare_fts_query("卵生动物有哪些？")

        assert question == FTSQuery(
            query="卵生动物",
            stripped_cjk_question_terms=True,
        )

    def test_cascade_delete_cleans_fts(self, db: Database):
        src = _make_source(db)
        db.add_chunk(src.id, "Unique searchable content xyzzy", 0)
        assert len(db.search_fts("xyzzy")) == 1

        db.delete_source(src.id)
        assert len(db.search_fts("xyzzy")) == 0

    def test_delete_chunks_by_source(self, db: Database):
        src = _make_source(db)
        db.add_chunk(src.id, "chunk one", 0)
        db.add_chunk(src.id, "chunk two", 1)
        deleted = db.delete_chunks_by_source(src.id)
        assert deleted == 2
        assert db.get_chunks_by_source(src.id) == []

    def test_delete_chunks_by_source_cleans_vec(self, db: Database):
        """delete_chunks_by_source must also clean vec_chunks."""
        src = _make_source(db)
        c = db.add_chunk(src.id, "vectorized chunk", 0)
        emb = _random_embedding()
        db.upsert_vec(c.id, emb)

        assert len(db.search_vec(emb, k=1)) == 1
        db.delete_chunks_by_source(src.id)
        assert len(db.search_vec(emb, k=1)) == 0

    def test_vec_upsert_and_search(self, db: Database):
        src = _make_source(db)
        c = db.add_chunk(src.id, "test", 0)
        emb = _random_embedding()
        db.upsert_vec(c.id, emb)

        results = db.search_vec(emb, k=1)
        assert len(results) == 1
        assert results[0][0] == c.id

    def test_vec_cleanup_on_source_delete(self, db: Database):
        src = _make_source(db)
        c = db.add_chunk(src.id, "test", 0)
        emb = _random_embedding()
        db.upsert_vec(c.id, emb)

        db.delete_source(src.id)
        # After source delete, vec_chunks should also be cleaned up
        results = db.search_vec(emb, k=1)
        assert len(results) == 0


# ── 5. TestWikiLinks ─────────────────────────────────────────────


class TestWikiLinks:
    def test_add_link_updates_indegree(self, db: Database):
        src = _make_source(db)
        target = _make_source(db)

        db.add_wiki_link(src.id, target.path, target.id)

        updated_target = db.get_source(target.id)
        assert updated_target is not None
        assert updated_target.indegree == 1

    def test_multiple_links_to_same_target(self, db: Database):
        s1 = _make_source(db)
        s2 = _make_source(db)
        target = _make_source(db)

        db.add_wiki_link(s1.id, target.path, target.id)
        db.add_wiki_link(s2.id, target.path, target.id)

        t = db.get_source(target.id)
        assert t is not None
        assert t.indegree == 2

    def test_delete_link_updates_indegree(self, db: Database):
        src = _make_source(db)
        target = _make_source(db)
        db.add_wiki_link(src.id, target.path, target.id)

        db.delete_links_by_source(src.id)
        t = db.get_source(target.id)
        assert t is not None
        assert t.indegree == 0

    def test_get_links_from_and_to(self, db: Database):
        src = _make_source(db)
        target = _make_source(db)
        db.add_wiki_link(src.id, target.path, target.id)

        outgoing = db.get_links_from(src.id)
        assert len(outgoing) == 1
        assert outgoing[0].target_note_id == target.id

        incoming = db.get_links_to(target.id)
        assert len(incoming) == 1
        assert incoming[0].source_note_id == src.id

    def test_duplicate_link_ignored(self, db: Database):
        src = _make_source(db)
        target = _make_source(db)

        link1 = db.add_wiki_link(src.id, target.path, target.id)
        link2 = db.add_wiki_link(src.id, target.path, target.id)

        assert link1 is not None
        assert link2 is None  # INSERT OR IGNORE
        assert len(db.get_links_from(src.id)) == 1


# ── 6. TestForwardRefs ───────────────────────────────────────────


class TestForwardRefs:
    def test_null_target_then_resolve(self, db: Database):
        """Link with NULL target_note_id → later resolve → indegree updates."""
        src = _make_source(db)

        # Link to note that doesn't exist yet
        link = db.add_wiki_link(src.id, "future-note", target_note_id=None)
        assert link is not None
        assert link.target_note_id is None

        # Later: the target note is created
        target = db.add_source(uid=_uid(), path="future-note")
        assert target.indegree == 0

        # Resolve forward refs
        resolved = db.resolve_forward_refs("future-note", target.id)
        assert resolved == 1

        # Indegree should be updated via UPDATE trigger
        t = db.get_source(target.id)
        assert t is not None
        assert t.indegree == 1

    def test_retarget_link_updates_both_indegrees(self, db: Database):
        """retargeting A→B must decrement A and increment B."""
        src = _make_source(db)
        target_a = _make_source(db)
        target_b = _make_source(db)

        # Link points to target_a
        link = db.add_wiki_link(src.id, "some-note", target_a.id)
        assert link is not None
        assert db.get_source(target_a.id).indegree == 1

        # Retarget to target_b via raw UPDATE
        db.conn.execute(
            "UPDATE wiki_links SET target_note_id = ? WHERE id = ?",
            (target_b.id, link.id),
        )
        db.conn.commit()

        assert db.get_source(target_a.id).indegree == 0
        assert db.get_source(target_b.id).indegree == 1

    def test_resolve_multiple_forward_refs(self, db: Database):
        s1 = _make_source(db)
        s2 = _make_source(db)

        db.add_wiki_link(s1.id, "future-note", target_note_id=None)
        db.add_wiki_link(s2.id, "future-note", target_note_id=None)

        target = db.add_source(uid=_uid(), path="future-note")
        resolved = db.resolve_forward_refs("future-note", target.id)
        assert resolved == 2

        t = db.get_source(target.id)
        assert t is not None
        assert t.indegree == 2


# ── 7. TestSuggestions ───────────────────────────────────────────


class TestSuggestions:
    def test_create_suggestion(self, db: Database):
        s1 = _make_source(db)
        s2 = _make_source(db)

        sug = db.add_suggestion(s1.id, s2.id, 0.85, "High similarity")
        assert isinstance(sug, Suggestion)
        assert sug.status == "pending"
        assert sug.score == 0.85
        assert sug.reason == "High similarity"

    def test_get_pending(self, db: Database):
        s1 = _make_source(db)
        s2 = _make_source(db)
        s3 = _make_source(db)

        db.add_suggestion(s1.id, s2.id, 0.9, "Very similar")
        db.add_suggestion(s1.id, s3.id, 0.7, "Somewhat similar")

        pending = db.get_pending_suggestions()
        assert len(pending) == 2
        # Ordered by score DESC
        assert pending[0].score >= pending[1].score

    def test_approve(self, db: Database):
        s1 = _make_source(db)
        s2 = _make_source(db)
        sug = db.add_suggestion(s1.id, s2.id, 0.85)

        approved = db.approve_suggestion(sug.id)
        assert approved is not None
        assert approved.status == "approved"
        assert approved.resolved_at is not None

        # No longer in pending
        assert len(db.get_pending_suggestions()) == 0

    def test_reject(self, db: Database):
        s1 = _make_source(db)
        s2 = _make_source(db)
        sug = db.add_suggestion(s1.id, s2.id, 0.85)

        rejected = db.reject_suggestion(sug.id)
        assert rejected is not None
        assert rejected.status == "rejected"
        assert rejected.resolved_at is not None


# ── 8. TestPragmas ───────────────────────────────────────────────


class TestPragmas:
    def test_wal_mode(self, db_file: Database):
        """WAL mode requires file-backed DB."""
        mode = db_file.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_busy_timeout(self, db: Database):
        timeout = db.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 5000

    def test_foreign_keys_on(self, db: Database):
        fk = db.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_wal_in_memory_fallback(self, db: Database):
        """In-memory DB can't use WAL, falls back to 'memory'."""
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "memory"


# ── v2: Tags CRUD ─────────────────────────────────────────────────


class TestTagCRUD:
    """Tests for source_tags table and tag methods."""

    def test_add_and_get_tags(self, db: Database):
        source = _make_source(db, path="notes/tagged.md")
        count = db.add_tags(source.id, ["ai", "ml", "deep-learning"])
        assert count == 3
        tags = db.get_tags(source.id)
        assert tags == ["ai", "deep-learning", "ml"]  # sorted

    def test_add_tags_ignores_duplicates(self, db: Database):
        source = _make_source(db, path="notes/dup-tags.md")
        db.add_tags(source.id, ["ai", "ml"])
        count = db.add_tags(source.id, ["ml", "new-tag"])
        assert count == 1  # only "new-tag" is new
        tags = db.get_tags(source.id)
        assert "new-tag" in tags

    def test_delete_tags_by_source(self, db: Database):
        source = _make_source(db, path="notes/del-tags.md")
        db.add_tags(source.id, ["a", "b", "c"])
        deleted = db.delete_tags_by_source(source.id)
        assert deleted == 3
        assert db.get_tags(source.id) == []

    def test_get_sources_by_tag(self, db: Database):
        s1 = _make_source(db, path="notes/s1.md")
        s2 = _make_source(db, path="notes/s2.md")
        s3 = _make_source(db, path="notes/s3.md")
        db.add_tags(s1.id, ["ai", "ml"])
        db.add_tags(s2.id, ["ai"])
        db.add_tags(s3.id, ["cooking"])
        ai_sources = db.get_sources_by_tag("ai")
        ai_ids = {s.id for s in ai_sources}
        assert s1.id in ai_ids
        assert s2.id in ai_ids
        assert s3.id not in ai_ids

    def test_tags_cascade_on_source_delete(self, db: Database):
        source = _make_source(db, path="notes/cascade-tags.md")
        db.add_tags(source.id, ["tag1", "tag2"])
        db.delete_source(source.id)
        tags = db.get_tags(source.id)
        assert tags == []

    def test_hierarchical_tags(self, db: Database):
        source = _make_source(db, path="notes/hier-tags.md")
        db.add_tags(source.id, ["ai-engineering/mcp", "ai-engineering/agents"])
        tags = db.get_tags(source.id)
        assert "ai-engineering/mcp" in tags


# ── v2: FTS Sources ───────────────────────────────────────────────


class TestFTSSources:
    """Tests for fts_sources search (title + aliases + headings)."""

    def test_search_by_title(self, db: Database):
        source = db.add_source(uid=_uid(), path="notes/ml.md", title="Machine Learning Basics")
        db.update_source(source.id, title="Machine Learning Basics")
        results = db.search_fts_sources("machine learning", limit=10)
        source_ids = [sid for sid, _ in results]
        assert source.id in source_ids

    def test_search_by_alias(self, db: Database):
        source = db.add_source(uid=_uid(), path="notes/ml-alias.md", title="ML Intro")
        db.update_source(source.id, aliases='["Machine Learning", "ML Basics"]')
        results = db.search_fts_sources("machine learning", limit=10)
        source_ids = [sid for sid, _ in results]
        assert source.id in source_ids

    def test_search_by_heading(self, db: Database):
        source = db.add_source(uid=_uid(), path="notes/workflow.md", title="Workflow")
        db.update_source(source.id, headings='["Capture inbox workflow", "Review queue"]')
        results = db.search_fts_sources("capture inbox", limit=10)
        source_ids = [sid for sid, _ in results]
        assert source.id in source_ids

    def test_search_by_chinese_question_terms(self, db: Database):
        source = db.add_source(
            uid=_uid(),
            path="notes/shanghai-theaters.md",
            title="上海剧院",
        )

        results = db.search_fts_sources("上海有哪些剧院？", limit=10)

        assert [source_id for source_id, _ in results] == [source.id]

    def test_search_returns_empty_on_no_match(self, db: Database):
        results = db.search_fts_sources("nonexistent_term_xyz", limit=10)
        assert results == []

    def test_search_handles_bad_query(self, db: Database):
        results = db.search_fts_sources("AND OR NOT", limit=10)
        assert isinstance(results, list)  # should not raise


# ── v2: get_source_by_stem ────────────────────────────────────────


class TestGetSourceByStem:
    """Tests for stem-based source lookup."""

    def test_finds_by_stem(self, db: Database):
        source = _make_source(db, path="notes/my-note.md")
        found = db.get_source_by_stem("my-note")
        assert found is not None
        assert found.id == source.id

    def test_finds_root_level(self, db: Database):
        source = _make_source(db, path="root-note.md")
        found = db.get_source_by_stem("root-note")
        assert found is not None
        assert found.id == source.id

    def test_returns_none_for_missing(self, db: Database):
        assert db.get_source_by_stem("no-such-note") is None


# ── v2: get_source_by_alias ──────────────────────────────────────


class TestGetSourceByAlias:
    """Tests for alias-based source lookup."""

    def test_finds_by_alias(self, db: Database):
        source = db.add_source(uid=_uid(), path="notes/aliased.md")
        db.update_source(source.id, aliases='["ML", "Machine Learning"]')
        found = db.get_source_by_alias("ML")
        assert found is not None
        assert found.id == source.id

    def test_returns_none_for_missing(self, db: Database):
        assert db.get_source_by_alias("nonexistent-alias") is None


# ── v2: Migration ────────────────────────────────────────────────


class TestMigrationV1ToV2:
    """Test that v1 databases migrate to v2 correctly."""

    def test_migrate_preserves_data(self):
        """Create a v1 database, add data, then migrate to v2."""
        db = Database(":memory:")

        # Manually create v1 schema (without aliases column, without source_tags/fts_sources)
        db._conn.executescript(f"""
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL UNIQUE,
                title TEXT,
                content_hash TEXT,
                status TEXT DEFAULT 'unprocessed',
                indegree INTEGER DEFAULT 0,
                fs_modified_at TEXT,
                indexed_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                token_count INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE wiki_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_note_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                target_note_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
                target_path TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(source_note_id, target_path)
            );
            CREATE TABLE suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_note_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                target_note_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                score REAL NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                resolved_at TEXT
            );
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[768] distance_metric=cosine
            );
            CREATE VIRTUAL TABLE fts_chunks USING fts5(
                content, content=chunks, content_rowid=id, tokenize='{db._fts_tokenizer}'
            );
            PRAGMA user_version = 1;
        """)

        # Add v1 data
        db._conn.execute(
            "INSERT INTO sources (uid, path, title, status) VALUES (?, ?, ?, ?)",
            (_uid(), "notes/test.md", "Test Note", "indexed"),
        )
        db._conn.commit()

        # Now migrate to v2
        db._migrate_v1_to_v2()

        # Verify migration
        version = db._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 2

        # Original data preserved, aliases column added with default
        row = db._conn.execute("SELECT * FROM sources WHERE path = 'notes/test.md'").fetchone()
        assert row is not None
        assert row["aliases"] == "[]"

        # New tables exist
        tables = {r[0] for r in db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "source_tags" in tables

        # fts_sources was backfilled
        fts_rows = db._conn.execute(
            "SELECT rowid FROM fts_sources WHERE fts_sources MATCH 'test'"
        ).fetchall()
        assert len(fts_rows) >= 1

        db.close()


class TestMigrationV2ToCurrent:
    """Test that v2 databases migrate through headings to the current schema."""

    def test_migrate_adds_headings_and_rebuilds_source_fts(self):
        db = Database(":memory:")

        db._conn.executescript(f"""
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL UNIQUE,
                title TEXT,
                content_hash TEXT,
                status TEXT DEFAULT 'unprocessed',
                indegree INTEGER DEFAULT 0,
                aliases TEXT DEFAULT '[]',
                fs_modified_at TEXT,
                indexed_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE VIRTUAL TABLE fts_sources USING fts5(
                title,
                aliases,
                content=sources,
                content_rowid=id,
                tokenize='{db._fts_tokenizer}'
            );
            PRAGMA user_version = 2;
        """)
        db._conn.execute(
            """INSERT INTO sources (uid, path, title, aliases, status)
               VALUES (?, ?, ?, ?, ?)""",
            (_uid(), "notes/workflow.md", "Workflow", '["Flow"]', "indexed"),
        )
        db._conn.execute("""
            INSERT INTO fts_sources(rowid, title, aliases)
            SELECT id, COALESCE(title, ''), COALESCE(aliases, '[]') FROM sources
        """)
        db._conn.commit()

        db.init_schema()

        version = db._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == Database.SCHEMA_VERSION

        source = db.get_source_by_path("notes/workflow.md")
        assert source is not None
        assert source.headings == "[]"
        assert source.status == "unprocessed"

        title_hits = db.search_fts_sources("workflow", limit=10)
        assert source.id in [sid for sid, _ in title_hits]

        db.update_source(source.id, headings='["Capture inbox workflow"]')
        heading_hits = db.search_fts_sources("capture inbox", limit=10)
        assert source.id in [sid for sid, _ in heading_hits]
        assert db.get_index_metadata() == {}

        db.close()


class TestIndexMetadata:
    """Index configuration metadata and rebuild reset helpers."""

    def test_set_and_get_index_metadata(self, db: Database):
        db.set_index_metadata({
            "embedder_model": "model-a",
            "embedding_dim": "768",
        })
        db.set_index_metadata({"embedder_model": "model-b"})

        metadata = db.get_index_metadata()

        assert metadata["embedder_model"] == "model-b"
        assert metadata["embedding_dim"] == "768"

    def test_recreate_vec_table_changes_dimension(self, db: Database):
        assert db.get_vector_dimension() == 768

        db.recreate_vec_table(1024)

        assert db.get_vector_dimension() == 1024

    def test_reset_index_contents_for_rebuild_preserves_sources(self, db: Database):
        source = _make_source(db, path="notes/rebuild.md", title="Rebuild")
        db.update_source(
            source.id,
            status="indexed",
            content_hash="old-hash",
            indexed_at="2026-01-01 00:00:00",
        )
        chunk = db.add_chunk(source.id, "old chunk", 0)
        db.upsert_vec(chunk.id, _random_embedding())
        db.add_tags(source.id, ["old-tag"])

        db.reset_index_contents_for_rebuild()

        source = db.get_source_by_path("notes/rebuild.md")
        assert source is not None
        assert source.status == "unprocessed"
        assert source.content_hash is None
        assert source.indexed_at is None
        assert source.indegree == 0
        assert db.get_chunks_by_source(source.id) == []
        assert db.get_tags(source.id) == []


class TestMigrationV3ToV4:
    """v3 databases gain index_metadata without touching existing rows."""

    def test_migrate_adds_index_metadata_table(self):
        db = Database(":memory:")
        db._conn.executescript("""
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL UNIQUE,
                title TEXT,
                content_hash TEXT,
                status TEXT DEFAULT 'unprocessed',
                indegree INTEGER DEFAULT 0,
                aliases TEXT DEFAULT '[]',
                headings TEXT DEFAULT '[]',
                fs_modified_at TEXT,
                indexed_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            PRAGMA user_version = 3;
        """)

        db.init_schema()

        version = db._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == Database.SCHEMA_VERSION
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "index_metadata" in tables
        assert db.get_index_metadata() == {}
        db.close()

"""Tests for sophia.ingest — integration tests with DB + embedder."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from sophia.db import Database
from sophia.embedder import Embedder
from sophia.ingest import ingest_file, ingest_vault


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
    return tmp_path / "vault"


def _write_md(vault: Path, rel_path: str, content: str) -> Path:
    """Helper to write a markdown file in the vault."""
    p = vault / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestIngestFile:
    def test_new_file_indexed(self, db: Database, embedder: Embedder, vault: Path):
        path = _write_md(vault, "note.md", "# My Note\n\nSome content here.")
        vault.mkdir(parents=True, exist_ok=True)

        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        assert result.status == "indexed"
        assert result.title == "My Note"

        chunks = db.get_chunks_by_source(result.id)
        assert len(chunks) >= 1

    def test_unchanged_file_skipped(self, db: Database, embedder: Embedder, vault: Path):
        content = "# Test\n\nContent here."
        path = _write_md(vault, "test.md", content)

        first = ingest_file(db, path, vault, embedder)
        second = ingest_file(db, path, vault, embedder)

        assert first is not None
        assert second is not None
        assert first.id == second.id
        # content_hash should match — was skipped
        assert first.content_hash == second.content_hash

    def test_changed_file_reindexed(self, db: Database, embedder: Embedder, vault: Path):
        path = _write_md(vault, "change.md", "# Original\n\nOriginal content.")
        first = ingest_file(db, path, vault, embedder)
        assert first is not None
        old_chunks = db.get_chunks_by_source(first.id)

        # Modify file
        path.write_text("# Updated\n\nNew content here.", encoding="utf-8")
        second = ingest_file(db, path, vault, embedder)
        assert second is not None
        new_chunks = db.get_chunks_by_source(second.id)

        # Chunks should be refreshed
        assert any("New content" in c.content for c in new_chunks)

    def test_non_md_skipped(self, db: Database, embedder: Embedder, vault: Path):
        path = vault / "readme.txt"
        vault.mkdir(parents=True, exist_ok=True)
        path.write_text("Not markdown.", encoding="utf-8")

        result = ingest_file(db, path, vault, embedder)
        assert result is None

    def test_binary_file_skipped(self, db: Database, embedder: Embedder, vault: Path):
        path = vault / "binary.md"
        vault.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x80\x81\x82\x83" * 100)

        result = ingest_file(db, path, vault, embedder)
        assert result is None

    def test_title_from_heading(self, db: Database, embedder: Embedder, vault: Path):
        path = _write_md(vault, "titled.md", "# Custom Title\n\nBody text.")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        assert result.title == "Custom Title"

    def test_title_from_filename(self, db: Database, embedder: Embedder, vault: Path):
        path = _write_md(vault, "no-heading.md", "Just body text, no heading.")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        assert result.title == "no-heading"

    def test_wiki_links_stored(self, db: Database, embedder: Embedder, vault: Path):
        # Create target first
        _write_md(vault, "target.md", "# Target\n\nTarget content.")
        ingest_file(db, vault / "target.md", vault, embedder)

        # Now create source with link to target
        path = _write_md(vault, "source.md", "# Source\n\nSee [[target]] for details.")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None

        links = db.get_links_from(result.id)
        assert len(links) == 1
        assert links[0].target_path == "target"
        assert links[0].target_note_id is not None  # resolved

    def test_forward_ref_resolved(self, db: Database, embedder: Embedder, vault: Path):
        """Link to non-existent file → NULL target, then resolved when target ingested."""
        # Ingest source first (target doesn't exist yet)
        path = _write_md(vault, "linker.md", "# Linker\n\nSee [[future-note]].")
        ingest_file(db, path, vault, embedder)

        linker = db.get_source_by_path("linker.md")
        links_before = db.get_links_from(linker.id)
        assert len(links_before) == 1
        assert links_before[0].target_note_id is None  # forward ref

        # Now ingest the target
        target_path = _write_md(vault, "future-note.md", "# Future\n\nNow I exist.")
        ingest_file(db, target_path, vault, embedder)

        # Forward ref should be resolved
        links_after = db.get_links_from(linker.id)
        assert len(links_after) == 1
        assert links_after[0].target_note_id is not None

    def test_chinese_file(self, db: Database, embedder: Embedder, vault: Path):
        content = "# 知识管理\n\n这是关于个人知识管理的笔记。\n\n## 方法\n\n使用卡片盒笔记法。"
        path = _write_md(vault, "知识.md", content)
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        assert result.status == "indexed"

        chunks = db.get_chunks_by_source(result.id)
        assert len(chunks) >= 1

    def test_empty_file(self, db: Database, embedder: Embedder, vault: Path):
        path = _write_md(vault, "empty.md", "")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        chunks = db.get_chunks_by_source(result.id)
        assert len(chunks) == 0

    def test_chunks_have_vectors(self, db: Database, embedder: Embedder, vault: Path):
        """Every chunk should have a corresponding vector in vec_chunks."""
        path = _write_md(vault, "vec-test.md", "# Test\n\nContent for vector test.")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None

        chunks = db.get_chunks_by_source(result.id)
        for chunk in chunks:
            # Search by chunk's own embedding — should find itself
            row = db.conn.execute(
                "SELECT chunk_id FROM vec_chunks WHERE chunk_id = ?",
                (chunk.id,),
            ).fetchone()
            assert row is not None


class TestIngestVault:
    def test_stats_correct(self, db: Database, embedder: Embedder, vault: Path):
        _write_md(vault, "a.md", "# Note A\n\nContent A.")
        _write_md(vault, "b.md", "# Note B\n\nContent B.")
        _write_md(vault, "sub/c.md", "# Note C\n\nContent C.")

        stats = ingest_vault(db, vault, embedder)
        assert stats["ingested"] == 3
        assert stats["unchanged"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

    def test_second_run_unchanged(self, db: Database, embedder: Embedder, vault: Path):
        _write_md(vault, "a.md", "# Note A\n\nContent A.")
        _write_md(vault, "b.md", "# Note B\n\nContent B.")

        ingest_vault(db, vault, embedder)
        stats = ingest_vault(db, vault, embedder)
        assert stats["ingested"] == 0
        assert stats["unchanged"] == 2

    def test_mixed_operations(self, db: Database, embedder: Embedder, vault: Path):
        _write_md(vault, "keep.md", "# Keep\n\nUnchanged.")
        _write_md(vault, "modify.md", "# Modify\n\nOriginal.")

        ingest_vault(db, vault, embedder)

        # Modify one file, add a new one
        _write_md(vault, "modify.md", "# Modify\n\nUpdated content.")
        _write_md(vault, "new.md", "# New\n\nBrand new note.")

        stats = ingest_vault(db, vault, embedder)
        assert stats["unchanged"] == 1  # keep.md
        assert stats["ingested"] == 2   # modify.md + new.md


    def test_skips_hidden_dirs(self, db: Database, embedder: Embedder, vault: Path):
        """ingest_vault should skip files in hidden directories."""
        _write_md(vault, "visible.md", "# Visible\n\nContent.")
        _write_md(vault, ".dev/research/design.md", "# Design\n\nHidden.")
        _write_md(vault, ".claude/memory.md", "# Memory\n\nHidden.")

        stats = ingest_vault(db, vault, embedder)
        assert stats["ingested"] == 1  # only visible.md

    def test_skips_todo_dir(self, db: Database, embedder: Embedder, vault: Path):
        """ingest_vault should skip files in the todo directory."""
        _write_md(vault, "note.md", "# Note\n\nContent.")
        _write_md(vault, "todo/todo.md", "# Todo\n\nTasks.")

        stats = ingest_vault(db, vault, embedder)
        assert stats["ingested"] == 1  # only note.md


class TestTimestamp:
    def test_indexed_at_is_real_timestamp(self, db: Database, embedder: Embedder, vault: Path):
        """indexed_at should be a real ISO timestamp, not a literal SQL expression."""
        path = _write_md(vault, "ts.md", "# Timestamp\n\nContent.")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        assert result.indexed_at is not None
        # Must look like YYYY-MM-DD HH:MM:SS, not "datetime('now')"
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result.indexed_at), (
            f"indexed_at should be ISO timestamp, got: {result.indexed_at!r}"
        )


class TestTransactionSafety:
    def test_rollback_on_embedding_failure(self, db: Database, embedder: Embedder, vault: Path):
        """If embedding fails mid-ingest, old chunks should be preserved (rollback)."""
        path = _write_md(vault, "rollback.md", "# Test\n\nOriginal content here.")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        original_chunks = db.get_chunks_by_source(result.id)
        assert len(original_chunks) >= 1

        # Modify file, then make embed_documents raise
        path.write_text("# Test\n\nNew content that will fail.", encoding="utf-8")
        with patch.object(embedder, "embed_documents", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                ingest_file(db, path, vault, embedder)

        # Original chunks should still be intact (transaction rolled back)
        surviving_chunks = db.get_chunks_by_source(result.id)
        assert len(surviving_chunks) == len(original_chunks)
        source = db.get_source(result.id)
        assert source is not None
        assert source.status == "indexed"


class TestLinkResolution:
    def test_subdirectory_link_by_stem(self, db: Database, embedder: Embedder, vault: Path):
        """[[note]] resolves to sub/note.md via stem matching."""
        _write_md(vault, "sub/note.md", "# Note\n\nIn subdirectory.")
        ingest_file(db, vault / "sub" / "note.md", vault, embedder)

        path = _write_md(vault, "linker.md", "# Linker\n\nSee [[note]].")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None

        links = db.get_links_from(result.id)
        assert len(links) == 1
        assert links[0].target_note_id is not None  # resolved via stem

    def test_path_style_link(self, db: Database, embedder: Embedder, vault: Path):
        """[[sub/target]] resolves to sub/target.md via path matching."""
        _write_md(vault, "sub/target.md", "# Target\n\nContent.")
        ingest_file(db, vault / "sub" / "target.md", vault, embedder)

        path = _write_md(vault, "source.md", "# Source\n\nSee [[sub/target]].")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None

        links = db.get_links_from(result.id)
        assert len(links) == 1
        assert links[0].target_note_id is not None

    def test_forward_ref_resolved_by_path(self, db: Database, embedder: Embedder, vault: Path):
        """Forward ref [[sub/target]] is resolved when sub/target.md is ingested."""
        path = _write_md(vault, "early.md", "# Early\n\nSee [[sub/note]].")
        ingest_file(db, path, vault, embedder)

        early = db.get_source_by_path("early.md")
        links_before = db.get_links_from(early.id)
        assert links_before[0].target_note_id is None  # unresolved

        # Ingest the target
        _write_md(vault, "sub/note.md", "# Note\n\nHere now.")
        ingest_file(db, vault / "sub" / "note.md", vault, embedder)

        links_after = db.get_links_from(early.id)
        assert links_after[0].target_note_id is not None  # resolved

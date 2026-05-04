"""Tests for seeklink.ingest — integration tests with DB + embedder."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from seeklink.db import CapabilityError, Database
from seeklink.embedder import Embedder
from seeklink.index_config import expected_index_metadata
from seeklink.ingest import (
    _EMBED_BATCH_SIZE,
    _parse_frontmatter,
    ingest_file,
    ingest_vault,
)


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


class FakeBatchEmbedder:
    MODEL_NAME = "fake-batch-embedder"
    EMBEDDING_DIM = 768

    def __init__(self, *, fail_on: str | None = None):
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        self.calls.append(list(texts))
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise RuntimeError("fake embed failure")
        return [b"\0" * (768 * 4) for _ in texts]


class Fake384Embedder:
    MODEL_NAME = "fake-384-embedder"
    EMBEDDING_DIM = 384

    def embed_documents(self, texts: list[str]) -> list[bytes]:
        return [b"\0" * (384 * 4) for _ in texts]


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
        assert db.get_index_metadata() == expected_index_metadata(embedder.MODEL_NAME)

    def test_single_file_rejects_mismatched_existing_index(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        path = _write_md(vault, "note.md", "# My Note\n\nSome content here.")
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        db.set_index_metadata({
            **expected_index_metadata(embedder.MODEL_NAME),
            "embedder_model": "different-model",
        })

        with pytest.raises(CapabilityError, match="full `seeklink index`"):
            ingest_file(db, path, vault, embedder)

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

    def test_title_ignores_fenced_code_heading(
        self, db: Database, embedder: Embedder, vault: Path
    ):
        path = _write_md(
            vault,
            "code-first.md",
            "```markdown\n# Fake Title\n```\n\n# Real Title\n\nBody text.",
        )
        result = ingest_file(db, path, vault, embedder)
        assert result is not None
        assert result.title == "Real Title"

    def test_headings_stored_for_source_search(self, db: Database, vault: Path):
        path = _write_md(
            vault,
            "workflow.md",
            "# Workflow\n\n## Capture inbox workflow\n\nBody text.",
        )

        result = ingest_file(db, path, vault, FakeBatchEmbedder())  # type: ignore[arg-type]

        assert result is not None
        assert json.loads(result.headings) == ["Capture inbox workflow"]
        source_ids = [sid for sid, _ in db.search_fts_sources("capture inbox")]
        assert result.id in source_ids

    def test_headings_ignore_fenced_code(self, db: Database, vault: Path):
        path = _write_md(
            vault,
            "workflow.md",
            "# Workflow\n\n```markdown\n## Fake heading\n```\n\n## Real heading",
        )

        result = ingest_file(db, path, vault, FakeBatchEmbedder())  # type: ignore[arg-type]

        assert result is not None
        assert json.loads(result.headings) == ["Real heading"]

    def test_unprocessed_same_hash_reindexed_after_schema_migration(
        self, db: Database, vault: Path
    ):
        content = "# Workflow\n\n## Capture inbox workflow\n\nBody text."
        path = _write_md(vault, "workflow.md", content)
        db.add_source(
            uid="existing",
            path="workflow.md",
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            status="unprocessed",
        )

        result = ingest_file(db, path, vault, FakeBatchEmbedder())  # type: ignore[arg-type]

        assert result is not None
        assert result.status == "indexed"
        assert json.loads(result.headings) == ["Capture inbox workflow"]
        assert db.get_chunks_by_source(result.id)

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

    def test_batches_embeddings_across_files(self, db: Database, vault: Path):
        fake = FakeBatchEmbedder()
        for i in range(40):
            _write_md(vault, f"note-{i:02d}.md", f"# Note {i}\n\nBatch content {i}.")

        stats = ingest_vault(db, vault, fake)  # type: ignore[arg-type]

        assert stats["ingested"] == 40
        assert stats["errors"] == 0
        call_sizes = [len(call) for call in fake.calls]
        assert len(call_sizes) == (40 + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE
        assert max(call_sizes) <= _EMBED_BATCH_SIZE
        assert sum(call_sizes) == 40

    def test_vault_progress_callback_reports_index_phases(
        self,
        db: Database,
        vault: Path,
    ):
        fake = FakeBatchEmbedder()
        events: list[tuple[str, dict]] = []
        for i in range(3):
            _write_md(vault, f"note-{i}.md", f"# Note {i}\n\nProgress content {i}.")

        stats = ingest_vault(
            db,
            vault,
            fake,  # type: ignore[arg-type]
            progress=lambda event, payload: events.append((event, payload)),
        )

        assert stats["ingested"] == 3
        event_names = [event for event, _payload in events]
        assert event_names[0] == "scan_start"
        assert "scan_done" in event_names
        assert "prepare_done" in event_names
        assert "embed_start" in event_names
        assert "embed_progress" in event_names
        assert "embed_done" in event_names
        assert "write_start" in event_names
        assert "write_progress" in event_names
        assert event_names[-1] == "done"
        prepare_done = next(
            payload for event, payload in events if event == "prepare_done"
        )
        assert prepare_done["files_to_index"] == 3
        assert prepare_done["chunks_to_embed"] == 3

    def test_batch_vault_resolves_forward_refs(self, db: Database, vault: Path):
        fake = FakeBatchEmbedder()
        _write_md(vault, "a.md", "# A\n\nSee [[b]].")
        _write_md(vault, "b.md", "# B\n\nTarget note.")

        stats = ingest_vault(db, vault, fake)  # type: ignore[arg-type]

        assert stats["ingested"] == 2
        source = db.get_source_by_path("a.md")
        assert source is not None
        links = db.get_links_from(source.id)
        assert len(links) == 1
        assert links[0].target_note_id is not None

    def test_full_vault_rebuilds_after_index_config_change(
        self, db: Database, vault: Path
    ):
        fake = FakeBatchEmbedder()
        _write_md(vault, "note.md", "# Note\n\nOriginal content.")
        first = ingest_vault(db, vault, fake)  # type: ignore[arg-type]
        assert first["ingested"] == 1

        source = db.get_source_by_path("note.md")
        assert source is not None
        old_chunk_count = len(db.get_chunks_by_source(source.id))
        db.set_index_metadata({
            **expected_index_metadata(fake.MODEL_NAME),
            "embedder_model": "old-model",
        })

        second = ingest_vault(db, vault, fake)  # type: ignore[arg-type]

        assert second["ingested"] == 1
        source = db.get_source_by_path("note.md")
        assert source is not None
        assert source.status == "indexed"
        assert len(db.get_chunks_by_source(source.id)) == old_chunk_count
        assert db.get_index_metadata() == expected_index_metadata(fake.MODEL_NAME)

    def test_batch_embedding_failure_isolated_to_bad_file(
        self, db: Database, vault: Path
    ):
        fake = FakeBatchEmbedder(fail_on="FAIL")
        _write_md(vault, "bad.md", "# Bad\n\nThis chunk will FAIL.")
        _write_md(vault, "good.md", "# Good\n\nThis chunk should index.")

        stats = ingest_vault(db, vault, fake)  # type: ignore[arg-type]

        assert stats["ingested"] == 1
        assert stats["errors"] == 1
        assert db.get_source_by_path("good.md") is not None
        assert db.get_source_by_path("bad.md") is None

    def test_full_vault_can_recreate_vec_table_for_embedder_dimension(
        self, db: Database, vault: Path
    ):
        fake = Fake384Embedder()
        _write_md(vault, "small.md", "# Small\n\nDimension-specific index.")

        stats = ingest_vault(db, vault, fake)  # type: ignore[arg-type]

        assert stats["ingested"] == 1
        assert db.get_vector_dimension() == 384
        assert db.get_index_metadata() == expected_index_metadata(
            fake.MODEL_NAME,
            fake.EMBEDDING_DIM,
        )


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


# ── v2: Frontmatter parsing ─────────────────────────────────────


class TestParseFrontmatter:
    """Tests for _parse_frontmatter helper."""

    def test_no_frontmatter(self):
        tags, aliases, body = _parse_frontmatter("# Title\n\nContent here.")
        assert tags == []
        assert aliases == []
        assert body == "# Title\n\nContent here."

    def test_inline_tags(self):
        content = "---\ntags: [ai, ml, deep-learning]\n---\n# Title\n\nBody."
        tags, aliases, body = _parse_frontmatter(content)
        assert tags == ["ai", "ml", "deep-learning"]
        assert "# Title" in body
        assert "tags:" not in body

    def test_block_list_tags(self):
        content = "---\ntags:\n  - ai\n  - ml\n  - deep-learning\n---\n# Title\n"
        tags, aliases, body = _parse_frontmatter(content)
        assert tags == ["ai", "ml", "deep-learning"]

    def test_inline_aliases(self):
        content = '---\naliases: [ML, "Machine Learning"]\n---\n# Title\n'
        tags, aliases, body = _parse_frontmatter(content)
        assert aliases == ["ML", "Machine Learning"]

    def test_block_list_aliases(self):
        content = "---\naliases:\n  - ML\n  - Machine Learning\n---\n# Title\n"
        tags, aliases, body = _parse_frontmatter(content)
        assert aliases == ["ML", "Machine Learning"]

    def test_mixed_formats(self):
        content = "---\ntags: [ai, ml]\naliases:\n  - Alias One\n  - Alias Two\n---\nBody."
        tags, aliases, body = _parse_frontmatter(content)
        assert tags == ["ai", "ml"]
        assert aliases == ["Alias One", "Alias Two"]
        assert body == "Body."

    def test_chinese_tags(self):
        content = "---\ntags: [知识管理, 学习方法]\n---\n# 标题\n"
        tags, aliases, body = _parse_frontmatter(content)
        assert tags == ["知识管理", "学习方法"]

    def test_hierarchical_tags(self):
        content = "---\ntags: [ai-engineering/mcp, ai-engineering/agents]\n---\n# Title\n"
        tags, aliases, body = _parse_frontmatter(content)
        assert "ai-engineering/mcp" in tags

    def test_malformed_frontmatter(self):
        content = "---\nthis is not valid yaml\n---\n# Title\n"
        tags, aliases, body = _parse_frontmatter(content)
        assert tags == []
        assert aliases == []
        assert "# Title" in body

    def test_empty_frontmatter(self):
        content = "---\n---\n# Title\n"
        tags, aliases, body = _parse_frontmatter(content)
        assert tags == []
        assert aliases == []
        assert body == "# Title\n"

    def test_other_fields_ignored(self):
        content = "---\ncreated: 2026-01-01\ntags: [ai]\ntype: literature\n---\nBody."
        tags, aliases, body = _parse_frontmatter(content)
        assert tags == ["ai"]
        assert aliases == []

    def test_quoted_values(self):
        content = "---\ntags: ['tag one', \"tag two\"]\n---\nBody."
        tags, aliases, body = _parse_frontmatter(content)
        assert "tag one" in tags
        assert "tag two" in tags


# ── v2: Frontmatter stripping in ingest ──────────────────────────


class TestFrontmatterIngest:
    """Test that frontmatter is stripped before chunking/embedding."""

    @pytest.fixture
    def vault(self, tmp_path):
        return tmp_path

    def test_frontmatter_not_in_chunks(self, db: Database, embedder: Embedder, vault: Path):
        content = "---\ntags: [ai, ml]\naliases: [ML Intro]\n---\n# Machine Learning\n\nML uses algorithms."
        path = vault / "fm-test.md"
        path.write_text(content, encoding="utf-8")
        source = ingest_file(db, path, vault, embedder)
        assert source is not None

        chunks = db.get_chunks_by_source(source.id)
        for chunk in chunks:
            assert "tags:" not in chunk.content
            assert "aliases:" not in chunk.content

    def test_tags_stored_in_db(self, db: Database, embedder: Embedder, vault: Path):
        content = "---\ntags: [ai, ml]\n---\n# Test\n\nContent."
        path = vault / "tags-test.md"
        path.write_text(content, encoding="utf-8")
        source = ingest_file(db, path, vault, embedder)
        assert source is not None

        tags = db.get_tags(source.id)
        assert "ai" in tags
        assert "ml" in tags

    def test_aliases_stored_in_source(self, db: Database, embedder: Embedder, vault: Path):
        content = '---\naliases: [ML, "Machine Learning"]\n---\n# ML Basics\n\nContent.'
        path = vault / "alias-test.md"
        path.write_text(content, encoding="utf-8")
        source = ingest_file(db, path, vault, embedder)
        assert source is not None
        assert "ML" in source.aliases
        assert "Machine Learning" in source.aliases

    def test_reindex_clears_old_tags(self, db: Database, embedder: Embedder, vault: Path):
        path = vault / "reindex-tags.md"
        path.write_text("---\ntags: [old-tag]\n---\n# Test\n\nV1.", encoding="utf-8")
        ingest_file(db, path, vault, embedder)

        # Modify: change tags
        path.write_text("---\ntags: [new-tag]\n---\n# Test\n\nV2.", encoding="utf-8")
        source = ingest_file(db, path, vault, embedder)
        assert source is not None

        tags = db.get_tags(source.id)
        assert "new-tag" in tags
        assert "old-tag" not in tags

    def test_alias_link_resolution(self, db: Database, embedder: Embedder, vault: Path):
        """[[ML]] resolves to a note with alias 'ML'."""
        target = vault / "ml-basics.md"
        target.write_text('---\naliases: [ML]\n---\n# ML Basics\n\nContent.', encoding="utf-8")
        ingest_file(db, target, vault, embedder)

        linker = vault / "linker.md"
        linker.write_text("# Linker\n\nSee [[ML]] for details.", encoding="utf-8")
        ingest_file(db, linker, vault, embedder)

        linker_source = db.get_source_by_path("linker.md")
        links = db.get_links_from(linker_source.id)
        assert len(links) >= 1
        # The [[ML]] link should resolve to ml-basics.md
        ml_source = db.get_source_by_path("ml-basics.md")
        assert any(lk.target_note_id == ml_source.id for lk in links)

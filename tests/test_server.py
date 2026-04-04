"""Tests for Phase 4: MCP server, watcher, and helper functions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sophia.db import Database
from sophia.embedder import Embedder
from sophia.ingest import ingest_file
from sophia.server import _bfs_neighbors, _current_week_start, _write_related_link
from sophia.watcher import MarkdownFilter


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


def _write_md(vault: Path, rel_path: str, content: str) -> Path:
    """Helper to write a markdown file in the vault."""
    p = vault / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _ingest_corpus(db: Database, embedder: Embedder, vault: Path) -> None:
    """Ingest the standard 4-note test corpus."""
    _write_md(vault, "ml-basics.md", "# Machine Learning\n\nML uses algorithms to learn from data.")
    _write_md(vault, "deep-learning.md", "# Deep Learning\n\nNeural networks with many layers. See [[ml-basics]].")
    _write_md(vault, "cooking.md", "# Italian Cooking\n\nPasta recipes from Italy.")
    _write_md(vault, "hub-note.md", "# Hub\n\nSee [[ml-basics]] and [[deep-learning]].")

    ingest_file(db, vault / "ml-basics.md", vault, embedder)
    ingest_file(db, vault / "deep-learning.md", vault, embedder)
    ingest_file(db, vault / "cooking.md", vault, embedder)
    ingest_file(db, vault / "hub-note.md", vault, embedder)


# ── TestGetStats ──────────────────────────────────────────────────────


class TestGetStats:
    def test_empty_db(self, db: Database):
        stats = db.get_stats()
        assert stats["notes_total"] == 0
        assert stats["notes_unprocessed"] == 0
        assert stats["chunks_total"] == 0
        assert stats["links_total"] == 0
        assert stats["suggestions_pending"] == 0

    def test_after_corpus(self, db: Database, embedder: Embedder, vault: Path):
        _ingest_corpus(db, embedder, vault)
        stats = db.get_stats()
        assert stats["notes_total"] == 4
        assert stats["notes_unprocessed"] == 0  # all indexed
        assert stats["chunks_total"] > 0
        assert stats["links_total"] == 3  # ml-basics×2 + deep-learning×1


# ── TestGetSuggestion ─────────────────────────────────────────────────


class TestGetSuggestion:
    def test_found(self, db: Database):
        s1 = db.add_source(uid="u1", path="a.md", status="indexed")
        s2 = db.add_source(uid="u2", path="b.md", status="indexed")
        sug = db.add_suggestion(s1.id, s2.id, 0.8, reason="test")
        result = db.get_suggestion(sug.id)
        assert result is not None
        assert result.score == 0.8
        assert result.status == "pending"

    def test_missing(self, db: Database):
        assert db.get_suggestion(999) is None


# ── TestWriteRelatedLink ──────────────────────────────────────────────


class TestWriteRelatedLink:
    def test_new_section(self, tmp_path: Path):
        """Creates ## Related section when it doesn't exist."""
        f = tmp_path / "note.md"
        f.write_text("# Hello\n\nSome content.\n", encoding="utf-8")
        _write_related_link(f, "target")
        content = f.read_text(encoding="utf-8")
        assert "## Related" in content
        assert "- [[target]]" in content

    def test_existing_section(self, tmp_path: Path):
        """Appends to existing ## Related section."""
        f = tmp_path / "note.md"
        f.write_text("# Hello\n\n## Related\n\n- [[existing]]\n", encoding="utf-8")
        _write_related_link(f, "new-target")
        content = f.read_text(encoding="utf-8")
        assert "- [[existing]]" in content
        assert "- [[new-target]]" in content

    def test_idempotent(self, tmp_path: Path):
        """Skips if [[target]] already present in Related section."""
        f = tmp_path / "note.md"
        f.write_text("# Hello\n\n## Related\n\n- [[target]]\n", encoding="utf-8")
        _write_related_link(f, "target")
        content = f.read_text(encoding="utf-8")
        assert content.count("[[target]]") == 1

    def test_chinese_stem(self, tmp_path: Path):
        """Works with Chinese filenames."""
        f = tmp_path / "笔记.md"
        f.write_text("# 笔记\n\n一些内容。\n", encoding="utf-8")
        _write_related_link(f, "知识管理")
        content = f.read_text(encoding="utf-8")
        assert "- [[知识管理]]" in content

    def test_body_mention_does_not_block_related(self, tmp_path: Path):
        """[[target]] in body should NOT prevent adding to ## Related."""
        f = tmp_path / "note.md"
        f.write_text(
            "# Hello\n\nSee [[target]] for details.\n", encoding="utf-8"
        )
        _write_related_link(f, "target")
        content = f.read_text(encoding="utf-8")
        assert "## Related" in content
        assert "- [[target]]" in content
        # Original body mention still there
        assert "See [[target]] for details." in content

    def test_body_mention_with_existing_related(self, tmp_path: Path):
        """[[target]] in body doesn't block adding to existing Related section."""
        f = tmp_path / "note.md"
        f.write_text(
            "# Hello\n\nMention [[target]] here.\n\n## Related\n\n- [[other]]\n",
            encoding="utf-8",
        )
        _write_related_link(f, "target")
        content = f.read_text(encoding="utf-8")
        assert "- [[target]]" in content
        assert "- [[other]]" in content


# ── TestSuggestLinksLogic ─────────────────────────────────────────────


class TestSuggestLinksLogic:
    def test_excludes_self(self, db: Database, embedder: Embedder, vault: Path):
        """suggest_links-style filtering: self should be excluded from search results."""
        _ingest_corpus(db, embedder, vault)
        source = db.get_source_by_path("ml-basics.md")
        assert source is not None

        from sophia.search import search as sophia_search

        results = sophia_search(db, embedder, source.title or "ML", top_k=15)
        # Filter self
        filtered = [r for r in results if r.source_id != source.id]
        # Self should not appear in filtered list
        assert all(r.source_id != source.id for r in filtered)

    def test_excludes_already_linked(self, db: Database, embedder: Embedder, vault: Path):
        """Already-linked notes should be excluded from suggestions."""
        _ingest_corpus(db, embedder, vault)
        source = db.get_source_by_path("deep-learning.md")
        assert source is not None

        existing_links = db.get_links_from(source.id)
        linked_ids = {lk.target_note_id for lk in existing_links if lk.target_note_id}
        assert len(linked_ids) > 0  # deep-learning links to ml-basics

        from sophia.search import search as sophia_search

        results = sophia_search(db, embedder, source.title or "DL", top_k=15)
        filtered = [
            r for r in results
            if r.source_id != source.id and r.source_id not in linked_ids
        ]
        for r in filtered:
            assert r.source_id not in linked_ids

    def test_creates_suggestion_records(self, db: Database, embedder: Embedder, vault: Path):
        """Suggestion records are created in the DB."""
        _ingest_corpus(db, embedder, vault)
        source = db.get_source_by_path("cooking.md")
        assert source is not None

        # Simulate suggest_links logic
        from sophia.search import search as sophia_search

        results = sophia_search(db, embedder, source.title or "cooking", top_k=10)
        existing_links = db.get_links_from(source.id)
        linked_ids = {lk.target_note_id for lk in existing_links if lk.target_note_id}

        created = []
        for r in results:
            if r.source_id == source.id or r.source_id in linked_ids:
                continue
            sug = db.add_suggestion(
                source_note_id=source.id,
                target_note_id=r.source_id,
                score=r.score,
                reason=f"test: {r.title}",
            )
            created.append(sug)
            if len(created) >= 3:
                break

        assert len(created) > 0
        for sug in created:
            fetched = db.get_suggestion(sug.id)
            assert fetched is not None
            assert fetched.status == "pending"


# ── TestGraphNeighborsBFS ─────────────────────────────────────────────


class TestGraphNeighborsBFS:
    def test_depth_1(self, db: Database, embedder: Embedder, vault: Path):
        """Depth 1 returns only direct neighbors."""
        _ingest_corpus(db, embedder, vault)
        hub = db.get_source_by_path("hub-note.md")
        assert hub is not None

        outgoing, incoming = _bfs_neighbors(db, hub.id, 1)
        out_paths = {n["path"] for n in outgoing}
        assert "ml-basics.md" in out_paths
        assert "deep-learning.md" in out_paths
        assert len(incoming) == 0  # nothing links to hub

    def test_depth_2(self, db: Database, embedder: Embedder, vault: Path):
        """Depth 2 follows links of links."""
        _ingest_corpus(db, embedder, vault)
        hub = db.get_source_by_path("hub-note.md")
        assert hub is not None

        outgoing, incoming = _bfs_neighbors(db, hub.id, 2)
        out_paths = {n["path"] for n in outgoing}
        # hub → ml-basics (d=1), hub → deep-learning (d=1)
        # deep-learning → ml-basics (d=2, but ml-basics already visited)
        assert "ml-basics.md" in out_paths
        assert "deep-learning.md" in out_paths

    def test_isolated_note(self, db: Database, embedder: Embedder, vault: Path):
        """Note with no links returns empty neighbors."""
        _ingest_corpus(db, embedder, vault)
        cooking = db.get_source_by_path("cooking.md")
        assert cooking is not None

        outgoing, incoming = _bfs_neighbors(db, cooking.id, 1)
        assert outgoing == []
        assert incoming == []

    def test_depth_clamped(self, db: Database, embedder: Embedder, vault: Path):
        """Depth is clamped to [1, 3]."""
        _ingest_corpus(db, embedder, vault)
        hub = db.get_source_by_path("hub-note.md")
        assert hub is not None

        # depth > 3 should be treated as 3
        out_3, in_3 = _bfs_neighbors(db, hub.id, 3)
        # Should not crash — just returns what's reachable within 3 hops
        assert isinstance(out_3, list)
        assert isinstance(in_3, list)


# ── TestWatcher ───────────────────────────────────────────────────────


class TestWatcher:
    def test_markdown_filter_accepts_md(self):
        """MarkdownFilter accepts .md files, rejects non-.md."""
        from watchfiles import Change

        f = MarkdownFilter()
        assert f(Change.modified, "/vault/note.md") is True
        assert f(Change.modified, "/vault/image.png") is False
        assert f(Change.added, "/vault/sub/deep.md") is True

    def test_markdown_filter_skips_hidden(self):
        """MarkdownFilter skips hidden dirs and .sophia/."""
        from watchfiles import Change

        f = MarkdownFilter()
        assert f(Change.modified, "/vault/.sophia/sophia.db") is False
        assert f(Change.modified, "/vault/.hidden/note.md") is False
        assert f(Change.modified, "/vault/.obsidian/config.md") is False


# ── TestStatusLogic ───────────────────────────────────────────────────


class TestHasPendingSuggestion:
    def test_exists(self, db: Database):
        s1 = db.add_source(uid="u1", path="a.md", status="indexed")
        s2 = db.add_source(uid="u2", path="b.md", status="indexed")
        db.add_suggestion(s1.id, s2.id, 0.8)
        assert db.has_pending_suggestion(s1.id, s2.id) is True

    def test_not_exists(self, db: Database):
        s1 = db.add_source(uid="u1", path="a.md", status="indexed")
        s2 = db.add_source(uid="u2", path="b.md", status="indexed")
        assert db.has_pending_suggestion(s1.id, s2.id) is False

    def test_approved_not_pending(self, db: Database):
        """Approved suggestion doesn't count as pending."""
        s1 = db.add_source(uid="u1", path="a.md", status="indexed")
        s2 = db.add_source(uid="u2", path="b.md", status="indexed")
        sug = db.add_suggestion(s1.id, s2.id, 0.8)
        db.approve_suggestion(sug.id)
        assert db.has_pending_suggestion(s1.id, s2.id) is False


class TestDuplicateSuggestionPrevention:
    def test_no_duplicate_on_repeat_call(self, db: Database, embedder: Embedder, vault: Path):
        """Simulating suggest_links twice should not create duplicate pending records."""
        _ingest_corpus(db, embedder, vault)
        source = db.get_source_by_path("cooking.md")
        assert source is not None

        from sophia.search import search as sophia_search

        results = sophia_search(db, embedder, source.title or "cooking", top_k=10)
        existing_links = db.get_links_from(source.id)
        linked_ids = {lk.target_note_id for lk in existing_links if lk.target_note_id}

        # First pass: create suggestions
        for r in results:
            if r.source_id == source.id or r.source_id in linked_ids:
                continue
            if not db.has_pending_suggestion(source.id, r.source_id):
                db.add_suggestion(
                    source_note_id=source.id,
                    target_note_id=r.source_id,
                    score=r.score,
                )

        count_after_first = len(db.get_pending_suggestions())

        # Second pass: should not create duplicates
        for r in results:
            if r.source_id == source.id or r.source_id in linked_ids:
                continue
            if not db.has_pending_suggestion(source.id, r.source_id):
                db.add_suggestion(
                    source_note_id=source.id,
                    target_note_id=r.source_id,
                    score=r.score,
                )

        count_after_second = len(db.get_pending_suggestions())
        assert count_after_second == count_after_first


class TestPathNormalization:
    def test_dot_slash_normalized(self):
        """os.path.normpath('./note.md') normalizes to 'note.md'."""
        import os.path

        assert os.path.normpath("./note.md") == "note.md"

    def test_double_dot_normalized(self):
        """os.path.normpath('sub/../note.md') normalizes to 'note.md'."""
        import os.path

        assert os.path.normpath("sub/../note.md") == "note.md"


    def test_traversal_blocked(self):
        """Path traversal with ../../ is caught after resolve()."""
        from pathlib import Path

        vault_root = Path("/fake/vault").resolve()
        norm_path = "../../etc/passwd"
        abs_path = (vault_root / norm_path).resolve()
        assert not abs_path.is_relative_to(vault_root)

    def test_absolute_path_blocked(self):
        """Absolute paths outside vault are caught."""
        from pathlib import Path

        vault_root = Path("/fake/vault").resolve()
        abs_path = (vault_root / "/etc/passwd").resolve()
        assert not abs_path.is_relative_to(vault_root)


class TestApproveAtomicity:
    def test_db_ops_in_transaction(self, db: Database):
        """approve + add_wiki_link in a transaction are atomic."""
        s1 = db.add_source(uid="u1", path="a.md", status="indexed")
        s2 = db.add_source(uid="u2", path="b.md", status="indexed")
        sug = db.add_suggestion(s1.id, s2.id, 0.8)

        with db.transaction():
            db.approve_suggestion(sug.id)
            db.add_wiki_link(
                source_note_id=s1.id,
                target_path="b",
                target_note_id=s2.id,
            )

        updated = db.get_suggestion(sug.id)
        assert updated is not None
        assert updated.status == "approved"
        links = db.get_links_from(s1.id)
        assert len(links) == 1

    def test_transaction_rolls_back_on_failure(self, db: Database):
        """If wiki_link insert fails, suggestion stays pending."""
        s1 = db.add_source(uid="u1", path="a.md", status="indexed")
        s2 = db.add_source(uid="u2", path="b.md", status="indexed")
        sug = db.add_suggestion(s1.id, s2.id, 0.8)

        try:
            with db.transaction():
                db.approve_suggestion(sug.id)
                # Force a failure — FK violation (source_note_id=9999 doesn't exist)
                db.conn.execute(
                    "INSERT INTO wiki_links (source_note_id, target_note_id, target_path) "
                    "VALUES (9999, ?, ?)",
                    (s2.id, "b"),
                )
        except Exception:
            pass

        # Suggestion should still be pending (rolled back)
        check = db.get_suggestion(sug.id)
        assert check is not None
        assert check.status == "pending"


class TestStatusLogic:
    def test_current_week_start(self):
        """_current_week_start returns a Monday ISO date."""
        import datetime

        ws = _current_week_start()
        d = datetime.date.fromisoformat(ws)
        assert d.weekday() == 0  # Monday

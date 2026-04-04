"""SQLite database wrapper for Synapsis — schema, capability check, CRUD."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from synapsis.models import BudgetEntry, Chunk, Source, Suggestion, WikiLink
from synapsis.tokenizer import register_jieba_tokenizer


class CapabilityError(Exception):
    """Raised when the runtime environment doesn't meet requirements."""


class Database:
    """Single connection wrapper over Synapsis's SQLite database.

    Manages schema lifecycle, capability checking, and CRUD for all 5 entity
    types: sources, chunks, wiki_links, suggestions, budget_log.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path = ":memory:"):
        self._path = str(path)
        self._conn = self._open()
        self._local = threading.local()
        self._tx_lock = threading.Lock()

    # ── Connection setup ─────────────────────────────────────────

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")

        # Load sqlite-vec extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Register jieba FTS5 tokenizer
        register_jieba_tokenizer(conn)

        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def _commit(self) -> None:
        """Commit unless inside an explicit transaction."""
        if not getattr(self._local, "in_transaction", False):
            self._conn.commit()

    @contextmanager
    def transaction(self):
        """Context manager for explicit transactions.

        Thread-safe: uses a lock to prevent overlapping transactions, and
        thread-local state so _commit() checks are per-thread.
        """
        with self._tx_lock:
            self._local.in_transaction = True
            try:
                yield
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            finally:
                self._local.in_transaction = False

    # ── Capability check ─────────────────────────────────────────

    def check_capabilities(self) -> None:
        """Verify runtime meets Synapsis's requirements. Raises CapabilityError."""
        # Python version
        if sys.version_info < (3, 11):
            raise CapabilityError(
                f"Python >= 3.11 required, got {sys.version_info.major}.{sys.version_info.minor}"
            )

        # SQLite version (tuple comparison avoids "3.9.0" > "3.45.0" string bug)
        sqlite_ver = self._conn.execute("SELECT sqlite_version()").fetchone()[0]
        ver_tuple = tuple(int(x) for x in sqlite_ver.split("."))
        if ver_tuple < (3, 45, 0):
            raise CapabilityError(f"SQLite >= 3.45 required, got {sqlite_ver}")

        # sqlite-vec
        try:
            self._conn.execute("SELECT vec_version()").fetchone()
        except Exception as e:
            raise CapabilityError(f"sqlite-vec not available: {e}") from e

        # FTS5
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_check USING fts5(x)"
            )
            self._conn.execute("DROP TABLE IF EXISTS _fts5_check")
        except Exception as e:
            raise CapabilityError(f"FTS5 not available: {e}") from e

    # ── Schema init ──────────────────────────────────────────────

    def init_schema(self) -> None:
        """Create all tables, virtual tables, triggers, and indexes.

        Uses PRAGMA user_version for versioning. Skips if already current.
        Raises CapabilityError if DB version is newer than code version.
        """
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == self.SCHEMA_VERSION:
            return
        if version > self.SCHEMA_VERSION:
            raise CapabilityError(
                f"DB schema version {version} > code version {self.SCHEMA_VERSION}. "
                "Delete the DB file to recreate."
            )

        # -- Tables (order matters for FK references) --

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
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
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                token_count INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_note_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                target_note_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
                target_path TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(source_note_id, target_path)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_note_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                target_note_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                score REAL NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                resolved_at TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start TEXT NOT NULL UNIQUE,
                tokens_used INTEGER DEFAULT 0,
                breakdown TEXT DEFAULT '{}',
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # -- Virtual tables --

        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[768] distance_metric=cosine
            )
        """)

        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                content,
                content=chunks,
                content_rowid=id,
                tokenize='jieba'
            )
        """)

        # -- Triggers: FTS5 sync (3) --

        self._create_trigger(
            "chunks_fts_insert",
            """CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
            END""",
        )

        self._create_trigger(
            "chunks_fts_delete",
            """CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, content)
                    VALUES ('delete', old.id, old.content);
            END""",
        )

        self._create_trigger(
            "chunks_fts_update",
            """CREATE TRIGGER chunks_fts_update AFTER UPDATE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, content)
                    VALUES ('delete', old.id, old.content);
                INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
            END""",
        )

        # -- Triggers: indegree maintenance (3) --

        self._create_trigger(
            "wikilinks_insert_indegree",
            """CREATE TRIGGER wikilinks_insert_indegree AFTER INSERT ON wiki_links
            WHEN new.target_note_id IS NOT NULL BEGIN
                UPDATE sources SET indegree = (
                    SELECT COUNT(*) FROM wiki_links
                    WHERE target_note_id = new.target_note_id
                ) WHERE id = new.target_note_id;
            END""",
        )

        self._create_trigger(
            "wikilinks_delete_indegree",
            """CREATE TRIGGER wikilinks_delete_indegree AFTER DELETE ON wiki_links
            WHEN old.target_note_id IS NOT NULL BEGIN
                UPDATE sources SET indegree = (
                    SELECT COUNT(*) FROM wiki_links
                    WHERE target_note_id = old.target_note_id
                ) WHERE id = old.target_note_id;
            END""",
        )

        # UPDATE triggers for target_note_id changes (covers all transitions):
        # - NULL → non-NULL (forward ref resolution)
        # - non-NULL A → non-NULL B (retargeting)
        # - non-NULL → NULL (handled by _old trigger)
        self._create_trigger(
            "wikilinks_update_indegree_new",
            """CREATE TRIGGER wikilinks_update_indegree_new
            AFTER UPDATE OF target_note_id ON wiki_links
            WHEN new.target_note_id IS NOT NULL BEGIN
                UPDATE sources SET indegree = (
                    SELECT COUNT(*) FROM wiki_links
                    WHERE target_note_id = new.target_note_id
                ) WHERE id = new.target_note_id;
            END""",
        )

        self._create_trigger(
            "wikilinks_update_indegree_old",
            """CREATE TRIGGER wikilinks_update_indegree_old
            AFTER UPDATE OF target_note_id ON wiki_links
            WHEN old.target_note_id IS NOT NULL
                AND (new.target_note_id IS NULL OR new.target_note_id != old.target_note_id)
            BEGIN
                UPDATE sources SET indegree = (
                    SELECT COUNT(*) FROM wiki_links
                    WHERE target_note_id = old.target_note_id
                ) WHERE id = old.target_note_id;
            END""",
        )

        # -- Indexes --

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_uid ON sources(uid)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_indegree ON sources(indegree)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wikilinks_source ON wiki_links(source_note_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wikilinks_target ON wiki_links(target_note_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wikilinks_target_path ON wiki_links(target_path)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_budget_week ON budget_log(week_start)"
        )

        # -- Set schema version --

        self._conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
        self._conn.commit()

    def _create_trigger(self, name: str, ddl: str) -> None:
        """Create trigger if it doesn't already exist (no IF NOT EXISTS for triggers)."""
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
        ).fetchone()
        if not exists:
            self._conn.execute(ddl)

    # ── Source CRUD ───────────────────────────────────────────────

    def add_source(
        self,
        uid: str,
        path: str,
        *,
        title: str | None = None,
        content_hash: str | None = None,
        status: str = "unprocessed",
        fs_modified_at: str | None = None,
    ) -> Source:
        cursor = self._conn.execute(
            """INSERT INTO sources (uid, path, title, content_hash, status, fs_modified_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (uid, path, title, content_hash, status, fs_modified_at),
        )
        self._commit()
        return self.get_source(cursor.lastrowid)  # type: ignore[return-value]

    def get_source(self, id: int) -> Source | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE id = ?", (id,)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def get_source_by_uid(self, uid: str) -> Source | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE uid = ?", (uid,)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def get_source_by_path(self, path: str) -> Source | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE path = ?", (path,)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def list_sources(self, status: str | None = None) -> list[Source]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM sources WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM sources").fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_sources_by_ids(self, ids: list[int]) -> dict[int, Source]:
        """Batch-fetch sources by IDs. Returns {source_id: Source}."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM sources WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {r["id"]: self._row_to_source(r) for r in rows}

    _SOURCE_UPDATABLE = frozenset({
        "path", "title", "content_hash", "status",
        "indegree", "fs_modified_at", "indexed_at",
    })

    def update_source(self, id: int, **kwargs: str | int | None) -> Source | None:
        if not kwargs:
            return self.get_source(id)
        bad_keys = set(kwargs) - self._SOURCE_UPDATABLE
        if bad_keys:
            raise ValueError(f"Cannot update columns: {bad_keys}")
        sets = []
        vals: list[str | int | None] = []
        for key, val in kwargs.items():
            sets.append(f"{key} = ?")
            vals.append(val)
        sets.append("updated_at = datetime('now')")
        vals.append(id)
        self._conn.execute(
            f"UPDATE sources SET {', '.join(sets)} WHERE id = ?", vals
        )
        self._commit()
        return self.get_source(id)

    def rename_source(self, id: int, new_path: str) -> Source | None:
        """Rename a source's path. UID remains stable."""
        return self.update_source(id, path=new_path)

    def delete_source(self, id: int) -> bool:
        """Delete source and clean up vec_chunks (no FK cascade for virtual tables)."""
        # vec_chunks cascade gap: explicitly delete before CASCADE removes chunks
        chunk_ids = self._conn.execute(
            "SELECT id FROM chunks WHERE source_id = ?", (id,)
        ).fetchall()
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            self._conn.execute(
                f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})",
                [row["id"] for row in chunk_ids],
            )

        cursor = self._conn.execute("DELETE FROM sources WHERE id = ?", (id,))
        self._commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"],
            uid=row["uid"],
            path=row["path"],
            title=row["title"],
            content_hash=row["content_hash"],
            status=row["status"],
            indegree=row["indegree"],
            fs_modified_at=row["fs_modified_at"],
            indexed_at=row["indexed_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Chunk CRUD ───────────────────────────────────────────────

    def add_chunk(
        self,
        source_id: int,
        content: str,
        chunk_index: int,
        *,
        char_start: int | None = None,
        char_end: int | None = None,
        token_count: int | None = None,
    ) -> Chunk:
        cursor = self._conn.execute(
            """INSERT INTO chunks (source_id, content, chunk_index,
               char_start, char_end, token_count)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (source_id, content, chunk_index, char_start, char_end, token_count),
        )
        self._commit()
        return self._get_chunk(cursor.lastrowid)  # type: ignore[return-value]

    def get_chunks_by_source(self, source_id: int) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY chunk_index",
            (source_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def delete_chunks_by_source(self, source_id: int) -> int:
        # Clean up vec_chunks first (virtual table, no FK cascade)
        chunk_ids = self._conn.execute(
            "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchall()
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            self._conn.execute(
                f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})",
                [row["id"] for row in chunk_ids],
            )
        cursor = self._conn.execute(
            "DELETE FROM chunks WHERE source_id = ?", (source_id,)
        )
        self._commit()
        return cursor.rowcount

    def search_fts(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        """Full-text search via FTS5. Returns (Chunk, bm25_rank) pairs."""
        rows = self._conn.execute(
            """SELECT c.id, c.source_id, c.content, c.chunk_index,
                      c.char_start, c.char_end, c.token_count, c.created_at,
                      fts.rank
            FROM fts_chunks fts
            JOIN chunks c ON c.id = fts.rowid
            WHERE fts_chunks MATCH ?
            ORDER BY fts.rank
            LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [
            (
                Chunk(
                    id=r["id"],
                    source_id=r["source_id"],
                    content=r["content"],
                    chunk_index=r["chunk_index"],
                    char_start=r["char_start"],
                    char_end=r["char_end"],
                    token_count=r["token_count"],
                    created_at=r["created_at"],
                ),
                r["rank"],
            )
            for r in rows
        ]

    def get_chunk(self, id: int) -> Chunk | None:
        """Get a single chunk by ID."""
        return self._get_chunk(id)

    def get_chunks_by_ids(self, ids: list[int]) -> dict[int, Chunk]:
        """Batch-fetch chunks by IDs. Returns {chunk_id: Chunk}."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        return {r["id"]: self._row_to_chunk(r) for r in rows}

    def _get_chunk(self, id: int) -> Chunk | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            source_id=row["source_id"],
            content=row["content"],
            chunk_index=row["chunk_index"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            token_count=row["token_count"],
            created_at=row["created_at"],
        )

    # ── Vec CRUD ─────────────────────────────────────────────────

    def upsert_vec(self, chunk_id: int, embedding: bytes) -> None:
        """Insert or replace a vector embedding for a chunk."""
        self._conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, embedding),
        )
        self._commit()

    def delete_vec_by_source(self, source_id: int) -> int:
        """Delete vec_chunks entries for all chunks belonging to a source."""
        chunk_ids = self._conn.execute(
            "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchall()
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = self._conn.execute(
            f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})",
            [row["id"] for row in chunk_ids],
        )
        self._commit()
        return cursor.rowcount

    def search_vec(
        self, embedding: bytes, k: int = 20
    ) -> list[tuple[int, float]]:
        """KNN vector search. Returns (chunk_id, distance) pairs."""
        rows = self._conn.execute(
            "SELECT chunk_id, distance FROM vec_chunks "
            "WHERE embedding MATCH ? AND k = ?",
            (embedding, k),
        ).fetchall()
        return [(row["chunk_id"], row["distance"]) for row in rows]

    # ── WikiLink CRUD ────────────────────────────────────────────

    def add_wiki_link(
        self,
        source_note_id: int,
        target_path: str,
        target_note_id: int | None = None,
    ) -> WikiLink | None:
        """Insert a wiki link (INSERT OR IGNORE on unique constraint)."""
        cursor = self._conn.execute(
            """INSERT OR IGNORE INTO wiki_links
               (source_note_id, target_note_id, target_path)
            VALUES (?, ?, ?)""",
            (source_note_id, target_note_id, target_path),
        )
        self._commit()
        if cursor.lastrowid and cursor.rowcount > 0:
            row = self._conn.execute(
                "SELECT * FROM wiki_links WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return self._row_to_wiki_link(row) if row else None
        return None

    def get_links_from(self, source_note_id: int) -> list[WikiLink]:
        rows = self._conn.execute(
            "SELECT * FROM wiki_links WHERE source_note_id = ?",
            (source_note_id,),
        ).fetchall()
        return [self._row_to_wiki_link(r) for r in rows]

    def get_links_to(self, target_note_id: int) -> list[WikiLink]:
        rows = self._conn.execute(
            "SELECT * FROM wiki_links WHERE target_note_id = ?",
            (target_note_id,),
        ).fetchall()
        return [self._row_to_wiki_link(r) for r in rows]

    def resolve_forward_refs(self, target_path: str, target_note_id: int) -> int:
        """Resolve forward references: set target_note_id where target_path matches."""
        cursor = self._conn.execute(
            """UPDATE wiki_links SET target_note_id = ?
            WHERE target_path = ? AND target_note_id IS NULL""",
            (target_note_id, target_path),
        )
        self._commit()
        return cursor.rowcount

    def delete_links_by_source(self, source_note_id: int) -> int:
        cursor = self._conn.execute(
            "DELETE FROM wiki_links WHERE source_note_id = ?",
            (source_note_id,),
        )
        self._commit()
        return cursor.rowcount

    @staticmethod
    def _row_to_wiki_link(row: sqlite3.Row) -> WikiLink:
        return WikiLink(
            id=row["id"],
            source_note_id=row["source_note_id"],
            target_note_id=row["target_note_id"],
            target_path=row["target_path"],
            created_at=row["created_at"],
        )

    # ── Suggestion CRUD ──────────────────────────────────────────

    def add_suggestion(
        self,
        source_note_id: int,
        target_note_id: int,
        score: float,
        reason: str | None = None,
    ) -> Suggestion:
        cursor = self._conn.execute(
            """INSERT INTO suggestions
               (source_note_id, target_note_id, score, reason)
            VALUES (?, ?, ?, ?)""",
            (source_note_id, target_note_id, score, reason),
        )
        self._commit()
        row = self._conn.execute(
            "SELECT * FROM suggestions WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return self._row_to_suggestion(row)

    def has_pending_suggestion(self, source_note_id: int, target_note_id: int) -> bool:
        """Check if a pending suggestion already exists for this pair."""
        row = self._conn.execute(
            "SELECT 1 FROM suggestions WHERE source_note_id = ? "
            "AND target_note_id = ? AND status = 'pending' LIMIT 1",
            (source_note_id, target_note_id),
        ).fetchone()
        return row is not None

    def get_pending_suggestions(self) -> list[Suggestion]:
        rows = self._conn.execute(
            "SELECT * FROM suggestions WHERE status = 'pending' ORDER BY score DESC"
        ).fetchall()
        return [self._row_to_suggestion(r) for r in rows]

    def approve_suggestion(self, id: int) -> Suggestion | None:
        self._conn.execute(
            """UPDATE suggestions
            SET status = 'approved', resolved_at = datetime('now')
            WHERE id = ? AND status = 'pending'""",
            (id,),
        )
        self._commit()
        row = self._conn.execute(
            "SELECT * FROM suggestions WHERE id = ?", (id,)
        ).fetchone()
        return self._row_to_suggestion(row) if row else None

    def reject_suggestion(self, id: int) -> Suggestion | None:
        self._conn.execute(
            """UPDATE suggestions
            SET status = 'rejected', resolved_at = datetime('now')
            WHERE id = ? AND status = 'pending'""",
            (id,),
        )
        self._commit()
        row = self._conn.execute(
            "SELECT * FROM suggestions WHERE id = ?", (id,)
        ).fetchone()
        return self._row_to_suggestion(row) if row else None

    def get_suggestion(self, id: int) -> Suggestion | None:
        """Get a single suggestion by ID."""
        row = self._conn.execute(
            "SELECT * FROM suggestions WHERE id = ?", (id,)
        ).fetchone()
        return self._row_to_suggestion(row) if row else None

    @staticmethod
    def _row_to_suggestion(row: sqlite3.Row) -> Suggestion:
        return Suggestion(
            id=row["id"],
            source_note_id=row["source_note_id"],
            target_note_id=row["target_note_id"],
            score=row["score"],
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )

    # ── Budget CRUD ──────────────────────────────────────────────

    def get_or_create_budget(self, week_start: str) -> BudgetEntry:
        row = self._conn.execute(
            "SELECT * FROM budget_log WHERE week_start = ?", (week_start,)
        ).fetchone()
        if row:
            return self._row_to_budget(row)

        self._conn.execute(
            "INSERT INTO budget_log (week_start) VALUES (?)", (week_start,)
        )
        self._commit()
        row = self._conn.execute(
            "SELECT * FROM budget_log WHERE week_start = ?", (week_start,)
        ).fetchone()
        return self._row_to_budget(row)

    def add_tokens(
        self, week_start: str, tokens: int, category: str = "search"
    ) -> BudgetEntry:
        entry = self.get_or_create_budget(week_start)
        breakdown = json.loads(entry.breakdown)
        breakdown[category] = breakdown.get(category, 0) + tokens
        new_total = entry.tokens_used + tokens

        self._conn.execute(
            """UPDATE budget_log
            SET tokens_used = ?, breakdown = ?, updated_at = datetime('now')
            WHERE week_start = ?""",
            (new_total, json.dumps(breakdown), week_start),
        )
        self._commit()

        row = self._conn.execute(
            "SELECT * FROM budget_log WHERE week_start = ?", (week_start,)
        ).fetchone()
        return self._row_to_budget(row)

    def get_stats(self) -> dict:
        """Get aggregate statistics for the knowledge base."""
        return {
            "notes_total": self._conn.execute(
                "SELECT COUNT(*) FROM sources"
            ).fetchone()[0],
            "notes_unprocessed": self._conn.execute(
                "SELECT COUNT(*) FROM sources WHERE status != 'indexed'"
            ).fetchone()[0],
            "chunks_total": self._conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0],
            "links_total": self._conn.execute(
                "SELECT COUNT(*) FROM wiki_links"
            ).fetchone()[0],
            "suggestions_pending": self._conn.execute(
                "SELECT COUNT(*) FROM suggestions WHERE status = 'pending'"
            ).fetchone()[0],
        }

    @staticmethod
    def _row_to_budget(row: sqlite3.Row) -> BudgetEntry:
        return BudgetEntry(
            id=row["id"],
            week_start=row["week_start"],
            tokens_used=row["tokens_used"],
            breakdown=row["breakdown"],
            updated_at=row["updated_at"],
        )

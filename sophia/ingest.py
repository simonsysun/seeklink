"""Ingestion pipeline — orchestrates parse → chunk → embed → store."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sophia.chunker import chunk_markdown
from sophia.db import Database
from sophia.embedder import Embedder
from sophia.link_parser import extract_wiki_links
from sophia.models import Source

logger = logging.getLogger(__name__)

# Non-hidden top-level dirs excluded from indexing (mirrors watcher._SKIP_DIRS)
_SKIP_DIRS = {"todo", "archive"}


def _utcnow() -> str:
    """Return current UTC timestamp in SQLite-compatible format."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def ingest_file(
    db: Database,
    path: Path,
    vault_root: Path,
    embedder: Embedder,
) -> Source | None:
    """Ingest a single markdown file into the database.

    Returns the Source if ingested/unchanged, None if skipped or errored.
    Wraps the mutation phase in a transaction for atomicity.
    """
    # Skip non-.md files
    if path.suffix.lower() != ".md":
        return None

    # Read file
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        logger.warning("Skipping %s: %s", path, e)
        return None

    # Compute content hash and relative path
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    rel_path = str(path.relative_to(vault_root))

    # Check existing source
    existing = db.get_source_by_path(rel_path)
    if existing is not None and existing.content_hash == content_hash:
        return existing  # unchanged

    # Prepare data outside transaction: title, chunks, embeddings, links
    title = _extract_title(content, path)
    chunks = chunk_markdown(content)

    if chunks:
        embeddings = embedder.embed_documents([c.text for c in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks"
            )
    else:
        embeddings = []

    targets = extract_wiki_links(content)

    # Mutate DB atomically
    with db.transaction():
        if existing is not None:
            # Re-index: delete old data
            db.delete_chunks_by_source(existing.id)
            db.delete_links_by_source(existing.id)
            source = existing
        else:
            source = db.add_source(
                uid=str(uuid4()),
                path=rel_path,
                content_hash=content_hash,
            )

        # Store chunks and vectors
        for i, (chunk_span, emb) in enumerate(zip(chunks, embeddings)):
            db_chunk = db.add_chunk(
                source_id=source.id,
                content=chunk_span.text,
                chunk_index=i,
                char_start=chunk_span.char_start,
                char_end=chunk_span.char_end,
                token_count=chunk_span.token_count,
            )
            db.upsert_vec(db_chunk.id, emb)

        # Store wiki-links
        for target in targets:
            target_source = _find_source_by_target(db, target)
            db.add_wiki_link(
                source_note_id=source.id,
                target_path=target,
                target_note_id=target_source.id if target_source else None,
            )

        # Resolve forward refs pointing TO this file (by stem and by relative path)
        stem = path.stem
        rel_no_ext = rel_path.removesuffix(".md")
        db.resolve_forward_refs(stem, source.id)
        if rel_no_ext != stem:
            db.resolve_forward_refs(rel_no_ext, source.id)

        # Update source status with real timestamp
        db.update_source(
            source.id,
            title=title,
            content_hash=content_hash,
            status="indexed",
            indexed_at=_utcnow(),
        )

    return db.get_source(source.id)


def ingest_vault(
    db: Database,
    vault_root: Path,
    embedder: Embedder | None = None,
) -> dict[str, int]:
    """Ingest all markdown files in a vault directory.

    Returns stats: {"ingested": N, "unchanged": N, "skipped": N, "errors": N}
    """
    if embedder is None:
        embedder = Embedder()

    stats = {"ingested": 0, "unchanged": 0, "skipped": 0, "errors": 0}

    for md_path in sorted(vault_root.rglob("*.md")):
        # Skip hidden dirs and excluded dirs (mirrors watcher.MarkdownFilter)
        try:
            rel = md_path.relative_to(vault_root)
        except ValueError:
            continue
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts):
            continue
        # Snapshot existing source to detect unchanged
        rel_path = str(md_path.relative_to(vault_root))
        existing = db.get_source_by_path(rel_path)

        try:
            result = ingest_file(db, md_path, vault_root, embedder)
        except Exception:
            logger.exception("Error ingesting %s", md_path)
            stats["errors"] += 1
            continue

        if result is None:
            stats["skipped"] += 1
        elif existing is not None and existing.content_hash == result.content_hash and existing.status == "indexed":
            stats["unchanged"] += 1
        else:
            stats["ingested"] += 1

    return stats


def _extract_title(content: str, path: Path) -> str:
    """Extract title from first # heading, falling back to filename stem."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped[2:].strip()
    return path.stem


def _find_source_by_target(db: Database, target: str) -> Source | None:
    """Resolve a wiki-link target to a source.

    Resolution order:
    1. Exact path match with .md suffix
    2. Exact path match as-is
    3. Stem match (only for simple names without path separators)
    """
    # Try exact path match
    source = db.get_source_by_path(target + ".md")
    if source:
        return source
    source = db.get_source_by_path(target)
    if source:
        return source

    # Fall back to stem match (only for simple names)
    if "/" not in target and "\\" not in target:
        for source in db.list_sources():
            if Path(source.path).stem == target:
                return source

    return None

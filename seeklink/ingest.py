"""Ingestion pipeline — orchestrates parse → chunk → embed → store."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from seeklink.chunker import chunk_markdown
from seeklink.db import Database
from seeklink.embedder import Embedder
from seeklink.link_parser import extract_wiki_links
from seeklink.models import Source

logger = logging.getLogger(__name__)

# Non-hidden top-level dirs excluded from indexing (mirrors watcher._SKIP_DIRS)
_SKIP_DIRS = {"todo", "archive"}

# Regex for YAML frontmatter block (handles empty frontmatter too).
# Public — search.py imports this to map body-relative chunk offsets back
# to full-file line numbers when building search results.
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)---\s*\n", re.DOTALL)
_FRONTMATTER_RE = FRONTMATTER_RE  # backward-compat alias within this module


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
        # Content unchanged — but refresh indexed_at so that freshness
        # checks don't keep warning about this file after a content-
        # preserving touch / git checkout / editor save-without-change.
        if existing.status == "indexed":
            db.update_source(existing.id, indexed_at=_utcnow())
        return existing

    # Parse frontmatter → extract tags, aliases, and body (content without YAML)
    tags, aliases, body = _parse_frontmatter(content)

    # Prepare data outside transaction: title, chunks, embeddings, links
    # Use body (stripped of frontmatter) for chunking/embedding/link parsing
    title = _extract_title(body, path)
    chunks = chunk_markdown(body)

    if chunks:
        embeddings = embedder.embed_documents([c.text for c in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedder returned {len(embeddings)} embeddings for {len(chunks)} chunks"
            )
    else:
        embeddings = []

    targets = extract_wiki_links(body)
    aliases_json = json.dumps(aliases, ensure_ascii=False)

    # Mutate DB atomically
    with db.transaction():
        if existing is not None:
            # Re-index: delete old data
            db.delete_chunks_by_source(existing.id)
            db.delete_links_by_source(existing.id)
            db.delete_tags_by_source(existing.id)
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

        # Store wiki-links (with alias-aware resolution)
        for target in targets:
            target_source = _find_source_by_target(db, target)
            db.add_wiki_link(
                source_note_id=source.id,
                target_path=target,
                target_note_id=target_source.id if target_source else None,
            )

        # Store tags
        if tags:
            db.add_tags(source.id, tags)

        # Resolve forward refs pointing TO this file (by stem, relative path, and aliases)
        stem = path.stem
        rel_no_ext = rel_path.removesuffix(".md")
        db.resolve_forward_refs(stem, source.id)
        if rel_no_ext != stem:
            db.resolve_forward_refs(rel_no_ext, source.id)
        for alias in aliases:
            db.resolve_forward_refs(alias, source.id)

        # Update source status with real timestamp
        db.update_source(
            source.id,
            title=title,
            content_hash=content_hash,
            status="indexed",
            indexed_at=_utcnow(),
            aliases=aliases_json,
        )

    return db.get_source(source.id)


def ingest_vault(
    db: Database,
    vault_root: Path,
    embedder: Embedder | None = None,
) -> dict[str, int]:
    """Ingest all markdown files in a vault directory and prune stale entries.

    Returns stats: {"ingested": N, "unchanged": N, "skipped": N, "errors": N, "pruned": N}

    After processing all existing files, walks DB entries and removes any
    whose path no longer exists on disk. This handles files that were
    deleted or moved outside the rhizome CLI (where no explicit
    ``seeklink index`` call would have fired).
    """
    if embedder is None:
        embedder = Embedder()

    stats = {"ingested": 0, "unchanged": 0, "skipped": 0, "errors": 0, "pruned": 0}
    seen_paths: set[str] = set()

    for md_path in sorted(vault_root.rglob("*.md")):
        try:
            rel = md_path.relative_to(vault_root)
        except ValueError:
            continue
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts):
            continue
        rel_path = str(rel)
        seen_paths.add(rel_path)
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

    # Prune DB entries for files that no longer exist on disk
    for src in db.list_sources():
        if src.path not in seen_paths:
            db.delete_source(src.id)
            stats["pruned"] += 1
            logger.info("Pruned stale entry: %s", src.path)

    return stats


def _parse_frontmatter(content: str) -> tuple[list[str], list[str], str]:
    """Parse YAML frontmatter, returning (tags, aliases, body).

    Handles both formats for tags and aliases:
    - Inline: `tags: [ai, ml, deep-learning]`
    - Block list: `tags:\\n  - ai\\n  - ml`

    Returns ([], [], content) if no frontmatter found or on parse error.
    Body is the content after the frontmatter block.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return [], [], content

    yaml_block = match.group(1)
    body = content[match.end():]

    tags = _parse_yaml_list_field(yaml_block, "tags")
    aliases = _parse_yaml_list_field(yaml_block, "aliases")

    return tags, aliases, body


def _parse_yaml_list_field(yaml_block: str, field: str) -> list[str]:
    """Extract a list field from a YAML block. Handles inline and block list formats."""
    # Try inline format: field: [a, b, c]
    inline_re = re.compile(rf"^{re.escape(field)}\s*:\s*\[([^\]]*)\]", re.MULTILINE)
    m = inline_re.search(yaml_block)
    if m:
        raw = m.group(1)
        items = [s.strip().strip("'\"") for s in raw.split(",")]
        return [s for s in items if s]

    # Try block list format:
    # field:
    #   - item1
    #   - item2
    block_re = re.compile(
        rf"^{re.escape(field)}\s*:\s*\n((?:\s+-\s+.+\n?)+)", re.MULTILINE
    )
    m = block_re.search(yaml_block)
    if m:
        block = m.group(1)
        items = re.findall(r"^\s+-\s+(.+)$", block, re.MULTILINE)
        return [s.strip().strip("'\"") for s in items if s.strip()]

    # Try single value: field: value
    single_re = re.compile(rf"^{re.escape(field)}\s*:\s+(.+)$", re.MULTILINE)
    m = single_re.search(yaml_block)
    if m:
        val = m.group(1).strip().strip("'\"")
        if val and not val.startswith("["):
            return [val]

    return []


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
    3. Stem match via SQL (for simple names without path separators)
    4. Alias match (check aliases JSON field)
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
        source = db.get_source_by_stem(target)
        if source:
            return source

        # Try alias match
        source = db.get_source_by_alias(target)
        if source:
            return source

    return None

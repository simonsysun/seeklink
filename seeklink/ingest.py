"""Ingestion pipeline — orchestrates parse → chunk → embed → store."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from seeklink.chunker import ChunkSpan, chunk_markdown
from seeklink.db import Database
from seeklink.embedder import Embedder
from seeklink.link_parser import extract_wiki_links
from seeklink.models import Source

logger = logging.getLogger(__name__)

# Non-hidden top-level dirs excluded from indexing (mirrors freshness._SKIP_DIRS)
_SKIP_DIRS = {"todo", "archive"}
_EMBED_BATCH_SIZE = 16

# Regex for YAML frontmatter block (handles empty frontmatter too).
# Public — search.py imports this to map body-relative chunk offsets back
# to full-file line numbers when building search results.
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)---\s*\n", re.DOTALL)
_FRONTMATTER_RE = FRONTMATTER_RE  # backward-compat alias within this module


@dataclass(slots=True)
class _PreparedFile:
    path: Path
    rel_path: str
    content_hash: str
    existing: Source | None
    title: str
    chunks: list[ChunkSpan]
    targets: list[str]
    tags: list[str]
    aliases: list[str]
    aliases_json: str
    headings_json: str
    unchanged: bool = False


@dataclass(frozen=True, slots=True)
class _ChunkEmbeddingItem:
    file_index: int
    chunk_index: int
    text: str


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
    if (
        existing is not None
        and existing.content_hash == content_hash
        and existing.status == "indexed"
    ):
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
    headings = _extract_headings(body, title)
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
    headings_json = json.dumps(headings, ensure_ascii=False)

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
            headings=headings_json,
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
    deleted or moved outside an explicit ``seeklink index`` run.
    """
    if embedder is None:
        embedder = Embedder()

    stats = {"ingested": 0, "unchanged": 0, "skipped": 0, "errors": 0, "pruned": 0}
    seen_paths: set[str] = set()
    prepared_files: list[_PreparedFile] = []

    for md_path in sorted(vault_root.rglob("*.md")):
        try:
            rel = md_path.relative_to(vault_root)
        except ValueError:
            continue
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts):
            continue
        rel_path = str(rel)
        seen_paths.add(rel_path)

        try:
            prepared = _prepare_file(db, md_path, vault_root)
        except Exception:
            logger.exception("Error preparing %s", md_path)
            stats["errors"] += 1
            continue

        if prepared is None:
            stats["skipped"] += 1
        elif prepared.unchanged:
            stats["unchanged"] += 1
        else:
            prepared_files.append(prepared)

    embeddings_by_file, embed_errors = _embed_prepared_files(
        prepared_files,
        embedder,
        batch_size=_EMBED_BATCH_SIZE,
    )

    for i, prepared in enumerate(prepared_files):
        if i in embed_errors:
            error = embed_errors[i]
            logger.error(
                "Error embedding %s: %s",
                prepared.path,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            stats["errors"] += 1
            continue

        try:
            _write_prepared_file(
                db,
                prepared,
                embeddings_by_file.get(i, []),
            )
        except Exception:
            logger.exception("Error ingesting %s", prepared.path)
            stats["errors"] += 1
            continue

        stats["ingested"] += 1

    # Prune DB entries for files that no longer exist on disk
    for src in db.list_sources():
        if src.path not in seen_paths:
            db.delete_source(src.id)
            stats["pruned"] += 1
            logger.info("Pruned stale entry: %s", src.path)

    return stats


def _prepare_file(
    db: Database,
    path: Path,
    vault_root: Path,
) -> _PreparedFile | None:
    """Read and parse a markdown file before batch embedding."""
    if path.suffix.lower() != ".md":
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        logger.warning("Skipping %s: %s", path, e)
        return None

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    rel_path = str(path.relative_to(vault_root))

    existing = db.get_source_by_path(rel_path)
    if (
        existing is not None
        and existing.content_hash == content_hash
        and existing.status == "indexed"
    ):
        db.update_source(existing.id, indexed_at=_utcnow())
        return _PreparedFile(
            path=path,
            rel_path=rel_path,
            content_hash=content_hash,
            existing=existing,
            title=existing.title or path.stem,
            chunks=[],
            targets=[],
            tags=[],
            aliases=[],
            aliases_json=existing.aliases,
            headings_json=existing.headings,
            unchanged=True,
        )

    tags, aliases, body = _parse_frontmatter(content)
    title = _extract_title(body, path)
    headings = _extract_headings(body, title)
    chunks = chunk_markdown(body)
    targets = extract_wiki_links(body)
    aliases_json = json.dumps(aliases, ensure_ascii=False)
    headings_json = json.dumps(headings, ensure_ascii=False)

    return _PreparedFile(
        path=path,
        rel_path=rel_path,
        content_hash=content_hash,
        existing=existing,
        title=title,
        chunks=chunks,
        targets=targets,
        tags=tags,
        aliases=aliases,
        aliases_json=aliases_json,
        headings_json=headings_json,
    )


def _embed_prepared_files(
    prepared_files: list[_PreparedFile],
    embedder: Embedder,
    *,
    batch_size: int,
) -> tuple[dict[int, list[bytes]], dict[int, Exception]]:
    """Embed all prepared chunks in length-sorted batches.

    Sorting by text length avoids mixing very short and very long chunks in the
    same ONNX batch, which otherwise wastes time on padding. Results are mapped
    back to the original file/chunk positions before writing.
    """
    items: list[_ChunkEmbeddingItem] = []
    for file_index, prepared in enumerate(prepared_files):
        for chunk_index, chunk in enumerate(prepared.chunks):
            items.append(
                _ChunkEmbeddingItem(
                    file_index=file_index,
                    chunk_index=chunk_index,
                    text=chunk.text,
                )
            )

    items.sort(key=lambda item: len(item.text))
    embeddings_by_file: dict[int, list[bytes | None]] = {
        file_index: [None] * len(prepared.chunks)
        for file_index, prepared in enumerate(prepared_files)
    }
    errors: dict[int, Exception] = {}

    def embed_items(batch: list[_ChunkEmbeddingItem]) -> None:
        if not batch:
            return
        try:
            embeddings = embedder.embed_documents([item.text for item in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedder returned {len(embeddings)} embeddings "
                    f"for {len(batch)} chunks"
                )
        except Exception as e:
            if len(batch) == 1:
                errors.setdefault(batch[0].file_index, e)
                return
            mid = len(batch) // 2
            embed_items(batch[:mid])
            embed_items(batch[mid:])
            return

        for item, embedding in zip(batch, embeddings):
            embeddings_by_file[item.file_index][item.chunk_index] = embedding

    for start in range(0, len(items), batch_size):
        embed_items(items[start : start + batch_size])

    out: dict[int, list[bytes]] = {}
    for file_index, embeddings in embeddings_by_file.items():
        if file_index in errors:
            continue
        if any(embedding is None for embedding in embeddings):
            errors.setdefault(
                file_index,
                RuntimeError(
                    f"missing embeddings for {prepared_files[file_index].rel_path}"
                ),
            )
            continue
        out[file_index] = [
            embedding for embedding in embeddings if embedding is not None
        ]

    return out, errors


def _write_prepared_file(
    db: Database,
    prepared: _PreparedFile,
    embeddings: list[bytes],
) -> Source | None:
    """Persist one prepared file after embeddings have been computed."""
    if len(embeddings) != len(prepared.chunks):
        raise RuntimeError(
            f"Embedding count mismatch for {prepared.rel_path}: "
            f"{len(embeddings)} embeddings for {len(prepared.chunks)} chunks"
        )

    with db.transaction():
        if prepared.existing is not None:
            db.delete_chunks_by_source(prepared.existing.id)
            db.delete_links_by_source(prepared.existing.id)
            db.delete_tags_by_source(prepared.existing.id)
            source = prepared.existing
        else:
            source = db.add_source(
                uid=str(uuid4()),
                path=prepared.rel_path,
                content_hash=prepared.content_hash,
            )

        for i, (chunk_span, emb) in enumerate(zip(prepared.chunks, embeddings)):
            db_chunk = db.add_chunk(
                source_id=source.id,
                content=chunk_span.text,
                chunk_index=i,
                char_start=chunk_span.char_start,
                char_end=chunk_span.char_end,
                token_count=chunk_span.token_count,
            )
            db.upsert_vec(db_chunk.id, emb)

        for target in prepared.targets:
            target_source = _find_source_by_target(db, target)
            db.add_wiki_link(
                source_note_id=source.id,
                target_path=target,
                target_note_id=target_source.id if target_source else None,
            )

        if prepared.tags:
            db.add_tags(source.id, prepared.tags)

        stem = prepared.path.stem
        rel_no_ext = prepared.rel_path.removesuffix(".md")
        db.resolve_forward_refs(stem, source.id)
        if rel_no_ext != stem:
            db.resolve_forward_refs(rel_no_ext, source.id)
        for alias in prepared.aliases:
            db.resolve_forward_refs(alias, source.id)

        db.update_source(
            source.id,
            title=prepared.title,
            content_hash=prepared.content_hash,
            status="indexed",
            indexed_at=_utcnow(),
            aliases=prepared.aliases_json,
            headings=prepared.headings_json,
        )

    return db.get_source(source.id)


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
    for level, heading in _iter_markdown_headings(content):
        if level == 1:
            return heading
    return path.stem


def _extract_headings(content: str, title: str) -> list[str]:
    """Extract markdown headings for source-level retrieval."""
    headings: list[str] = []
    seen: set[str] = set()
    title_key = title.casefold()
    for _, heading in _iter_markdown_headings(content):
        if not heading or heading.casefold() == title_key:
            continue
        key = heading.casefold()
        if key in seen:
            continue
        seen.add(key)
        headings.append(heading)
    return headings


def _iter_markdown_headings(content: str) -> Iterator[tuple[int, str]]:
    """Yield ATX headings outside fenced and indented code blocks."""
    fence: tuple[str, int] | None = None
    for line in content.splitlines():
        leading_spaces = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)
            char = marker[0]
            count = len(marker)
            if fence is None:
                fence = (char, count)
            elif char == fence[0] and count >= fence[1]:
                fence = None
            continue

        if fence is not None or leading_spaces >= 4:
            continue

        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if not match:
            continue
        yield len(match.group(1)), match.group(2).strip()


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

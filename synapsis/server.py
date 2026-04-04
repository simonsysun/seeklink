"""FastMCP server — 8 tools for Synapsis PKM engine."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from synapsis.db import Database
from synapsis.embedder import Embedder
from synapsis.ingest import ingest_file
from synapsis.search import search as synapsis_search
from synapsis.watcher import run_watcher_with_retry

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    db: Database
    embedder: Embedder
    vault_root: Path
    watcher_task: asyncio.Task | None
    watcher_stop: asyncio.Event


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize DB, embedder, watcher on startup; clean up on shutdown."""
    vault_root = Path(os.environ.get("SYNAPSIS_VAULT", ".")).resolve()
    db_dir = vault_root / ".synapsis"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "synapsis.db"

    logger.info("Synapsis starting — vault: %s, db: %s", vault_root, db_path)

    db = Database(db_path)
    try:
        db.check_capabilities()
        db.init_schema()
        embedder = Embedder()

        stop_event = asyncio.Event()
        watcher_task = asyncio.create_task(
            run_watcher_with_retry(vault_root, db, embedder, stop_event)
        )

        try:
            yield AppContext(
                db=db,
                embedder=embedder,
                vault_root=vault_root,
                watcher_task=watcher_task,
                watcher_stop=stop_event,
            )
        finally:
            stop_event.set()
            watcher_task.cancel()
            try:
                await watcher_task
            except asyncio.CancelledError:
                pass
    finally:
        db.close()
        logger.info("Synapsis shut down")


mcp = FastMCP("synapsis", lifespan=app_lifespan)


def _get_ctx(ctx: Context) -> AppContext:
    """Extract AppContext from MCP request context."""
    return ctx.request_context.lifespan_context


# ── Tool 1: get_unprocessed ──────────────────────────────────────────


@mcp.tool()
async def get_unprocessed(ctx: Context) -> list[dict]:
    """List notes that are new or modified since last indexing."""
    try:
        app = _get_ctx(ctx)
        sources = await asyncio.to_thread(app.db.list_sources, "unprocessed")
        return [
            {
                "path": s.path,
                "title": s.title,
                "status": s.status,
                "fs_modified_at": s.fs_modified_at,
                "indexed_at": s.indexed_at,
            }
            for s in sources
        ]
    except Exception as e:
        return [{"error": str(e)}]


# ── Tool 2: index_note ───────────────────────────────────────────────


@mcp.tool()
async def index_note(path: str, force: bool = False, ctx: Context = None) -> dict:
    """Index a note: chunk, embed, update FTS/vector indexes, parse [[wiki-links]].

    Args:
        path: Relative path to the .md file from vault root.
        force: Re-index even if content hash is unchanged.
    """
    try:
        app = _get_ctx(ctx)
        # Normalize path: resolve ./note.md, sub/../note.md etc.
        norm_path = os.path.normpath(path)
        abs_path = (app.vault_root / norm_path).resolve()

        # Security: block path traversal (../../etc/passwd, /etc/passwd, etc.)
        if not abs_path.is_relative_to(app.vault_root):
            return {"error": f"Path escapes vault: {path}"}

        norm_path = str(abs_path.relative_to(app.vault_root))

        if not abs_path.exists():
            return {"error": f"File not found: {norm_path}"}

        if force:
            source = await asyncio.to_thread(
                app.db.get_source_by_path, norm_path
            )
            if source is not None:
                await asyncio.to_thread(
                    app.db.update_source, source.id, content_hash=None
                )

        result = await asyncio.to_thread(
            ingest_file, app.db, abs_path, app.vault_root, app.embedder
        )

        if result is None:
            return {"error": f"Failed to index: {norm_path}"}

        chunks = await asyncio.to_thread(app.db.get_chunks_by_source, result.id)
        links = await asyncio.to_thread(app.db.get_links_from, result.id)

        return {
            "source_id": result.id,
            "chunks_created": len(chunks),
            "links_parsed": len(links),
            "status": result.status,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool 3: search ───────────────────────────────────────────────────


@mcp.tool()
async def search(
    query: str, top_k: int = 10, expand: bool = False, ctx: Context = None
) -> list[dict]:
    """Search the knowledge base.

    Args:
        query: Search query (Chinese, English, or mixed).
        top_k: Number of results to return.
        expand: If true, also follow links from top results for deeper search.
    """
    try:
        app = _get_ctx(ctx)
        results = await asyncio.to_thread(
            synapsis_search, app.db, app.embedder, query, top_k=top_k, expand=expand
        )
        return [
            {
                "source_id": r.source_id,
                "path": r.path,
                "title": r.title,
                "content_preview": r.content[:200] if r.content else "",
                "rrf_score": r.score,
                "indegree": r.indegree,
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


# ── Tool 4: suggest_links ────────────────────────────────────────────


@mcp.tool()
async def suggest_links(path: str, max_suggestions: int = 5, ctx: Context = None) -> list[dict]:
    """Find notes that might be worth linking to from this note.
    Returns suggestions with relevance scores.
    Does NOT write any links — user must approve via approve_suggestion.

    Args:
        path: Relative path to the source note.
        max_suggestions: Maximum number of suggestions.
    """
    try:
        app = _get_ctx(ctx)
        source = await asyncio.to_thread(app.db.get_source_by_path, path)
        if source is None:
            return [{"error": f"Note not found: {path}"}]

        # Use title as query — concise and topical
        query = source.title or Path(path).stem
        results = await asyncio.to_thread(
            synapsis_search,
            app.db,
            app.embedder,
            query,
            top_k=max_suggestions + 10,
        )

        # Get existing outgoing links to filter
        existing_links = await asyncio.to_thread(app.db.get_links_from, source.id)
        linked_ids = {lk.target_note_id for lk in existing_links if lk.target_note_id}

        suggestions = []
        for r in results:
            # Skip self and already-linked
            if r.source_id == source.id or r.source_id in linked_ids:
                continue

            # Skip if a pending suggestion already exists for this pair
            already = await asyncio.to_thread(
                app.db.has_pending_suggestion, source.id, r.source_id
            )
            if already:
                continue

            reason = f"Relevant to '{query}' (score: {r.score:.4f})"
            sug = await asyncio.to_thread(
                app.db.add_suggestion,
                source_note_id=source.id,
                target_note_id=r.source_id,
                score=r.score,
                reason=reason,
            )
            suggestions.append(
                {
                    "suggestion_id": sug.id,
                    "target_path": r.path,
                    "target_title": r.title,
                    "score": r.score,
                    "reason": reason,
                }
            )
            if len(suggestions) >= max_suggestions:
                break

        return suggestions
    except Exception as e:
        return [{"error": str(e)}]


# ── Tool 5: approve_suggestion ───────────────────────────────────────


@mcp.tool()
async def approve_suggestion(suggestion_id: int, ctx: Context = None) -> dict:
    """Approve a link suggestion. Appends [[target]] to the source note.

    Args:
        suggestion_id: ID from suggest_links results.
    """
    try:
        app = _get_ctx(ctx)
        sug = await asyncio.to_thread(app.db.get_suggestion, suggestion_id)
        if sug is None:
            return {"error": f"Suggestion not found: {suggestion_id}"}
        if sug.status != "pending":
            return {"error": f"Suggestion already {sug.status}"}

        source = await asyncio.to_thread(app.db.get_source, sug.source_note_id)
        target = await asyncio.to_thread(app.db.get_source, sug.target_note_id)
        if source is None or target is None:
            return {"error": "Source or target note no longer exists"}

        file_path = app.vault_root / source.path
        target_stem = Path(target.path).stem

        # Write file first (watcher will reconcile if DB step fails)
        await asyncio.to_thread(_write_related_link, file_path, target_stem)

        # DB operations atomically
        def _approve_db() -> None:
            with app.db.transaction():
                app.db.approve_suggestion(suggestion_id)
                app.db.add_wiki_link(
                    source_note_id=source.id,
                    target_path=target_stem,
                    target_note_id=target.id,
                )

        await asyncio.to_thread(_approve_db)

        return {
            "status": "approved",
            "link_written": f"[[{target_stem}]]",
            "file_modified": source.path,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool 6: reject_suggestion ────────────────────────────────────────


@mcp.tool()
async def reject_suggestion(suggestion_id: int, ctx: Context = None) -> dict:
    """Reject a link suggestion.

    Args:
        suggestion_id: ID from suggest_links results.
    """
    try:
        app = _get_ctx(ctx)
        sug = await asyncio.to_thread(app.db.get_suggestion, suggestion_id)
        if sug is None:
            return {"error": f"Suggestion not found: {suggestion_id}"}
        if sug.status != "pending":
            return {"error": f"Suggestion already {sug.status}"}

        await asyncio.to_thread(app.db.reject_suggestion, suggestion_id)
        return {"status": "rejected"}
    except Exception as e:
        return {"error": str(e)}


# ── Tool 7: graph_neighbors ──────────────────────────────────────────


@mcp.tool()
async def graph_neighbors(path: str, depth: int = 1, ctx: Context = None) -> dict:
    """Show the link neighborhood of a note.

    Args:
        path: Relative path to the note.
        depth: 1 = direct links, 2 = links of links (max 3).
    """
    try:
        app = _get_ctx(ctx)
        source = await asyncio.to_thread(app.db.get_source_by_path, path)
        if source is None:
            return {"error": f"Note not found: {path}"}

        depth = max(1, min(depth, 3))
        outgoing, incoming = await asyncio.to_thread(
            _bfs_neighbors, app.db, source.id, depth
        )

        return {
            "center": {
                "path": source.path,
                "title": source.title,
                "indegree": source.indegree,
            },
            "outgoing": outgoing,
            "incoming": incoming,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool 8: status ───────────────────────────────────────────────────


@mcp.tool()
async def status(ctx: Context = None) -> dict:
    """Get system status: index stats, graph size, budget usage."""
    try:
        app = _get_ctx(ctx)
        stats = await asyncio.to_thread(app.db.get_stats)

        # Budget for current week
        week_start = _current_week_start()
        budget = await asyncio.to_thread(app.db.get_or_create_budget, week_start)
        breakdown = json.loads(budget.breakdown)

        watcher_running = (
            app.watcher_task is not None and not app.watcher_task.done()
        )

        return {
            **stats,
            "budget": {
                "week_start": budget.week_start,
                "tokens_used": budget.tokens_used,
                "breakdown": breakdown,
            },
            "watcher_running": watcher_running,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Helpers ───────────────────────────────────────────────────────────


def _write_related_link(file_path: Path, target_stem: str) -> None:
    """Append [[target]] to a note's ## Related section.

    Idempotent: skips if [[target]] already present in the Related section.
    Creates ## Related section at end if it doesn't exist.
    """
    content = file_path.read_text(encoding="utf-8")
    link_line = f"- [[{target_stem}]]"

    # Find ## Related section
    related_pattern = re.compile(r"^## Related\s*$", re.MULTILINE)
    match = related_pattern.search(content)

    if match:
        after_heading = match.end()
        if after_heading < len(content) and content[after_heading] == "\n":
            after_heading += 1

        # Find end of section (next heading or EOF)
        next_heading = re.search(r"^## ", content[after_heading:], re.MULTILINE)
        if next_heading:
            section_end = after_heading + next_heading.start()
        else:
            section_end = len(content)

        section = content[after_heading:section_end]

        # Idempotent: check if link already in Related section
        if f"[[{target_stem}]]" in section:
            return

        # Insert at end of section
        if section.rstrip():
            new_content = (
                content[:section_end].rstrip()
                + "\n"
                + link_line
                + "\n"
                + content[section_end:]
            )
        else:
            new_content = (
                content[:after_heading]
                + "\n"
                + link_line
                + "\n"
                + content[section_end:]
            )
    else:
        # No Related section — create one at end
        if content and not content.endswith("\n"):
            content += "\n"
        new_content = content + "\n## Related\n\n" + link_line + "\n"

    file_path.write_text(new_content, encoding="utf-8")


def _bfs_neighbors(
    db: Database, center_id: int, depth: int
) -> tuple[list[dict], list[dict]]:
    """BFS outgoing and incoming neighbors up to `depth` hops.

    Returns (outgoing_list, incoming_list) with path/title/indegree/depth.
    """
    # Outgoing BFS
    outgoing_results: list[tuple[int, int]] = []  # (source_id, depth)
    visited_out: set[int] = {center_id}
    frontier_out = [center_id]

    for d in range(1, depth + 1):
        next_frontier = []
        for sid in frontier_out:
            for link in db.get_links_from(sid):
                tid = link.target_note_id
                if tid is not None and tid not in visited_out:
                    visited_out.add(tid)
                    outgoing_results.append((tid, d))
                    next_frontier.append(tid)
        frontier_out = next_frontier

    # Incoming BFS
    incoming_results: list[tuple[int, int]] = []
    visited_in: set[int] = {center_id}
    frontier_in = [center_id]

    for d in range(1, depth + 1):
        next_frontier = []
        for sid in frontier_in:
            for link in db.get_links_to(sid):
                fid = link.source_note_id
                if fid not in visited_in:
                    visited_in.add(fid)
                    incoming_results.append((fid, d))
                    next_frontier.append(fid)
        frontier_in = next_frontier

    # Batch-fetch sources
    all_ids = [sid for sid, _ in outgoing_results] + [
        sid for sid, _ in incoming_results
    ]
    sources = db.get_sources_by_ids(all_ids)

    def _to_dict(sid: int, d: int) -> dict:
        s = sources.get(sid)
        if s is None:
            return {"source_id": sid, "depth": d}
        return {
            "path": s.path,
            "title": s.title,
            "indegree": s.indegree,
            "depth": d,
        }

    outgoing = [_to_dict(sid, d) for sid, d in outgoing_results]
    incoming = [_to_dict(sid, d) for sid, d in incoming_results]

    return outgoing, incoming


def _current_week_start() -> str:
    """Return ISO date of Monday of the current UTC week."""
    now = datetime.now(UTC)
    monday = now.date() - timedelta(days=now.weekday())
    return monday.isoformat()


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Run the Synapsis MCP server over stdio."""
    # Configure logging to stderr (stdio MCP needs clean stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    mcp.run(transport="stdio")


def main_sse() -> None:
    """Run the Synapsis MCP server over SSE (for remote/Docker access)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    host = os.environ.get("SYNAPSIS_SSE_HOST", "0.0.0.0")
    port = int(os.environ.get("SYNAPSIS_SSE_PORT", "8767"))

    # Override FastMCP settings for SSE mode:
    # - Bind to all interfaces so Docker containers can connect
    # - Disable DNS rebinding protection (allows Host: host.docker.internal)
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security = None

    logger.info("Synapsis SSE server starting on %s:%d", host, port)
    mcp.run(transport="sse")

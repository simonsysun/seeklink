"""Shared app bootstrap + graph/link helpers.

Used to live in server.py alongside the FastMCP server. The MCP server
was removed in v0.1.1 (see vault log `logs/rhizome-dev/2026-W16.md`).
These helpers survived because the daemon and CLI still rely on them
for db initialization and graph traversal.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from seeklink.db import Database
from seeklink.embedder import Embedder
from seeklink.models import Source

logger = logging.getLogger(__name__)


# ── DB + embedder bootstrap ─────────────────────────────────────


def init_app(vault_path: Path | None = None) -> tuple[Database, Embedder, Path]:
    """Initialize DB + embedder for a vault.

    Returns (db, embedder, vault_root). Callers are responsible for
    closing the db when done (`db.close()`).
    """
    vault_root = (
        vault_path or Path(os.environ.get("SEEKLINK_VAULT", "."))
    ).resolve()
    db_dir = vault_root / ".seeklink"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "seeklink.db"

    logger.info("SeekLink starting — vault: %s, db: %s", vault_root, db_path)

    db = Database(db_path)
    db.check_capabilities()
    db.init_schema()
    embedder = Embedder()

    return db, embedder, vault_root


# ── Graph BFS (for future backlinks / graph command) ────────────


def bfs_neighbors(
    db: Database, center_id: int, depth: int
) -> tuple[list[dict], list[dict]]:
    """BFS outgoing and incoming neighbors up to `depth` hops from center.

    Returns (outgoing_list, incoming_list) where each entry is a dict
    with path / title / indegree / depth. Used by the graph command
    (currently exposed only as an internal helper — CLI wiring is
    deferred; see vault log for the production-loop backlinks plan).
    """
    outgoing_results: list[tuple[int, int]] = []
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


# ── Link suggestion helper (used by future suggest_links CLI) ───


def write_related_link(file_path: Path, target_stem: str) -> None:
    """Append [[target]] to a note's ## Related section.

    Idempotent: skips if [[target]] already present in the Related section.
    Creates the ## Related section at end of file if it doesn't exist.
    Preserved from server.py for future suggest_links CLI support.
    """
    content = file_path.read_text(encoding="utf-8")
    link_line = f"- [[{target_stem}]]"

    related_pattern = re.compile(r"^## Related\s*$", re.MULTILINE)
    match = related_pattern.search(content)

    if match:
        after_heading = match.end()
        if after_heading < len(content) and content[after_heading] == "\n":
            after_heading += 1

        next_heading = re.search(r"^## ", content[after_heading:], re.MULTILINE)
        if next_heading:
            section_end = after_heading + next_heading.start()
        else:
            section_end = len(content)

        section = content[after_heading:section_end]

        if f"[[{target_stem}]]" in section:
            return

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
        if content and not content.endswith("\n"):
            content += "\n"
        new_content = content + "\n## Related\n\n" + link_line + "\n"

    file_path.write_text(new_content, encoding="utf-8")

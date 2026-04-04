"""File watcher — monitors vault for markdown changes and auto-indexes."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from watchfiles import Change, awatch

from seeklink.db import Database
from seeklink.embedder import Embedder
from seeklink.ingest import ingest_file

logger = logging.getLogger(__name__)


_SKIP_DIRS = {"todo", "archive"}  # non-hidden top-level dirs excluded from indexing


class MarkdownFilter:
    """Callable filter for watchfiles: only .md files, skip hidden dirs and .seeklink/."""

    def __call__(self, change: Change, path: str) -> bool:
        p = Path(path)
        # Skip hidden directories and explicitly excluded dirs
        for part in p.parts:
            if part.startswith(".") or part in _SKIP_DIRS:
                return False
        return p.suffix.lower() == ".md"


async def watch_vault(
    vault_root: Path,
    db: Database,
    embedder: Embedder,
    stop_event: asyncio.Event,
) -> None:
    """Watch vault for markdown changes and ingest them.

    Runs until stop_event is set. Catches per-file errors to keep the loop alive.
    Handles deleted files by removing them from the index.
    """
    md_filter = MarkdownFilter()

    async for changes in awatch(vault_root, watch_filter=md_filter, stop_event=stop_event):
        for change_type, path_str in changes:
            if change_type == Change.deleted:
                rel_path = str(Path(path_str).relative_to(vault_root))
                try:
                    source = await asyncio.to_thread(db.get_source_by_path, rel_path)
                    if source is not None:
                        await asyncio.to_thread(db.delete_source, source.id)
                        logger.info("Deleted from index: %s", path_str)
                except Exception:
                    logger.exception("Error handling deletion of %s", path_str)
                continue

            path = Path(path_str)
            try:
                await asyncio.to_thread(ingest_file, db, path, vault_root, embedder)
                logger.info("Auto-indexed: %s", path)
            except Exception:
                logger.exception("Error auto-indexing %s", path)


async def run_watcher_with_retry(
    vault_root: Path,
    db: Database,
    embedder: Embedder,
    stop_event: asyncio.Event,
    *,
    max_retries: int = 10,
) -> None:
    """Run watch_vault with exponential backoff on crashes.

    Retries up to max_retries times with delay 1s → 60s (capped).
    """
    delay = 1.0
    for attempt in range(max_retries):
        if stop_event.is_set():
            return
        try:
            await watch_vault(vault_root, db, embedder, stop_event)
            return  # Clean exit (stop_event was set)
        except Exception:
            logger.exception(
                "Watcher crashed (attempt %d/%d), retrying in %.0fs",
                attempt + 1,
                max_retries,
                delay,
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                return  # Stop requested during backoff
            except TimeoutError:
                pass
            delay = min(delay * 2, 60.0)

    logger.error("Watcher exhausted %d retries, giving up", max_retries)

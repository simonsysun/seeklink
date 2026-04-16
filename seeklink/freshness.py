"""Freshness check — warn when vault files have changed since last index.

Replacement for the watchdog-based file watcher. Instead of running a
background daemon to observe filesystem events, we do a bidirectional
scan on every `search` / `status` call:

  1. Disk → DB: find new .md files that haven't been indexed yet
  2. DB → Disk: find indexed entries whose file was modified or deleted

If either direction finds problems, a warning is printed to stderr
suggesting the user run `seeklink index` (or `--force`).

This is intentionally non-destructive: no automatic reindexing, no
blocking. The user always decides when to pay the reindex cost.

Known limitations (Mac-only v0.1.1 scope):
- mtime is not preserved by `git clone`, `scp`, `tar` without `-m` etc.
  → false positives, solve with `seeklink index --force`
- Single daemon socket is not keyed by vault path; multi-vault on one
  machine will route to the wrong daemon. Fix when multi-vault is real.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STALE_WARN_THRESHOLD = 1  # warn even for a single issue (catch early)
_SKIP_DIRS = {"todo", "archive"}  # match ingest._SKIP_DIRS


def check_freshness(db, vault_root: Path, warn_fp=sys.stderr) -> int:
    """Bidirectional freshness scan: disk vs index.

    Returns the total number of suspect files (new + stale + missing).
    Prints a warning to ``warn_fp`` if any are found. Typical cost at
    50-5000 files is 5-50ms (one DB query + one os.walk).
    """
    try:
        sources = db.list_sources()
    except Exception:
        logger.debug("freshness check skipped: db.list_sources failed")
        return 0

    # Build lookup structures
    indexed_paths: dict[str, datetime | None] = {}
    for src in sources:
        ts = None
        if src.indexed_at:
            try:
                ts = _parse_utc(src.indexed_at)
            except ValueError:
                pass
        indexed_paths[src.path] = ts

    # Walk vault to find ALL current .md files
    disk_paths: set[str] = set()
    for md_path in vault_root.rglob("*.md"):
        try:
            rel = md_path.relative_to(vault_root)
        except ValueError:
            continue
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts):
            continue
        disk_paths.add(str(rel))

    new_files: list[str] = []
    stale: list[str] = []
    missing: list[str] = []

    # Direction 1: disk files not yet indexed
    for rel_path in disk_paths:
        if rel_path not in indexed_paths:
            new_files.append(rel_path)

    # Direction 2: indexed files — check mtime or existence
    for src_path, indexed_at in indexed_paths.items():
        abs_path = vault_root / src_path
        try:
            stat = abs_path.stat()
        except FileNotFoundError:
            missing.append(src_path)
            continue
        except OSError:
            continue

        if indexed_at is None:
            stale.append(src_path)
            continue

        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        indexed_utc = indexed_at.replace(tzinfo=timezone.utc)
        if mtime > indexed_utc and (mtime - indexed_utc).total_seconds() > 2:
            stale.append(src_path)

    total = len(new_files) + len(stale) + len(missing)
    if total >= STALE_WARN_THRESHOLD:
        parts = []
        if new_files:
            parts.append(f"{len(new_files)} new")
        if stale:
            parts.append(f"{len(stale)} modified")
        if missing:
            parts.append(f"{len(missing)} deleted")
        print(
            f"seeklink: {' + '.join(parts)} file(s) since last index. "
            f"Run `seeklink index` to refresh.",
            file=warn_fp,
            flush=True,
        )
    return total


def _parse_utc(iso: str) -> datetime:
    """Parse an SQLite-ish timestamp into a UTC datetime."""
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        dt = datetime.fromisoformat(iso)
    return dt

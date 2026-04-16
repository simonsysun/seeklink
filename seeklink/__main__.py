"""Entry point for `python -m seeklink` and `seeklink` CLI.

Subcommands:
  daemon   — run the Unix-socket daemon (eager-loaded models, never exits)
  search   — cold-start search (in-process; daemon-free)
  index    — cold-start index ingestion
  status   — show vault / index stats and freshness warnings

The MCP server (`serve`) was removed in v0.1.1. See vault log
`logs/rhizome-dev/2026-W16.md` for rationale. Agents that used to talk
to seeklink over MCP should invoke the CLI via `subprocess` now, or
connect to the daemon socket via `seeklink.cli_client`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seeklink",
        description="Hybrid semantic search for markdown vaults.",
    )
    sub = parser.add_subparsers(dest="command")

    # daemon — Unix socket resident server
    daemon_p = sub.add_parser(
        "daemon",
        help="Run the seeklink daemon (Unix socket, eager-loaded models)",
    )
    daemon_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")

    # search
    search_p = sub.add_parser("search", help="Search the vault")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    search_p.add_argument("--tags", nargs="*", help="Filter by tags (all must match)")
    search_p.add_argument("--folder", help="Filter by folder prefix")
    search_p.add_argument("--top-k", type=int, default=10, help="Number of results")
    search_p.add_argument(
        "--title-weight",
        type=float,
        default=None,
        help=(
            "Override the title-channel RRF weight (default 1.5). "
            "Raise toward 3.0 for 'find the definitive article' queries; "
            "lower toward 0.5 for 'surface raw log moments' queries."
        ),
    )

    # index
    index_p = sub.add_parser("index", help="Index notes")
    index_p.add_argument("path", nargs="?", help="File to index (omit for full vault)")
    index_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")

    # status
    status_p = sub.add_parser("status", help="Show index status")
    status_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "daemon":
        _cmd_daemon(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "index":
        _cmd_index(args)
    elif args.command == "status":
        _cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_daemon(args: argparse.Namespace) -> None:
    from seeklink.daemon import run_daemon

    _setup_logging()
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(run_daemon(args.vault))


def _cmd_search(args: argparse.Namespace) -> None:
    from seeklink.app import init_app
    from seeklink.freshness import check_freshness
    from seeklink.search import search as seeklink_search

    _setup_logging()
    try:
        db, embedder, vault_root = init_app(args.vault)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        check_freshness(db, vault_root)
        search_kwargs = {
            "top_k": args.top_k,
            "tags": args.tags,
            "folder": args.folder,
        }
        if args.title_weight is not None:
            search_kwargs["title_weight"] = args.title_weight
        results = seeklink_search(db, embedder, args.query, **search_kwargs)
        for r in results:
            print(f"  {r.score:.4f}  {r.path}  {r.title or ''}")
            if r.content:
                preview = r.content[:120].replace("\n", " ")
                print(f"           {preview}")
        if not results:
            print("No results.")
    finally:
        db.close()


def _cmd_index(args: argparse.Namespace) -> None:
    from seeklink.app import init_app
    from seeklink.ingest import ingest_file, ingest_vault

    _setup_logging()
    try:
        db, embedder, vault_root = init_app(args.vault)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.path:
            abs_path = (vault_root / args.path).resolve()
            if not abs_path.is_relative_to(vault_root):
                print(f"Error: path escapes vault: {args.path}", file=sys.stderr)
                sys.exit(1)
            if not abs_path.exists():
                print(f"Error: File not found: {args.path}", file=sys.stderr)
                sys.exit(1)
            result = ingest_file(db, abs_path, vault_root, embedder)
            if result:
                print(f"Indexed: {result.path} ({result.status})")
            else:
                print(f"Skipped: {args.path}")
        else:
            stats = ingest_vault(db, vault_root, embedder)
            print(
                f"Done: {stats['ingested']} indexed, "
                f"{stats['unchanged']} unchanged, "
                f"{stats['skipped']} skipped, "
                f"{stats['errors']} errors"
            )
    finally:
        db.close()


def _cmd_status(args: argparse.Namespace) -> None:
    from seeklink.app import init_app
    from seeklink.freshness import check_freshness

    _setup_logging()
    try:
        db, embedder, vault_root = init_app(args.vault)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        check_freshness(db, vault_root)
        stats = db.get_stats()
        print(f"Vault:       {vault_root}")
        print(f"Notes:       {stats['notes_total']} ({stats['notes_unprocessed']} unprocessed)")
        print(f"Chunks:      {stats['chunks_total']}")
        print(f"Links:       {stats['links_total']}")
        print(f"Suggestions: {stats['suggestions_pending']} pending")
    finally:
        db.close()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


if __name__ == "__main__":
    main()

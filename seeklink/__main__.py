"""Entry point for `python -m seeklink` and `seeklink` CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seeklink",
        description="Hybrid semantic search for markdown vaults.",
    )
    sub = parser.add_subparsers(dest="command")

    # serve (default)
    serve_p = sub.add_parser("serve", help="Run MCP server (default)")
    serve_p.add_argument("--sse", action="store_true", help="Use SSE transport instead of stdio")

    # search
    search_p = sub.add_parser("search", help="Search the vault")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    search_p.add_argument("--tags", nargs="*", help="Filter by tags (all must match)")
    search_p.add_argument("--folder", help="Filter by folder prefix")
    search_p.add_argument("--top-k", type=int, default=10, help="Number of results")

    # index
    index_p = sub.add_parser("index", help="Index notes")
    index_p.add_argument("path", nargs="?", help="File to index (omit for full vault)")
    index_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")

    # status
    status_p = sub.add_parser("status", help="Show index status")
    status_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")

    args = parser.parse_args()

    # Default to serve
    if args.command is None or args.command == "serve":
        _cmd_serve(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "index":
        _cmd_index(args)
    elif args.command == "status":
        _cmd_status(args)


def _cmd_serve(args: argparse.Namespace) -> None:
    if getattr(args, "sse", False):
        from seeklink.server import main_sse
        main_sse()
    else:
        from seeklink.server import main as server_main
        server_main()


def _cmd_search(args: argparse.Namespace) -> None:
    from seeklink.search import search as seeklink_search
    from seeklink.server import init_app

    _setup_logging()
    try:
        db, embedder, vault_root = init_app(args.vault)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        results = seeklink_search(
            db, embedder, args.query,
            top_k=args.top_k,
            tags=args.tags,
            folder=args.folder,
        )
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
    from seeklink.ingest import ingest_file, ingest_vault
    from seeklink.server import init_app

    _setup_logging()
    try:
        db, embedder, vault_root = init_app(args.vault)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.path:
            abs_path = (vault_root / args.path).resolve()
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
    from seeklink.server import init_app

    _setup_logging()
    try:
        db, embedder, vault_root = init_app(args.vault)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
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

"""Entry point for `python -m seeklink` and `seeklink` CLI.

Subcommands:
  daemon   — run the Unix-socket daemon (eager-loaded models, never exits)
  search   — search the vault (daemon-first; cold-start fallback)
  index    — index notes (daemon-first; cold-start fallback)
  status   — show vault / index stats (daemon-first; cold-start fallback)

Dispatch: when `--vault` is not passed, the CLI tries the daemon socket
first (auto-spawning the daemon on first call). If the daemon is unreachable
or returns an error, the command falls back to an in-process cold-start.
Passing `--vault` always uses cold-start because the daemon is bound to a
single vault (selected via SEEKLINK_VAULT or cwd at daemon-start time);
multi-vault daemon support is tracked in TODOS.md.

Freshness warnings print only on the cold-start path today; daemon
freshness integration is tracked in TODOS.md.

The MCP server (`serve`) was removed in v0.1.1. See vault log
`logs/rhizome-dev/2026-W16.md` for rationale. Agents that used to talk
to seeklink over MCP should invoke the CLI via `subprocess` now, or
connect to the daemon socket via `seeklink.cli_client`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Model defaults — must stay in sync with seeklink/embedder.py::Embedder.MODEL_NAME
# and seeklink/reranker.py::_DEFAULT_MODEL. Duplicated here to avoid importing
# those modules (which pull in fastembed / mlx-lm) during cold CLI startup.
_DEFAULT_EMBEDDER_MODEL = "jinaai/jina-embeddings-v2-base-zh"
_DEFAULT_RERANKER_MODEL = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"


def _resolve_default_vault() -> Path:
    """Resolve the vault the CLI would use when `--vault` is not passed.

    Mirrors `seeklink.app.init_app`: honors `SEEKLINK_VAULT` env var,
    falls back to cwd, and returns a resolved absolute path.
    """
    return Path(os.environ.get("SEEKLINK_VAULT", ".")).resolve()


def _resolve_expected_models() -> tuple[str, str]:
    """Return (expected_embedder_name, expected_reranker_name) matching
    what the daemon's status endpoint would report if cold-started now.

    Mirrors Embedder/Reranker class-level attribute evaluation without
    importing those modules. Reranker reports literal ``"disabled"``
    when `SEEKLINK_RERANKER_MODEL` is empty, so we translate that here.
    """
    embedder = os.environ.get("SEEKLINK_EMBEDDER_MODEL", _DEFAULT_EMBEDDER_MODEL)
    reranker_env = os.environ.get("SEEKLINK_RERANKER_MODEL")
    if reranker_env is None:
        reranker = _DEFAULT_RERANKER_MODEL
    elif reranker_env == "":
        reranker = "disabled"
    else:
        reranker = reranker_env
    return embedder, reranker


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


def _should_use_daemon(args: argparse.Namespace) -> bool:
    """Whether to try the daemon path.

    False if --vault was explicitly passed, because the daemon is bound to a
    single vault (selected at daemon-start time) and cannot safely serve a
    different one. Multi-vault daemon support is tracked in TODOS.md.
    """
    return getattr(args, "vault", None) is None


def _try_daemon(cmd: str, daemon_args: dict) -> dict | None:
    """Call the daemon; return response dict on success, None on failure.

    Passes the caller's expected vault and model config so
    `cli_client.call()` can refuse to reuse a daemon that is bound to
    a different vault or was started with a different embedder/reranker
    (e.g. a stale daemon still running under an older
    ``SEEKLINK_VAULT`` / ``SEEKLINK_EMBEDDER_MODEL`` /
    ``SEEKLINK_RERANKER_MODEL``). Without these guards, a stale daemon
    could silently serve the wrong DB or return queries embedded with a
    model that no longer matches the on-disk vectors.

    Auto-spawn happens inside `cli_client.call()`. A None return means
    the caller should fall back to an in-process cold-start.
    """
    from seeklink import cli_client

    expected_embedder, expected_reranker = _resolve_expected_models()
    resp = cli_client.call(
        cmd,
        daemon_args,
        expected_vault=_resolve_default_vault(),
        expected_embedder=expected_embedder,
        expected_reranker=expected_reranker,
    )
    if resp.get("ok"):
        return resp
    logger.debug("Daemon call failed: %s — falling back to cold-start", resp.get("error"))
    return None


def _print_search_results(results: list) -> None:
    """Print a uniform view across daemon-dict and cold-start-SearchResult shapes."""
    for r in results:
        if isinstance(r, dict):
            score = r["score"]
            path = r["path"]
            title = r.get("title") or ""
            preview_src = r.get("content_preview", "")
        else:
            score = r.score
            path = r.path
            title = r.title or ""
            preview_src = r.content or ""
        print(f"  {score:.4f}  {path}  {title}")
        if preview_src:
            preview = preview_src[:120].replace("\n", " ")
            print(f"           {preview}")
    if not results:
        print("No results.")


def _cmd_search(args: argparse.Namespace) -> None:
    _setup_logging()

    if _should_use_daemon(args):
        daemon_args: dict = {"query": args.query, "top_k": args.top_k}
        if args.tags is not None:
            daemon_args["tags"] = args.tags
        if args.folder is not None:
            daemon_args["folder"] = args.folder
        if args.title_weight is not None:
            daemon_args["title_weight"] = args.title_weight
        resp = _try_daemon("search", daemon_args)
        if resp is not None:
            _print_search_results(resp["result"])
            return

    # Cold-start fallback (explicit --vault, or daemon unreachable)
    from seeklink.app import init_app
    from seeklink.freshness import check_freshness
    from seeklink.search import search as seeklink_search

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
        _print_search_results(results)
    finally:
        db.close()


def _cmd_index(args: argparse.Namespace) -> None:
    _setup_logging()

    if _should_use_daemon(args):
        daemon_args: dict = {}
        if args.path:
            daemon_args["path"] = args.path
        resp = _try_daemon("index", daemon_args)
        if resp is not None:
            result = resp["result"]
            if args.path:
                # single-file index: {"path": "...", "status": "indexed"|"skipped"|...}
                status = result.get("status", "?")
                if status == "skipped":
                    print(f"Skipped: {result.get('path', args.path)}")
                else:
                    print(f"Indexed: {result.get('path', args.path)} ({status})")
            else:
                stats = result
                print(
                    f"Done: {stats.get('ingested', 0)} indexed, "
                    f"{stats.get('unchanged', 0)} unchanged, "
                    f"{stats.get('skipped', 0)} skipped, "
                    f"{stats.get('errors', 0)} errors"
                )
            return

    # Cold-start fallback
    from seeklink.app import init_app
    from seeklink.ingest import ingest_file, ingest_vault

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
    """Always cold-start.

    `status` only reads SQLite stats + freshness; it never embeds or
    reranks. Routing it through the daemon would eagerly load the
    embedder and reranker (hundreds of MB, ~700 MB model download on
    first run) just to print a few numbers. `Embedder()` construction
    is cheap (the model is lazy-loaded per fastembed), so cold-start is
    fast regardless of daemon state.
    """
    _setup_logging()

    from seeklink.app import init_app
    from seeklink.freshness import check_freshness

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
        # Show which models the vault WOULD use. These are config values,
        # not load state — we don't import/instantiate the reranker here
        # to keep `status` off the mlx-lm import path.
        expected_embedder, expected_reranker = _resolve_expected_models()
        print(f"Embedder:    {expected_embedder}")
        print(f"Reranker:    {expected_reranker}")
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

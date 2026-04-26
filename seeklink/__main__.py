"""Entry point for `python -m seeklink` and `seeklink` CLI.

Subcommands:
  daemon   — run the Unix-socket daemon (eager-loaded models, never exits)
  search   — search the vault (daemon-first; cold-start fallback)
  index    — index notes (daemon-first; cold-start fallback)
  status   — show vault / index stats (always cold-start; no model load)
  get      — print a line range of a vault file (direct filesystem read)

Dispatch: when `--vault` is not passed to `search` / `index`, the CLI
tries the daemon socket first (auto-spawning the daemon on first call)
and falls back to an in-process cold-start if the daemon is unreachable.
Passing `--vault` always uses cold-start because the daemon is bound to
a single vault (selected via SEEKLINK_VAULT or cwd at daemon-start time).
`status` and `get` never route through the daemon.

Agents integrating SeekLink should invoke the CLI via `subprocess` or
connect to the daemon socket via `seeklink.cli_client` for structured
output.
"""

from __future__ import annotations

import argparse
import json
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
        "--rerank-k",
        type=int,
        default=20,
        help=(
            "Number of first-stage candidates to rerank with the cross-encoder "
            "(default: 20)"
        ),
    )
    search_p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking for this query",
    )
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
    search_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON object instead of text output",
    )

    # index
    index_p = sub.add_parser("index", help="Index notes")
    index_p.add_argument("path", nargs="?", help="File to index (omit for full vault)")
    index_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")

    # status
    status_p = sub.add_parser("status", help="Show index status")
    status_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    status_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON object instead of text output",
    )

    # get — print a line-range slice of a vault file
    get_p = sub.add_parser(
        "get",
        help="Print a line range of a vault file (agent-friendly window read)",
    )
    get_p.add_argument(
        "path",
        help=(
            "Vault-relative path, optionally with ':LINE' suffix. "
            "Examples: notes/fsrs.md, logs/2026-W15.md:42"
        ),
    )
    get_p.add_argument(
        "-l", "--lines",
        type=int,
        default=None,
        help=(
            "Number of lines to print starting at LINE (default: 100 when "
            "LINE is given, else the whole file)."
        ),
    )
    get_p.add_argument(
        "-C", "--context",
        type=int,
        default=None,
        help=(
            "Print N lines before and after LINE, grep-style. Requires a "
            "PATH:LINE argument and cannot be combined with --lines."
        ),
    )
    get_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")

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
    elif args.command == "get":
        _cmd_get(args)
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
    `cli_client.call()` can restart a daemon that is bound to a different
    vault or was started with a different embedder/reranker (e.g. a stale
    daemon still running under an older
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
    """Print a uniform view across daemon-dict and cold-start-SearchResult shapes.

    Displays `path:line` when the result has a valid line_start (> 0) so
    agents can shell out to `seeklink get path:line -l N` for a precise
    context window. Title-only matches and results without computed line
    info fall back to `path`.
    """
    for r in results:
        if isinstance(r, dict):
            score = r["score"]
            path = r["path"]
            title = r.get("title") or ""
            preview_src = r.get("content_preview", "")
            line_start = r.get("line_start", 0)
        else:
            score = r.score
            path = r.path
            title = r.title or ""
            preview_src = r.content or ""
            line_start = getattr(r, "line_start", 0)
        path_disp = f"{path}:{line_start}" if line_start > 0 else path
        print(f"  {score:.4f}  {path_disp}  {title}")
        if preview_src:
            preview = preview_src[:120].replace("\n", " ")
            print(f"           {preview}")
    if not results:
        print("No results.")


def _emit_json(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")


def _search_result_to_json(result) -> dict:
    if isinstance(result, dict):
        content_preview = result.get("content_preview", "")
        return {
            "source_id": result.get("source_id"),
            "path": result["path"],
            "title": result.get("title") or "",
            "content_preview": content_preview[:200] if content_preview else "",
            "score": result["score"],
            "indegree": result.get("indegree", 0),
            "line_start": result.get("line_start", 0),
            "line_end": result.get("line_end", 0),
        }

    return {
        "source_id": result.source_id,
        "path": result.path,
        "title": result.title or "",
        "content_preview": result.content[:200] if result.content else "",
        "score": result.score,
        "indegree": result.indegree,
        "line_start": result.line_start,
        "line_end": result.line_end,
    }


def _search_json_payload(
    *,
    query: str,
    vault: str | Path,
    top_k: int,
    rerank_k: int,
    reranking_enabled: bool,
    tags: list[str] | None,
    folder: str | None,
    embedder: str,
    reranker: str,
    results: list,
) -> dict:
    return {
        "ok": True,
        "json_schema_version": 1,
        "query": query,
        "vault": str(vault),
        "top_k": top_k,
        "reranking": {
            "enabled": reranking_enabled,
            "rerank_k": rerank_k if reranking_enabled else 0,
        },
        "filters": {
            "tags": tags or [],
            "folder": folder,
        },
        "models": {
            "embedder": embedder,
            "reranker": reranker,
        },
        "results": [_search_result_to_json(r) for r in results],
    }


def _status_json_payload(
    *,
    vault: str | Path,
    stats: dict,
    embedder: str,
    reranker: str,
    db_schema_version: int,
    freshness_count: int,
) -> dict:
    return {
        "ok": True,
        "json_schema_version": 1,
        "vault": str(vault),
        "database": {
            "schema_version": db_schema_version,
            "wal_bytes": stats["wal_bytes"],
        },
        "stats": {
            "notes_total": stats["notes_total"],
            "notes_unprocessed": stats["notes_unprocessed"],
            "chunks_total": stats["chunks_total"],
            "links_total": stats["links_total"],
            "suggestions_pending": stats["suggestions_pending"],
        },
        "freshness": {
            "checked": True,
            "fresh": freshness_count == 0,
            "suspect_files": freshness_count,
        },
        "models": {
            "embedder": embedder,
            "reranker": reranker,
        },
    }


def _cmd_search(args: argparse.Namespace) -> None:
    _setup_logging()

    if args.rerank_k < 1:
        print("Error: --rerank-k must be >= 1", file=sys.stderr)
        sys.exit(1)

    if _should_use_daemon(args):
        daemon_args: dict = {"query": args.query, "top_k": args.top_k}
        daemon_args["rerank_k"] = args.rerank_k
        if args.no_rerank:
            daemon_args["no_rerank"] = True
        if args.tags is not None:
            daemon_args["tags"] = args.tags
        if args.folder is not None:
            daemon_args["folder"] = args.folder
        if args.title_weight is not None:
            daemon_args["title_weight"] = args.title_weight
        resp = _try_daemon("search", daemon_args)
        if resp is not None:
            if getattr(args, "json", False):
                expected_embedder, expected_reranker = _resolve_expected_models()
                _emit_json(
                    _search_json_payload(
                        query=args.query,
                        vault=resp.get("vault", _resolve_default_vault()),
                        top_k=args.top_k,
                        rerank_k=args.rerank_k,
                        reranking_enabled=not args.no_rerank
                        and resp.get("reranker") != "disabled",
                        tags=args.tags,
                        folder=args.folder,
                        embedder=resp.get("embedder", expected_embedder),
                        reranker=resp.get("reranker", expected_reranker),
                        results=resp["result"],
                    )
                )
            else:
                _print_search_results(resp["result"])
            return

    # Cold-start fallback (explicit --vault, or daemon unreachable).
    # Constructs a Reranker() unless --no-rerank is requested so the
    # cold-start path produces the same default rankings as the daemon path.
    # Reranker self-disables on platforms without MLX (Linux, Intel macOS)
    # or when SEEKLINK_RERANKER_MODEL="" so construction is safe.
    from seeklink.app import init_app
    from seeklink.freshness import check_freshness
    from seeklink.reranker import Reranker
    from seeklink.search import search as seeklink_search

    try:
        db, embedder, vault_root = init_app(args.vault)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    reranker = None if args.no_rerank else Reranker()

    try:
        check_freshness(db, vault_root)
        search_kwargs = {
            "top_k": args.top_k,
            "tags": args.tags,
            "folder": args.folder,
            "reranker": reranker,
            "rerank_k": args.rerank_k,
            "vault_root": vault_root,
        }
        if args.title_weight is not None:
            search_kwargs["title_weight"] = args.title_weight
        results = seeklink_search(db, embedder, args.query, **search_kwargs)
        if getattr(args, "json", False):
            if reranker is None:
                _, reranker_name = _resolve_expected_models()
            else:
                reranker_name = "disabled" if reranker.disabled else reranker.MODEL_NAME
            _emit_json(
                _search_json_payload(
                    query=args.query,
                    vault=vault_root,
                    top_k=args.top_k,
                    rerank_k=args.rerank_k,
                    reranking_enabled=reranker is not None and not reranker.disabled,
                    tags=args.tags,
                    folder=args.folder,
                    embedder=embedder.MODEL_NAME,
                    reranker=reranker_name,
                    results=results,
                )
            )
        else:
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
        freshness_count = check_freshness(db, vault_root)
        stats = db.get_stats()
        # Show which models the vault WOULD use. These are config values,
        # not load state — we don't import/instantiate the reranker here
        # to keep `status` off the mlx-lm import path.
        expected_embedder, expected_reranker = _resolve_expected_models()
        if getattr(args, "json", False):
            _emit_json(
                _status_json_payload(
                    vault=vault_root,
                    stats=stats,
                    embedder=expected_embedder,
                    reranker=expected_reranker,
                    db_schema_version=db.SCHEMA_VERSION,
                    freshness_count=freshness_count,
                )
            )
        else:
            print(f"Vault:       {vault_root}")
            print(f"Notes:       {stats['notes_total']} ({stats['notes_unprocessed']} unprocessed)")
            print(f"Chunks:      {stats['chunks_total']}")
            print(f"Links:       {stats['links_total']}")
            print(f"Suggestions: {stats['suggestions_pending']} pending")
            print(f"Embedder:    {expected_embedder}")
            print(f"Reranker:    {expected_reranker}")
    finally:
        db.close()


def _cmd_get(args: argparse.Namespace) -> None:
    """Print a line range of a vault file (no DB lookup required).

    Usage:
        seeklink get PATH              # whole file
        seeklink get PATH:LINE          # 100 lines starting at LINE (default)
        seeklink get PATH:LINE -l N     # N lines starting at LINE
        seeklink get PATH:LINE -C N     # N lines before and after LINE
        seeklink get PATH -l N          # first N lines

    Resolves PATH against --vault (or SEEKLINK_VAULT, or cwd). Reads with
    universal-newline translation so CRLF files print as \\n-terminated.
    Exit 0 on success, 1 on missing file. Warnings to stderr for
    out-of-range LINE.
    """
    _setup_logging()

    # Parse `path:LINE` suffix
    raw = args.path
    from_line: int | None = None
    if ":" in raw:
        head, _, tail = raw.rpartition(":")
        if tail.isdigit():
            raw = head
            from_line = int(tail)
    rel_path = raw

    if args.context is not None:
        if args.context < 0:
            print("Error: --context must be >= 0", file=sys.stderr)
            sys.exit(1)
        if args.lines is not None:
            print("Error: --context cannot be combined with --lines", file=sys.stderr)
            sys.exit(1)
        if from_line is None:
            print("Error: --context requires PATH:LINE", file=sys.stderr)
            sys.exit(1)

    # Resolve vault root without initializing the DB or loading models.
    vault_root_env = os.environ.get("SEEKLINK_VAULT")
    if args.vault is not None:
        vault_root = args.vault
    elif vault_root_env:
        vault_root = Path(vault_root_env)
    else:
        vault_root = Path.cwd()
    vault_root = vault_root.resolve()

    abs_path = (vault_root / rel_path).resolve()
    # Security: reject path escapes
    try:
        abs_path.relative_to(vault_root)
    except ValueError:
        print(f"Error: path escapes vault: {rel_path}", file=sys.stderr)
        sys.exit(1)

    if not abs_path.is_file():
        print(f"Error: {rel_path} not found in {vault_root}", file=sys.stderr)
        sys.exit(1)

    try:
        text = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error: could not read {rel_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Count logical lines. `split("\n")` on text ending with "\n" produces
    # a trailing empty element that does NOT correspond to a real line —
    # drop it so `file:LINE` beyond-EOF warnings fire correctly.
    lines = text.split("\n")
    if lines and lines[-1] == "" and text.endswith("\n"):
        lines = lines[:-1]
    n_lines = len(lines)

    # Slice decision matrix:
    # - No :LINE, no -l  → whole file
    # - No :LINE, -l N   → first N lines
    # - :LINE, no -l     → 100 lines starting at LINE
    # - :LINE, -l N      → N lines starting at LINE
    # - :LINE, -C N      → N lines before and after LINE
    if from_line is None:
        start_idx = 0
        end_idx = n_lines if args.lines is None else min(args.lines, n_lines)
    else:
        if from_line < 1:
            print(
                f"Warning: LINE={from_line} < 1, clamping to 1",
                file=sys.stderr,
            )
            from_line = 1
        if from_line > n_lines:
            print(
                f"Warning: LINE={from_line} beyond EOF ({n_lines} lines); "
                "empty output",
                file=sys.stderr,
            )
            return
        hit_idx = from_line - 1
        if args.context is not None:
            start_idx = max(0, hit_idx - args.context)
            end_idx = min(hit_idx + args.context + 1, n_lines)
        else:
            start_idx = hit_idx
            n = args.lines if args.lines is not None else 100
            end_idx = min(start_idx + n, n_lines)

    out = "\n".join(lines[start_idx:end_idx])
    # Preserve trailing newline if the original file had one AND we're at EOF
    if end_idx == n_lines and text.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    sys.stdout.write(out)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


if __name__ == "__main__":
    main()

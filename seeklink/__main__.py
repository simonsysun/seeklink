"""Entry point for `python -m seeklink` and `seeklink` CLI.

Subcommands:
  daemon   — run/manage the Unix-socket daemon (eager-loaded models)
  search   — search the vault (daemon-first; cold-start fallback)
  index    — index notes (full-vault in-process; single-file daemon-first)
  status   — show vault / index stats (always cold-start; no model load)
  doctor   — diagnose runtime environment and index compatibility
  get      — print a line range of a vault file (direct filesystem read)
  mcp      — run the optional read-only MCP stdio adapter

Dispatch: when `--vault` is not passed to `search` / single-file `index`,
the CLI tries the daemon socket first (auto-spawning the daemon on first call)
and falls back to an in-process cold-start if the daemon is unreachable.
Passing `--vault` always uses cold-start because the daemon is bound to
a single vault (selected via SEEKLINK_VAULT or cwd at daemon-start time).
Full-vault `index`, `status`, and `get` never route through the daemon.

Agents integrating SeekLink can use CLI JSON, the daemon socket, or the
optional `seeklink[mcp]` stdio adapter for structured output.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from seeklink.index_config import (
    compatibility_state,
    embedding_dimension_for_embedder,
    ensure_index_compatible_for_search,
    expected_index_metadata,
    resolve_embedder_model,
)

logger = logging.getLogger(__name__)


# Reranker default is duplicated here to avoid importing seeklink.reranker,
# which pulls in mlx-lm during cold CLI startup.
_DEFAULT_RERANKER_MODEL = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"
_NO_DAEMON_ENV = "SEEKLINK_NO_DAEMON"


def _parse_rerank_k(raw: str) -> int | str:
    if raw == "auto":
        return raw
    try:
        value = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--rerank-k must be a positive integer or 'auto'"
        ) from e
    if value < 1:
        raise argparse.ArgumentTypeError("--rerank-k must be >= 1")
    return value


def _validate_rerank_k(value: int | str) -> None:
    if value == "auto":
        return
    if not isinstance(value, int) or value < 1:
        print(
            "Error: --rerank-k must be a positive integer or 'auto'",
            file=sys.stderr,
        )
        sys.exit(1)


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().casefold() not in {"", "0", "false", "no", "off"}


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
    embedder = resolve_embedder_model()
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
        help="Manage the seeklink daemon (Unix socket, eager-loaded models)",
    )
    daemon_p.add_argument(
        "daemon_action",
        nargs="?",
        choices=["run", "status", "stop", "restart", "pid"],
        default="run",
        help=(
            "Daemon action. Default: run. Use status/stop/restart/pid for "
            "lifecycle controls."
        ),
    )
    daemon_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    daemon_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON for daemon lifecycle commands",
    )

    # search
    search_p = sub.add_parser("search", help="Search the vault")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    search_p.add_argument("--tags", nargs="*", help="Filter by tags (all must match)")
    search_p.add_argument("--folder", help="Filter by folder prefix")
    search_p.add_argument("--top-k", type=int, default=10, help="Number of results")
    search_p.add_argument(
        "--rerank-k",
        type=_parse_rerank_k,
        default="auto",
        help=(
            "Number of first-stage candidates to rerank with the cross-encoder "
            "or 'auto' for query-sensitive routing (default: auto)"
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
    search_p.add_argument(
        "--no-daemon",
        action="store_true",
        help="Force an in-process search instead of using the daemon",
    )

    # index
    index_p = sub.add_parser("index", help="Index notes")
    index_p.add_argument("path", nargs="?", help="File to index (omit for full vault)")
    index_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    index_p.add_argument(
        "--no-daemon",
        action="store_true",
        help="Force in-process indexing instead of using the daemon",
    )

    # status
    status_p = sub.add_parser("status", help="Show index status")
    status_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    status_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON object instead of text output",
    )

    # doctor
    doctor_p = sub.add_parser("doctor", help="Diagnose environment and index health")
    doctor_p.add_argument("--vault", type=Path, help="Vault path (default: cwd)")
    doctor_p.add_argument(
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

    # mcp — read-only stdio adapter for MCP clients
    mcp_p = sub.add_parser(
        "mcp",
        help="Run the optional read-only MCP stdio adapter",
    )
    mcp_p.add_argument(
        "--vault",
        type=Path,
        help="Vault path. Defaults to SEEKLINK_VAULT; cwd is not used for MCP.",
    )

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
    elif args.command == "doctor":
        _cmd_doctor(args)
    elif args.command == "get":
        _cmd_get(args)
    elif args.command == "mcp":
        _cmd_mcp(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_daemon(args: argparse.Namespace) -> None:
    _setup_logging()

    action = getattr(args, "daemon_action", "run")
    if action == "run":
        from seeklink.daemon import run_daemon

        logging.getLogger().setLevel(logging.INFO)
        sys.exit(run_daemon(args.vault))
    if action == "status":
        _cmd_daemon_status(args)
        return
    if action == "stop":
        _cmd_daemon_stop(args)
        return
    if action == "restart":
        _cmd_daemon_restart(args)
        return
    if action == "pid":
        _cmd_daemon_pid(args)
        return

    print(f"Error: unknown daemon action: {action}", file=sys.stderr)
    sys.exit(1)


def _cmd_mcp(args: argparse.Namespace) -> None:
    """Run the optional read-only MCP stdio adapter."""
    _setup_logging()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from seeklink.mcp_server import run_mcp_server
            from seeklink.mcp_services import ServiceError
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("mcp"):
            print(
                'Error: MCP support is not installed. Install with `pip install "seeklink[mcp]"`.',
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    try:
        run_mcp_server(args.vault)
    except ServiceError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        sys.exit(1)


def _daemon_not_running_payload() -> dict[str, Any]:
    from seeklink import cli_client

    return {
        "ok": True,
        "json_schema_version": 1,
        "daemon": {
            "running": False,
            "socket": str(cli_client.SOCKET_PATH),
        },
    }


def _daemon_running_payload(status_response: dict[str, Any]) -> dict[str, Any]:
    from seeklink import cli_client

    result = status_response.get("result") or {}
    return {
        "ok": True,
        "json_schema_version": 1,
        "daemon": {
            "running": True,
            "pid": result.get("pid"),
            "socket": result.get("socket") or str(cli_client.SOCKET_PATH),
            "vault": result.get("vault"),
            "embedder": result.get("embedder"),
            "reranker": result.get("reranker"),
            "started_at": result.get("started_at"),
            "uptime_s": result.get("uptime_s"),
            "idle_s": result.get("idle_s"),
            "idle_timeout_s": result.get("idle_timeout_s"),
            "requests_served": result.get("requests_served"),
            "rss_bytes": result.get("rss_bytes"),
        },
    }


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GB"


def _format_seconds(value: float | int | None) -> str:
    if value is None:
        return "never"
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.1f}m"
    return f"{value / 3600:.1f}h"


def _cmd_daemon_status(args: argparse.Namespace) -> None:
    from seeklink import cli_client

    status = cli_client.probe_status()
    if not status.get("ok"):
        payload = _daemon_not_running_payload()
        if getattr(args, "json", False):
            _emit_json(payload)
        else:
            print("Daemon: not running")
            print(f"Socket: {payload['daemon']['socket']}")
        return

    payload = _daemon_running_payload(status)
    if getattr(args, "json", False):
        _emit_json(payload)
        return

    daemon = payload["daemon"]
    print("Daemon: running")
    print(f"PID:    {daemon.get('pid') or 'unknown'}")
    print(f"Socket: {daemon.get('socket')}")
    print(f"Vault:  {daemon.get('vault')}")
    print(f"Embedder: {daemon.get('embedder')}")
    print(f"Reranker: {daemon.get('reranker')}")
    print(f"Uptime: {_format_seconds(daemon.get('uptime_s'))}")
    print(f"Idle:   {_format_seconds(daemon.get('idle_s'))}")
    print(f"Idle timeout: {_format_seconds(daemon.get('idle_timeout_s'))}")
    print(f"Memory: {_format_bytes(daemon.get('rss_bytes'))}")


def _cmd_daemon_stop(args: argparse.Namespace) -> None:
    from seeklink import cli_client

    result = cli_client.stop_daemon()
    ok = bool(result.get("ok"))
    status = (result.get("result") or {}).get("status")
    if getattr(args, "json", False):
        _emit_json(
            {
                "ok": ok,
                "json_schema_version": 1,
                "daemon": {
                    "running": False if ok else None,
                    "socket": str(cli_client.SOCKET_PATH),
                    "status": status,
                },
                **({} if ok else {"error": result.get("error", "unknown error")}),
            }
        )
    else:
        if ok:
            if status == "not_running":
                print("Daemon: not running")
            else:
                print("Daemon: stopped")
        else:
            print(f"Error: {result.get('error', 'unknown error')}", file=sys.stderr)

    if not ok:
        sys.exit(1)


def _cmd_daemon_restart(args: argparse.Namespace) -> None:
    from seeklink import cli_client

    stopped = cli_client.stop_daemon()
    if not stopped.get("ok"):
        if getattr(args, "json", False):
            _emit_json(
                {
                    "ok": False,
                    "json_schema_version": 1,
                    "error": stopped.get("error", "unknown error"),
                }
            )
        else:
            print(f"Error: {stopped.get('error', 'unknown error')}", file=sys.stderr)
        sys.exit(1)

    started = cli_client.start_daemon(vault=args.vault)
    if not started.get("ok"):
        if getattr(args, "json", False):
            _emit_json(
                {
                    "ok": False,
                    "json_schema_version": 1,
                    "error": started.get("error", "unknown error"),
                }
            )
        else:
            print(f"Error: {started.get('error', 'unknown error')}", file=sys.stderr)
        sys.exit(1)

    payload = _daemon_running_payload(started)
    if getattr(args, "json", False):
        _emit_json(payload)
    else:
        print(f"Daemon: restarted pid={payload['daemon'].get('pid')}")


def _cmd_daemon_pid(args: argparse.Namespace) -> None:
    from seeklink import cli_client

    status = cli_client.probe_status()
    if not status.get("ok"):
        if getattr(args, "json", False):
            _emit_json(_daemon_not_running_payload())
        else:
            print("Error: daemon not running", file=sys.stderr)
        sys.exit(1)

    payload = _daemon_running_payload(status)
    if getattr(args, "json", False):
        _emit_json(payload)
        return
    pid = payload["daemon"].get("pid")
    if pid is None:
        print("Error: daemon status did not report a pid", file=sys.stderr)
        sys.exit(1)
    print(pid)


def _should_use_daemon(args: argparse.Namespace) -> bool:
    """Whether to try the daemon path.

    False if --vault was explicitly passed, because the daemon is bound to a
    single vault (selected at daemon-start time) and cannot safely serve a
    different one. Multi-vault daemon support is tracked in TODOS.md.
    """
    if getattr(args, "no_daemon", False) or _env_flag(_NO_DAEMON_ENV):
        return False
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
    rerank_k: int | str,
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
    index_metadata: dict[str, str],
    index_compatibility: dict,
) -> dict:
    return {
        "ok": True,
        "json_schema_version": 1,
        "vault": str(vault),
        "database": {
            "schema_version": db_schema_version,
            "wal_bytes": stats["wal_bytes"],
        },
        "index": {
            "metadata": index_metadata,
            "compatibility": index_compatibility,
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


def _add_doctor_check(
    checks: list[dict],
    *,
    name: str,
    ok: bool,
    detail: str,
    required: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "ok": ok,
            "required": required,
            "detail": detail,
        }
    )


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Lightweight diagnostics. Does not download or load ML models."""
    _setup_logging()

    checks: list[dict] = []
    _add_doctor_check(
        checks,
        name="python",
        ok=sys.version_info >= (3, 11),
        detail=sys.version.split()[0],
    )
    _add_doctor_check(
        checks,
        name="sqlite",
        ok=sqlite3.sqlite_version_info >= (3, 45),
        detail=sqlite3.sqlite_version,
    )
    mlx_installed = importlib.util.find_spec("mlx_lm") is not None
    _add_doctor_check(
        checks,
        name="mlx_lm",
        ok=mlx_installed,
        detail="installed" if mlx_installed else "not installed",
        required=False,
    )

    from seeklink import cli_client

    daemon_status = cli_client.probe_status()
    if daemon_status.get("ok"):
        daemon_payload = _daemon_running_payload(daemon_status)["daemon"]
        daemon_detail = (
            f"running pid={daemon_payload.get('pid') or 'unknown'}, "
            f"vault={daemon_payload.get('vault')}, "
            f"memory={_format_bytes(daemon_payload.get('rss_bytes'))}"
        )
    else:
        daemon_payload = _daemon_not_running_payload()["daemon"]
        daemon_detail = "not running"
    _add_doctor_check(
        checks,
        name="daemon",
        ok=True,
        detail=daemon_detail,
        required=False,
    )

    stats: dict | None = None
    index_compatibility: dict | None = None
    vault_root = _resolve_default_vault() if args.vault is None else args.vault.resolve()
    db = None
    try:
        from seeklink.app import init_app

        db, embedder, vault_root = init_app(args.vault)
        stats = db.get_stats()
        expected_metadata = expected_index_metadata(
            embedder.MODEL_NAME,
            embedding_dimension_for_embedder(embedder),
        )
        index_compatibility = compatibility_state(
            stored=db.get_index_metadata(),
            expected=expected_metadata,
            chunks_total=stats["chunks_total"],
            vector_dimension=db.get_vector_dimension(),
        )
        _add_doctor_check(
            checks,
            name="database",
            ok=True,
            detail=str(vault_root / ".seeklink" / "seeklink.db"),
        )
        _add_doctor_check(
            checks,
            name="index_compatibility",
            ok=bool(index_compatibility["compatible"]),
            detail=str(index_compatibility["state"]),
        )
    except Exception as e:
        _add_doctor_check(
            checks,
            name="database",
            ok=False,
            detail=str(e),
        )
    finally:
        if db is not None:
            db.close()

    ok = all(check["ok"] for check in checks if check["required"])
    if getattr(args, "json", False):
        _emit_json(
            {
                "ok": ok,
                "json_schema_version": 1,
                "vault": str(vault_root),
                "checks": checks,
                "stats": stats or {},
                "index": {
                    "compatibility": index_compatibility or {},
                },
                "daemon": daemon_payload,
            }
        )
    else:
        for check in checks:
            label = "OK" if check["ok"] else ("WARN" if not check["required"] else "FAIL")
            print(f"{label}: {check['name']} — {check['detail']}")

    if not ok:
        sys.exit(1)


def _cmd_search(args: argparse.Namespace) -> None:
    _setup_logging()

    _validate_rerank_k(args.rerank_k)

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
        ensure_index_compatible_for_search(
            db,
            embedder_model=embedder.MODEL_NAME,
            embedding_dim=embedding_dimension_for_embedder(embedder),
        )
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


def _cmd_index(args: argparse.Namespace) -> None:
    _setup_logging()

    if args.path and _should_use_daemon(args):
        daemon_args: dict = {}
        daemon_args["path"] = args.path
        resp = _try_daemon("index", daemon_args)
        if resp is not None:
            result = resp["result"]
            # single-file index: {"path": "...", "status": "indexed"|"skipped"|...}
            status = result.get("status", "?")
            if status == "skipped":
                print(f"Skipped: {result.get('path', args.path)}")
            else:
                print(f"Indexed: {result.get('path', args.path)} ({status})")
            return

    # Cold-start fallback. Full-vault indexing always stays on this path so
    # progress can stream to stderr without expanding the daemon protocol.
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
            stats = ingest_vault(
                db,
                vault_root,
                embedder,
                progress=_IndexProgressPrinter(),
            )
            print(
                f"Done: {stats['ingested']} indexed, "
                f"{stats['unchanged']} unchanged, "
                f"{stats['skipped']} skipped, "
                f"{stats['errors']} errors"
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


class _IndexProgressPrinter:
    """Time-throttled progress renderer for full-vault cold-start indexing."""

    def __init__(self, *, min_interval_s: float = 15.0) -> None:
        self._min_interval_s = min_interval_s
        self._last_progress_at = 0.0

    def __call__(self, event: str, payload: dict) -> None:
        now = time.monotonic()

        if event == "scan_start":
            self._force_print(now, "Scanning vault...")
        elif event == "scan_done":
            self._force_print(
                now,
                f"Found {payload.get('files_total', 0)} markdown files."
            )
        elif event == "prepare_progress":
            if self._due(now):
                self._print(
                    "Preparing files: "
                    f"{payload.get('files_seen', 0)}/"
                    f"{payload.get('files_total', 0)}, "
                    f"{payload.get('chunks_to_embed', 0)} chunks so far, "
                    f"{self._fmt_elapsed(payload.get('elapsed_s', 0.0))} elapsed"
                )
                self._last_progress_at = now
        elif event == "prepare_done":
            self._force_print(
                now,
                "Prepared "
                f"{payload.get('files_to_index', 0)} files, "
                f"{payload.get('chunks_to_embed', 0)} chunks to embed "
                f"({payload.get('unchanged', 0)} unchanged, "
                f"{payload.get('skipped', 0)} skipped, "
                f"{payload.get('errors', 0)} errors)."
            )
        elif event == "embed_start":
            chunks_total = payload.get("chunks_total", 0)
            if chunks_total:
                self._force_print(
                    now,
                    "Embedding "
                    f"{chunks_total} chunks in "
                    f"{payload.get('batches_total', 0)} batches "
                    f"(batch size {payload.get('batch_size', 0)})..."
                )
            else:
                self._force_print(now, "No new chunks to embed.")
        elif event == "embed_progress":
            chunks_done = payload.get("chunks_done", 0)
            chunks_total = payload.get("chunks_total", 0)
            if chunks_done >= chunks_total or self._due(now):
                self._print(
                    "Embedding chunks: "
                    f"{chunks_done}/{chunks_total}, "
                    f"{payload.get('batches_done', 0)}/"
                    f"{payload.get('batches_total', 0)} batches, "
                    f"{self._fmt_elapsed(payload.get('elapsed_s', 0.0))} elapsed"
                )
                self._last_progress_at = now
        elif event == "embed_done":
            chunks_total = payload.get("chunks_total", 0)
            if chunks_total:
                self._force_print(
                    now,
                    "Embedding complete: "
                    f"{chunks_total} chunks, "
                    f"{self._fmt_elapsed(payload.get('elapsed_s', 0.0))} elapsed."
                )
        elif event == "write_start":
            files_to_index = payload.get("files_to_index", 0)
            if files_to_index:
                self._force_print(
                    now,
                    f"Writing index for {files_to_index} files...",
                )
        elif event == "write_progress":
            files_written = payload.get("files_written", 0)
            files_to_index = payload.get("files_to_index", 0)
            if files_written >= files_to_index or self._due(now):
                self._print(
                    "Writing index: "
                    f"{files_written}/{files_to_index} files, "
                    f"{self._fmt_elapsed(payload.get('elapsed_s', 0.0))} elapsed"
                )
                self._last_progress_at = now

    def _due(self, now: float) -> bool:
        return now - self._last_progress_at >= self._min_interval_s

    @staticmethod
    def _print(message: str) -> None:
        print(message, file=sys.stderr)

    def _force_print(self, now: float, message: str) -> None:
        self._print(message)
        self._last_progress_at = now

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{seconds / 60:.1f}m"


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
        expected_metadata = expected_index_metadata(expected_embedder)
        index_metadata = db.get_index_metadata()
        index_compatibility = compatibility_state(
            stored=index_metadata,
            expected=expected_metadata,
            chunks_total=stats["chunks_total"],
            vector_dimension=db.get_vector_dimension(),
        )
        if getattr(args, "json", False):
            _emit_json(
                _status_json_payload(
                    vault=vault_root,
                    stats=stats,
                    embedder=expected_embedder,
                    reranker=expected_reranker,
                    db_schema_version=db.SCHEMA_VERSION,
                    freshness_count=freshness_count,
                    index_metadata=index_metadata,
                    index_compatibility=index_compatibility,
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
            print(f"Index:       {index_compatibility['state']}")
            if not index_compatibility["compatible"]:
                print(
                    "Warning: index config mismatch; run `seeklink index` "
                    "to rebuild derived vectors.",
                    file=sys.stderr,
                )
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

"""Unix socket daemon — eager-loads models once, serves search requests.

This is the "resident" mode for seeklink. The first CLI invocation on a
cold machine spawns this process; it loads the embedder and reranker,
binds a Unix socket at ~/.rhizome/seeklink.sock, and serves requests
until SIGTERM / SIGINT, protocol shutdown, or idle timeout.

Protocol: length-prefixed JSON over the socket. Each request is one
connection; the daemon reads a 4-byte big-endian length, then that
many bytes of UTF-8 JSON, processes the request, and writes a
length-prefixed JSON response before closing the connection.

Request schema:
    {"cmd": "search" | "status" | "index" | "shutdown", "args": {...}}

Response schema:
    {"ok": true,  "result": ...}   on success
    {"ok": false, "error": "..."}  on failure
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

SOCKET_PATH = Path.home() / ".rhizome" / "seeklink.sock"
MAX_MESSAGE_BYTES = 10_000_000  # 10MB — generous for any realistic payload
DEFAULT_IDLE_TIMEOUT_SECONDS = 900
_IDLE_TIMEOUT_ENV = "SEEKLINK_DAEMON_IDLE_TIMEOUT"


def _parse_idle_timeout(raw: str | None = None) -> int | None:
    """Parse the daemon idle timeout env value.

    Returns seconds, or None for "never auto-exit". Invalid values fall back
    to the default so one bad env var does not disable daemon startup.
    """
    if raw is None:
        raw = os.environ.get(_IDLE_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_IDLE_TIMEOUT_SECONDS

    value = raw.strip().casefold()
    if value in {"", "0", "off", "false", "no", "never"}:
        return None
    try:
        seconds = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %ds",
            _IDLE_TIMEOUT_ENV,
            raw,
            DEFAULT_IDLE_TIMEOUT_SECONDS,
        )
        return DEFAULT_IDLE_TIMEOUT_SECONDS
    if seconds < 1:
        logger.warning(
            "Invalid %s=%r; using default %ds",
            _IDLE_TIMEOUT_ENV,
            raw,
            DEFAULT_IDLE_TIMEOUT_SECONDS,
        )
        return DEFAULT_IDLE_TIMEOUT_SECONDS
    return seconds


def _rss_bytes(pid: int | None = None) -> int | None:
    """Best-effort resident memory for daemon diagnostics."""
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    try:
        return int(out.strip()) * 1024
    except (TypeError, ValueError):
        return None


def _daemon_metadata(state: dict[str, Any] | None) -> dict[str, Any]:
    """Return daemon lifecycle metadata for status/doctor consumers."""
    now = time.monotonic()
    if state is None:
        started_at = time.time()
        started_monotonic = now
        last_activity = now
        idle_timeout = _parse_idle_timeout()
        requests_served = 0
    else:
        started_at = state["started_at"]
        started_monotonic = state["started_monotonic"]
        last_activity = state["last_activity"]
        idle_timeout = state["idle_timeout_s"]
        requests_served = state["requests_served"]

    return {
        "pid": os.getpid(),
        "socket": str(SOCKET_PATH),
        "started_at": started_at,
        "uptime_s": max(0.0, now - started_monotonic),
        "idle_s": max(0.0, now - last_activity),
        "idle_timeout_s": idle_timeout,
        "requests_served": requests_served,
        "rss_bytes": _rss_bytes(),
    }


def _idle_timed_out(state: dict[str, Any], now: float | None = None) -> bool:
    timeout = state["idle_timeout_s"]
    if timeout is None:
        return False
    now = time.monotonic() if now is None else now
    return now - state["last_activity"] >= timeout


def run_daemon(vault_path: Path | None = None) -> int:
    """Run the seeklink daemon. Blocks until SIGTERM / SIGINT.

    Returns the exit code to use.
    """
    from seeklink.db import Database
    from seeklink.embedder import Embedder
    from seeklink.reranker import Reranker

    # Setup socket directory; check for an already-running daemon before
    # touching the socket file. This prevents a concurrent startup from
    # unlinking a live daemon's socket and orphaning it.
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(1.0)
            probe.connect(str(SOCKET_PATH))
            probe.close()
            logger.info(
                "Another daemon is already running at %s — exiting",
                SOCKET_PATH,
            )
            return 0
        except (ConnectionRefusedError, OSError):
            # Socket is stale (crashed daemon) — safe to remove
            try:
                SOCKET_PATH.unlink()
            except OSError as e:
                logger.warning("Failed to remove stale socket: %s", e)

    # Resolve vault and initialize database
    vault_root = (
        vault_path or Path(os.environ.get("SEEKLINK_VAULT", "."))
    ).resolve()
    db_dir = vault_root / ".seeklink"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "seeklink.db"

    logger.info(
        "Daemon starting — vault: %s, socket: %s", vault_root, SOCKET_PATH
    )

    db = Database(db_path)
    db.check_capabilities()
    db.init_schema()

    embedder = Embedder()
    reranker = Reranker()

    # Eager-load embedder so the first real query is fast
    logger.info("Warming up embedder...")
    try:
        embedder.embed_query("warmup")
    except Exception:
        logger.exception("Embedder warmup failed")

    # Only warm up reranker if explicitly enabled — otherwise we waste
    # seconds loading a large model that will never be called.
    if not reranker.disabled:
        logger.info("Warming up reranker (%s)...", reranker.MODEL_NAME)
        try:
            reranker.rerank("warmup", ["warmup passage"])
        except Exception:
            logger.exception("Reranker warmup failed (continuing)")

    logger.info(
        "Daemon ready — embedder: %s, reranker: %s",
        embedder.MODEL_NAME,
        "disabled" if reranker.disabled else reranker.MODEL_NAME,
    )

    idle_timeout = _parse_idle_timeout()
    state: dict[str, Any] = {
        "started_at": time.time(),
        "started_monotonic": time.monotonic(),
        "last_activity": time.monotonic(),
        "idle_timeout_s": idle_timeout,
        "requests_served": 0,
    }

    # Bind Unix socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCKET_PATH))
    server.listen(8)
    server.settimeout(1.0)

    shutdown_requested = {"flag": False}

    def _shutdown(signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received: %d", signum)
        shutdown_requested["flag"] = True
        try:
            server.close()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    exit_code = 0
    try:
        while not shutdown_requested["flag"]:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                if _idle_timed_out(state):
                    logger.info(
                        "Daemon idle timeout reached (%ss); shutting down",
                        idle_timeout,
                    )
                    shutdown_requested["flag"] = True
                continue
            except OSError:
                break  # socket closed during shutdown
            try:
                _handle_connection(
                    conn,
                    db,
                    embedder,
                    reranker,
                    vault_root,
                    daemon_state=state,
                    request_shutdown=lambda: shutdown_requested.__setitem__(
                        "flag", True
                    ),
                )
                state["requests_served"] += 1
                state["last_activity"] = time.monotonic()
            except Exception:
                logger.exception("Error handling connection")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        logger.exception("Daemon crashed")
        exit_code = 1
    finally:
        try:
            server.close()
        except Exception:
            pass
        try:
            if SOCKET_PATH.exists():
                SOCKET_PATH.unlink()
        except Exception:
            pass
        db.close()
        logger.info("Daemon stopped.")

    return exit_code


def _handle_connection(
    conn: socket.socket,
    db: Any,
    embedder: Any,
    reranker: Any,
    vault_root: Path,
    daemon_state: dict[str, Any] | None = None,
    request_shutdown: Callable[[], None] | None = None,
) -> None:
    """Handle a single client connection: read request, execute, send response."""
    from seeklink.index_config import (
        compatibility_state,
        embedding_dimension_for_embedder,
        ensure_index_compatible_for_search,
        expected_index_metadata,
    )
    from seeklink.ingest import ingest_file, ingest_vault
    from seeklink.search import search as do_search

    data = _recv_framed(conn)
    if data is None:
        return

    try:
        req = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as e:
        _send_error(conn, f"invalid JSON: {e}")
        return

    cmd = req.get("cmd")
    args = req.get("args") or {}
    should_shutdown = False

    try:
        if cmd == "search":
            query = args["query"]
            ensure_index_compatible_for_search(
                db,
                embedder_model=embedder.MODEL_NAME,
                embedding_dim=embedding_dimension_for_embedder(embedder),
            )
            results = do_search(
                db,
                embedder,
                query,
                top_k=args.get("top_k", 10),
                expand=args.get("expand", False),
                tags=args.get("tags"),
                folder=args.get("folder"),
                title_weight=args.get("title_weight", 1.5),
                reranker=None if args.get("no_rerank") else reranker,
                rerank_k=args.get("rerank_k", "auto"),
                vault_root=vault_root,
            )
            response = {
                "ok": True,
                "vault": str(vault_root),
                "embedder": embedder.MODEL_NAME,
                "reranker": (
                    "disabled" if reranker.disabled else reranker.MODEL_NAME
                ),
                "result": [
                    {
                        "source_id": r.source_id,
                        "path": r.path,
                        "title": r.title,
                        "content_preview": r.content[:200] if r.content else "",
                        "score": r.score,
                        "indegree": r.indegree,
                        "line_start": r.line_start,
                        "line_end": r.line_end,
                    }
                    for r in results
                ],
            }

        elif cmd == "status":
            stats = db.get_stats()
            expected_metadata = expected_index_metadata(embedder.MODEL_NAME)
            index_metadata = db.get_index_metadata()
            index_compatibility = compatibility_state(
                stored=index_metadata,
                expected=expected_metadata,
                chunks_total=stats["chunks_total"],
                vector_dimension=db.get_vector_dimension(),
            )
            response = {
                "ok": True,
                "result": {
                    **stats,
                    "vault": str(vault_root),
                    "reranker": (
                        "disabled" if reranker.disabled else reranker.MODEL_NAME
                    ),
                    "embedder": embedder.MODEL_NAME,
                    "index_metadata": index_metadata,
                    "index_compatibility": index_compatibility,
                    **_daemon_metadata(daemon_state),
                },
            }

        elif cmd == "index":
            path = args.get("path")
            if path:
                abs_path = (vault_root / path).resolve()
                if not abs_path.is_relative_to(vault_root):
                    _send_error(conn, f"path escapes vault: {path}")
                    return
                if not abs_path.exists():
                    _send_error(conn, f"file not found: {path}")
                    return
                result = ingest_file(db, abs_path, vault_root, embedder)
                response = {
                    "ok": True,
                    "result": {
                        "path": path,
                        "status": result.status if result else "skipped",
                    },
                }
            else:
                stats = ingest_vault(db, vault_root, embedder)
                response = {"ok": True, "result": stats}

        elif cmd == "shutdown":
            response = {"ok": True, "result": {"status": "shutting_down"}}
            should_shutdown = True

        else:
            _send_error(conn, f"unknown command: {cmd}")
            return

        _send_framed(conn, json.dumps(response).encode("utf-8"))
        if should_shutdown and request_shutdown is not None:
            request_shutdown()

    except Exception as e:
        logger.exception("Command %r failed", cmd)
        _send_error(conn, f"{type(e).__name__}: {e}")


def _recv_framed(conn: socket.socket) -> bytes | None:
    """Receive a length-prefixed message. Returns None on EOF."""
    header = b""
    while len(header) < 4:
        chunk = conn.recv(4 - len(header))
        if not chunk:
            return None
        header += chunk
    length = int.from_bytes(header, "big")
    if length <= 0 or length > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message length: {length}")
    data = b""
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _send_framed(conn: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed message."""
    header = len(payload).to_bytes(4, "big")
    conn.sendall(header + payload)


def _send_error(conn: socket.socket, msg: str) -> None:
    """Send a framed error response."""
    try:
        _send_framed(
            conn, json.dumps({"ok": False, "error": msg}).encode("utf-8")
        )
    except Exception:
        pass

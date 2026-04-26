"""Thin Unix-socket client for the seeklink daemon.

Tries to reach the daemon at ~/.rhizome/seeklink.sock. If the socket is
missing or the daemon is unreachable, spawns a new daemon subprocess
(detached) and waits for it to come up before retrying. Any failure to
spawn or connect is reported back as a failed response — callers can
fall back to a cold-start in-process execution if they prefer.

Vault-binding guard: the daemon is single-vault (bound at startup via
SEEKLINK_VAULT or cwd). If a caller passes expected config to ``call()``,
the client first probes the daemon's reported status. On mismatch it asks
the stale daemon to shut down, waits for the socket to clear, and then
auto-spawns a daemon under the caller's current env/cwd.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Known limitation: one socket per machine. If multiple vaults need
# concurrent daemons, hash the vault path into the socket name:
#   SOCKET_PATH = Path.home() / ".rhizome" / f"seeklink-{hash(vault)}.sock"
# Deferred until multi-vault becomes a real use case.
SOCKET_PATH = Path.home() / ".rhizome" / "seeklink.sock"
SPAWN_WAIT_SECONDS = 60.0  # cold start includes model load, give it time
SHUTDOWN_WAIT_SECONDS = 10.0


def call(
    cmd: str,
    args: dict[str, Any],
    *,
    expected_vault: Path | None = None,
    expected_embedder: str | None = None,
    expected_reranker: str | None = None,
) -> dict[str, Any]:
    """Send a command to the daemon. Auto-spawns daemon if unreachable.

    If any ``expected_*`` is provided, probes the daemon's reported
    status first and restarts the daemon if its vault or model config
    does not match what the caller expects. Callers should pass these
    whenever the intended config is inferred from env/cwd rather than
    an explicit flag — this prevents a stale daemon from:

    - serving or mutating the wrong vault's database (vault mismatch), or
    - returning results computed with the wrong embedder/reranker model
      (config mismatch), which can silently corrupt rankings and, if
      the embedder's vector width changes, make ``search_vec`` fail.

    On mismatch, this function requests daemon shutdown and retries
    through the normal spawn path. If restart fails, callers can still
    fall back to an in-process cold-start.

    Returns the daemon's JSON response as a dict. Never raises — on
    failure returns ``{"ok": False, "error": "..."}``.
    """
    need_probe = (
        expected_vault is not None
        or expected_embedder is not None
        or expected_reranker is not None
    )
    if need_probe:
        probe = _call_once_with_spawn("status", {})
        if not probe.get("ok"):
            return probe  # daemon unreachable / spawn failed
        result = probe.get("result") or {}

        mismatch = _config_mismatch_error(
            result,
            expected_vault=expected_vault,
            expected_embedder=expected_embedder,
            expected_reranker=expected_reranker,
        )
        if mismatch is not None:
            return _restart_and_retry(cmd, args, mismatch)

        # All checks passed — send the real command without re-spawning.
        try:
            return _connect_and_send(cmd, args)
        except Exception as e:
            return {"ok": False, "error": f"call failed: {e}"}

    return _call_once_with_spawn(cmd, args)


def _config_mismatch_error(
    status: dict[str, Any],
    *,
    expected_vault: Path | None,
    expected_embedder: str | None,
    expected_reranker: str | None,
) -> str | None:
    """Return a human-readable mismatch reason, or None if config matches."""
    if expected_vault is not None:
        daemon_vault_raw = status.get("vault")
        if daemon_vault_raw is None:
            return "daemon status returned no vault"
        try:
            daemon_vault = Path(daemon_vault_raw).resolve()
            want_vault = expected_vault.resolve()
        except OSError as e:
            return f"vault resolution failed: {e}"
        if daemon_vault != want_vault:
            return (
                f"daemon is bound to vault {daemon_vault}, but caller "
                f"expects {want_vault}"
            )

    if expected_embedder is not None:
        daemon_embedder = status.get("embedder")
        if daemon_embedder != expected_embedder:
            return (
                f"daemon embedder is {daemon_embedder!r}, caller "
                f"expects {expected_embedder!r}"
            )

    if expected_reranker is not None:
        daemon_reranker = status.get("reranker")
        # Accept the daemon's self-disabled state even if we expected a
        # real model name. `run_daemon()` downgrades the reranker to
        # "disabled" at warmup time on platforms where mlx_lm cannot
        # load (Linux, Intel macOS). Rejecting that would break the
        # daemon-first workflow on those supported setups. But if the
        # caller explicitly asked for "disabled" (via empty
        # `SEEKLINK_RERANKER_MODEL`), a running daemon with a real
        # reranker IS a mismatch — the user asked for raw RRF scores
        # and would silently get reranked ones.
        is_mismatch = (
            daemon_reranker != expected_reranker
            and not (
                expected_reranker != "disabled"
                and daemon_reranker == "disabled"
            )
        )
        if is_mismatch:
            return (
                f"daemon reranker is {daemon_reranker!r}, caller "
                f"expects {expected_reranker!r}"
            )

    return None


def _restart_and_retry(
    cmd: str, args: dict[str, Any], mismatch: str
) -> dict[str, Any]:
    logger.info("Daemon config mismatch; restarting: %s", mismatch)
    shutdown = _shutdown_daemon()
    if not shutdown.get("ok"):
        return {
            "ok": False,
            "error": (
                f"{mismatch}; failed to shut down stale daemon: "
                f"{shutdown.get('error', 'unknown error')}"
            ),
        }

    if not _wait_for_socket_shutdown(SHUTDOWN_WAIT_SECONDS):
        return {
            "ok": False,
            "error": (
                f"{mismatch}; stale daemon did not stop within "
                f"{SHUTDOWN_WAIT_SECONDS}s"
            ),
        }

    return _call_once_with_spawn(cmd, args)


def _shutdown_daemon() -> dict[str, Any]:
    """Ask the currently running daemon to exit. Never spawns a daemon."""
    try:
        return _connect_and_send("shutdown", {})
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _call_once_with_spawn(cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """Try the daemon, spawn + retry once if unreachable."""
    try:
        return _connect_and_send(cmd, args)
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        logger.debug("Daemon unreachable: %s — spawning", e)

    _spawn_daemon()
    if not _wait_for_socket(SPAWN_WAIT_SECONDS):
        return {
            "ok": False,
            "error": f"daemon failed to start within {SPAWN_WAIT_SECONDS}s",
        }

    try:
        return _connect_and_send(cmd, args)
    except Exception as e:
        return {"ok": False, "error": f"call failed after spawn: {e}"}


def _connect_and_send(cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """Open a fresh socket, send one framed request, read one response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(SOCKET_PATH))

        payload = json.dumps({"cmd": cmd, "args": args}).encode("utf-8")
        header = len(payload).to_bytes(4, "big")
        sock.sendall(header + payload)

        # Read 4-byte length header
        hdr = b""
        while len(hdr) < 4:
            chunk = sock.recv(4 - len(hdr))
            if not chunk:
                raise ConnectionError("daemon closed connection before header")
            hdr += chunk
        length = int.from_bytes(hdr, "big")

        # Read message body
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                raise ConnectionError("daemon closed connection mid-response")
            data += chunk

        return json.loads(data.decode("utf-8"))
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _spawn_daemon() -> None:
    """Fork a detached daemon subprocess.

    The subprocess inherits the current environment (including
    SEEKLINK_VAULT / SEEKLINK_EMBEDDER_MODEL / SEEKLINK_RERANKER_MODEL)
    so that configuration is consistent with the invoking CLI.
    """
    try:
        subprocess.Popen(
            [sys.executable, "-m", "seeklink", "daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from parent process group
        )
    except Exception as e:
        logger.warning("Failed to spawn daemon: %s", e)


def _wait_for_socket(timeout: float) -> bool:
    """Poll until the socket exists and accepts connections, or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if SOCKET_PATH.exists():
            try:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.settimeout(0.5)
                probe.connect(str(SOCKET_PATH))
                probe.close()
                return True
            except (ConnectionRefusedError, OSError):
                pass
        time.sleep(0.2)
    return False


def _wait_for_socket_shutdown(timeout: float) -> bool:
    """Wait until the socket is gone or no longer accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not SOCKET_PATH.exists():
            return True
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.5)
            probe.connect(str(SOCKET_PATH))
            probe.close()
        except (ConnectionRefusedError, OSError):
            return True
        time.sleep(0.1)
    return False

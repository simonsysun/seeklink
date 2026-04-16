"""Thin Unix-socket client for the seeklink daemon.

Tries to reach the daemon at ~/.rhizome/seeklink.sock. If the socket is
missing or the daemon is unreachable, spawns a new daemon subprocess
(detached) and waits for it to come up before retrying. Any failure to
spawn or connect is reported back as a failed response — callers can
fall back to a cold-start in-process execution if they prefer.
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


def call(cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """Send a command to the daemon. Auto-spawns daemon if unreachable.

    Returns the daemon's JSON response as a dict. Never raises — on
    failure returns ``{"ok": False, "error": "..."}``.
    """
    # First attempt
    try:
        return _connect_and_send(cmd, args)
    except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
        logger.debug("Daemon unreachable: %s — spawning", e)

    # Spawn daemon and retry
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

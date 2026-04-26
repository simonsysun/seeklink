"""Protocol-level tests for the Unix-socket daemon handler."""

from __future__ import annotations

import json
import socket
from pathlib import Path

from seeklink.daemon import _handle_connection


def _send_request(sock: socket.socket, payload: dict) -> None:
    raw = json.dumps(payload).encode("utf-8")
    sock.sendall(len(raw).to_bytes(4, "big") + raw)


def _recv_response(sock: socket.socket) -> dict:
    header = sock.recv(4)
    length = int.from_bytes(header, "big")
    data = b""
    while len(data) < length:
        data += sock.recv(length - len(data))
    return json.loads(data.decode("utf-8"))


def test_shutdown_command_sends_ack_and_requests_shutdown():
    client, server = socket.socketpair()
    shutdown_requested: list[bool] = []

    try:
        _send_request(client, {"cmd": "shutdown", "args": {}})
        _handle_connection(
            server,
            db=None,
            embedder=None,
            reranker=None,
            vault_root=Path("/tmp/vault"),
            request_shutdown=lambda: shutdown_requested.append(True),
        )
        response = _recv_response(client)
    finally:
        client.close()
        server.close()

    assert response == {"ok": True, "result": {"status": "shutting_down"}}
    assert shutdown_requested == [True]

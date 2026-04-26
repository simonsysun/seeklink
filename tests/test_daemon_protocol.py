"""Protocol-level tests for the Unix-socket daemon handler."""

from __future__ import annotations

import json
import socket
import importlib
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


def test_search_no_rerank_passes_none_to_search(monkeypatch):
    client, server = socket.socketpair()
    captured: dict = {}

    class FakeEmbedder:
        MODEL_NAME = "test-embedder"

    class FakeReranker:
        disabled = False
        MODEL_NAME = "test-reranker"

    def fake_search(db, embedder, query, **kwargs):
        captured["query"] = query
        captured["reranker"] = kwargs["reranker"]
        captured["rerank_k"] = kwargs["rerank_k"]
        return []

    search_module = importlib.import_module("seeklink.search")
    monkeypatch.setattr(search_module, "search", fake_search)
    fake_reranker = FakeReranker()

    try:
        _send_request(
            client,
            {
                "cmd": "search",
                "args": {
                    "query": "memory",
                    "top_k": 3,
                    "rerank_k": 7,
                    "no_rerank": True,
                },
            },
        )
        _handle_connection(
            server,
            db=object(),
            embedder=FakeEmbedder(),
            reranker=fake_reranker,
            vault_root=Path("/tmp/vault"),
        )
        response = _recv_response(client)
    finally:
        client.close()
        server.close()

    assert response["ok"] is True
    assert response["result"] == []
    assert captured == {
        "query": "memory",
        "reranker": None,
        "rerank_k": 7,
    }


def test_search_auto_rerank_k_passes_through(monkeypatch):
    client, server = socket.socketpair()
    captured: dict = {}

    class FakeEmbedder:
        MODEL_NAME = "test-embedder"

    class FakeReranker:
        disabled = False
        MODEL_NAME = "test-reranker"

    def fake_search(db, embedder, query, **kwargs):
        captured["query"] = query
        captured["reranker"] = kwargs["reranker"]
        captured["rerank_k"] = kwargs["rerank_k"]
        return []

    search_module = importlib.import_module("seeklink.search")
    monkeypatch.setattr(search_module, "search", fake_search)
    fake_reranker = FakeReranker()

    try:
        _send_request(
            client,
            {
                "cmd": "search",
                "args": {
                    "query": "memory",
                    "top_k": 3,
                    "rerank_k": "auto",
                },
            },
        )
        _handle_connection(
            server,
            db=object(),
            embedder=FakeEmbedder(),
            reranker=fake_reranker,
            vault_root=Path("/tmp/vault"),
        )
        response = _recv_response(client)
    finally:
        client.close()
        server.close()

    assert response["ok"] is True
    assert response["result"] == []
    assert captured == {
        "query": "memory",
        "reranker": fake_reranker,
        "rerank_k": "auto",
    }

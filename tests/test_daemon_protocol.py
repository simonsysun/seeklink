"""Protocol-level tests for the Unix-socket daemon handler."""

from __future__ import annotations

import json
import socket
import importlib
from pathlib import Path

from seeklink.daemon import _handle_connection, _parse_idle_timeout, _idle_timed_out


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


def test_parse_idle_timeout_values(monkeypatch):
    monkeypatch.delenv("SEEKLINK_DAEMON_IDLE_TIMEOUT", raising=False)
    assert _parse_idle_timeout(None) == 900
    assert _parse_idle_timeout("15") == 15
    assert _parse_idle_timeout("0") is None
    assert _parse_idle_timeout("off") is None
    assert _parse_idle_timeout("false") is None
    assert _parse_idle_timeout("no") is None
    assert _parse_idle_timeout("never") is None
    assert _parse_idle_timeout("bad") == 900
    assert _parse_idle_timeout("-1") == 900


def test_idle_timed_out():
    state = {
        "idle_timeout_s": 10,
        "last_activity": 100.0,
    }
    assert not _idle_timed_out(state, now=109.9)
    assert _idle_timed_out(state, now=110.0)

    state["idle_timeout_s"] = None
    assert not _idle_timed_out(state, now=10_000.0)


def test_status_response_includes_daemon_metadata():
    client, server = socket.socketpair()

    class FakeDb:
        def get_stats(self):
            return {
                "notes_total": 0,
                "notes_unprocessed": 0,
                "chunks_total": 0,
                "links_total": 0,
                "suggestions_pending": 0,
                "wal_bytes": 0,
            }

        def get_index_metadata(self):
            return {}

        def get_vector_dimension(self):
            return None

    class FakeEmbedder:
        MODEL_NAME = "test-embedder"
        EMBEDDING_DIM = 768

    class FakeReranker:
        disabled = True
        MODEL_NAME = "test-reranker"

    state = {
        "started_at": 1778000000.0,
        "started_monotonic": 100.0,
        "last_activity": 120.0,
        "idle_timeout_s": 900,
        "requests_served": 3,
    }

    try:
        _send_request(client, {"cmd": "status", "args": {}})
        _handle_connection(
            server,
            db=FakeDb(),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            vault_root=Path("/tmp/vault"),
            daemon_state=state,
        )
        response = _recv_response(client)
    finally:
        client.close()
        server.close()

    assert response["ok"] is True
    result = response["result"]
    assert result["vault"] == "/tmp/vault"
    assert result["embedder"] == "test-embedder"
    assert result["reranker"] == "disabled"
    assert isinstance(result["pid"], int)
    assert result["socket"].endswith("seeklink.sock")
    assert result["started_at"] == 1778000000.0
    assert result["idle_timeout_s"] == 900
    assert result["requests_served"] == 3
    assert isinstance(result["uptime_s"], float)
    assert isinstance(result["idle_s"], float)
    assert result["rss_bytes"] is None or isinstance(result["rss_bytes"], int)


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
    index_config_module = importlib.import_module("seeklink.index_config")
    monkeypatch.setattr(search_module, "search", fake_search)
    monkeypatch.setattr(
        index_config_module,
        "ensure_index_compatible_for_search",
        lambda db, *, embedder_model, embedding_dim=None: None,
    )
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
    index_config_module = importlib.import_module("seeklink.index_config")
    monkeypatch.setattr(search_module, "search", fake_search)
    monkeypatch.setattr(
        index_config_module,
        "ensure_index_compatible_for_search",
        lambda db, *, embedder_model, embedding_dim=None: None,
    )
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


def test_search_defaults_to_auto_rerank_k(monkeypatch):
    client, server = socket.socketpair()
    captured: dict = {}

    class FakeEmbedder:
        MODEL_NAME = "test-embedder"

    class FakeReranker:
        disabled = False
        MODEL_NAME = "test-reranker"

    def fake_search(db, embedder, query, **kwargs):
        captured["query"] = query
        captured["rerank_k"] = kwargs["rerank_k"]
        return []

    search_module = importlib.import_module("seeklink.search")
    index_config_module = importlib.import_module("seeklink.index_config")
    monkeypatch.setattr(search_module, "search", fake_search)
    monkeypatch.setattr(
        index_config_module,
        "ensure_index_compatible_for_search",
        lambda db, *, embedder_model, embedding_dim=None: None,
    )

    try:
        _send_request(
            client,
            {
                "cmd": "search",
                "args": {"query": "memory"},
            },
        )
        _handle_connection(
            server,
            db=object(),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
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
        "rerank_k": "auto",
    }

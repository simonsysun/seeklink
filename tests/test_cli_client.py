"""Tests for daemon client config guards and auto-respawn behavior."""

from __future__ import annotations

from pathlib import Path

from seeklink import cli_client


def test_call_respawns_on_vault_mismatch(tmp_path: Path, monkeypatch):
    expected = tmp_path / "expected"
    stale = tmp_path / "stale"
    expected.mkdir()
    stale.mkdir()

    calls: list[tuple[str, dict]] = []
    shutdowns: list[bool] = []

    def fake_call_once(cmd: str, args: dict) -> dict:
        calls.append((cmd, args))
        if cmd == "status":
            return {
                "ok": True,
                "result": {
                    "vault": str(stale),
                    "embedder": "embedder-a",
                    "reranker": "reranker-a",
                },
            }
        return {"ok": True, "result": "retried"}

    monkeypatch.setattr(cli_client, "_call_once_with_spawn", fake_call_once)
    monkeypatch.setattr(
        cli_client,
        "_shutdown_daemon",
        lambda: shutdowns.append(True) or {"ok": True},
    )
    monkeypatch.setattr(cli_client, "_wait_for_socket_shutdown", lambda _timeout: True)

    resp = cli_client.call(
        "search",
        {"query": "memory"},
        expected_vault=expected,
        expected_embedder="embedder-a",
        expected_reranker="reranker-a",
    )

    assert resp == {"ok": True, "result": "retried"}
    assert shutdowns == [True]
    assert calls == [
        ("status", {}),
        ("search", {"query": "memory"}),
    ]


def test_call_respawns_when_user_disables_running_reranker(tmp_path: Path, monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_call_once(cmd: str, args: dict) -> dict:
        calls.append((cmd, args))
        if cmd == "status":
            return {
                "ok": True,
                "result": {
                    "vault": str(tmp_path),
                    "embedder": "embedder-a",
                    "reranker": "reranker-a",
                },
            }
        return {"ok": True, "result": "raw-rrf"}

    monkeypatch.setattr(cli_client, "_call_once_with_spawn", fake_call_once)
    monkeypatch.setattr(cli_client, "_shutdown_daemon", lambda: {"ok": True})
    monkeypatch.setattr(cli_client, "_wait_for_socket_shutdown", lambda _timeout: True)

    resp = cli_client.call(
        "search",
        {"query": "memory"},
        expected_vault=tmp_path,
        expected_embedder="embedder-a",
        expected_reranker="disabled",
    )

    assert resp == {"ok": True, "result": "raw-rrf"}
    assert calls == [
        ("status", {}),
        ("search", {"query": "memory"}),
    ]


def test_call_accepts_platform_reranker_downgrade(tmp_path: Path, monkeypatch):
    shutdowns: list[bool] = []

    monkeypatch.setattr(
        cli_client,
        "_call_once_with_spawn",
        lambda cmd, args: {
            "ok": True,
            "result": {
                "vault": str(tmp_path),
                "embedder": "embedder-a",
                "reranker": "disabled",
            },
        },
    )
    monkeypatch.setattr(
        cli_client,
        "_connect_and_send",
        lambda cmd, args: {"ok": True, "result": f"{cmd}-ok"},
    )
    monkeypatch.setattr(
        cli_client,
        "_shutdown_daemon",
        lambda: shutdowns.append(True) or {"ok": True},
    )

    resp = cli_client.call(
        "search",
        {"query": "memory"},
        expected_vault=tmp_path,
        expected_embedder="embedder-a",
        expected_reranker="reranker-a",
    )

    assert resp == {"ok": True, "result": "search-ok"}
    assert shutdowns == []


def test_call_returns_failure_when_stale_daemon_will_not_shutdown(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        cli_client,
        "_call_once_with_spawn",
        lambda cmd, args: {
            "ok": True,
            "result": {
                "vault": str(tmp_path / "stale"),
                "embedder": "embedder-a",
                "reranker": "reranker-a",
            },
        },
    )
    monkeypatch.setattr(
        cli_client,
        "_shutdown_daemon",
        lambda: {"ok": False, "error": "unknown command: shutdown"},
    )

    resp = cli_client.call(
        "search",
        {"query": "memory"},
        expected_vault=tmp_path,
        expected_embedder="embedder-a",
        expected_reranker="reranker-a",
    )

    assert resp["ok"] is False
    assert "failed to shut down stale daemon" in resp["error"]
    assert "unknown command: shutdown" in resp["error"]

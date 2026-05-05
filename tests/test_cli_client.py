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


def test_probe_status_never_spawns(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_connect(cmd: str, args: dict) -> dict:
        calls.append((cmd, args))
        return {"ok": True, "result": {"pid": 123}}

    monkeypatch.setattr(cli_client, "_connect_and_send", fake_connect)
    monkeypatch.setattr(
        cli_client,
        "_spawn_daemon",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("spawned")),
    )

    assert cli_client.probe_status() == {"ok": True, "result": {"pid": 123}}
    assert calls == [("status", {})]


def test_stop_daemon_is_success_when_socket_missing(monkeypatch):
    class FakeSocketPath:
        def exists(self):
            return False

        def __str__(self):
            return "/tmp/seeklink.sock"

    monkeypatch.setattr(cli_client, "SOCKET_PATH", FakeSocketPath())

    assert cli_client.stop_daemon() == {
        "ok": True,
        "result": {"status": "not_running"},
    }


def test_stop_daemon_sends_shutdown_and_waits(monkeypatch):
    calls: list[str] = []

    class FakeSocketPath:
        def exists(self):
            return True

        def __str__(self):
            return "/tmp/seeklink.sock"

    monkeypatch.setattr(cli_client, "SOCKET_PATH", FakeSocketPath())
    monkeypatch.setattr(
        cli_client,
        "_shutdown_daemon",
        lambda: calls.append("shutdown") or {"ok": True},
    )
    monkeypatch.setattr(
        cli_client,
        "_wait_for_socket_shutdown",
        lambda timeout: calls.append(f"wait:{timeout}") or True,
    )

    assert cli_client.stop_daemon(timeout=7.0) == {
        "ok": True,
        "result": {"status": "stopped"},
    }
    assert calls == ["shutdown", "wait:7.0"]


def test_start_daemon_spawns_and_returns_status(tmp_path: Path, monkeypatch):
    calls: list[tuple[str, Path | float | None]] = []
    monkeypatch.setattr(
        cli_client,
        "_spawn_daemon",
        lambda *, vault=None: calls.append(("spawn", vault)),
    )
    monkeypatch.setattr(
        cli_client,
        "_wait_for_socket",
        lambda timeout: calls.append(("wait", timeout)) or True,
    )
    monkeypatch.setattr(
        cli_client,
        "probe_status",
        lambda: {"ok": True, "result": {"pid": 123}},
    )

    assert cli_client.start_daemon(vault=tmp_path, timeout=2.0) == {
        "ok": True,
        "result": {"pid": 123},
    }
    assert calls == [("spawn", tmp_path), ("wait", 2.0)]

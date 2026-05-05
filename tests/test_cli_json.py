"""Tests for machine-readable CLI JSON output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

import seeklink.__main__ as cli
from seeklink.search import SearchResult


def test_search_json_daemon_response(capsys, monkeypatch):
    def fake_try_daemon(cmd: str, daemon_args: dict) -> dict:
        assert cmd == "search"
        assert daemon_args == {
            "query": "记忆保持力",
            "top_k": 1,
            "rerank_k": 20,
            "tags": ["learning"],
            "folder": "notes",
        }
        return {
            "ok": True,
            "vault": "/tmp/vault",
            "embedder": "test-embedder",
            "reranker": "disabled",
            "result": [
                {
                    "source_id": 7,
                    "path": "notes/memory.md",
                    "title": "记忆保持力",
                    "content_preview": "间隔重复可以提高长期记忆保持力。",
                    "score": 0.875,
                    "indegree": 3,
                    "line_start": 12,
                    "line_end": 14,
                }
            ],
        }

    monkeypatch.setattr(cli, "_try_daemon", fake_try_daemon)
    args = argparse.Namespace(
        query="记忆保持力",
        vault=None,
        tags=["learning"],
        folder="notes",
        top_k=1,
        rerank_k=20,
        no_rerank=False,
        title_weight=None,
        json=True,
    )

    cli._cmd_search(args)

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["ok"] is True
    assert payload["json_schema_version"] == 1
    assert payload["query"] == "记忆保持力"
    assert payload["vault"] == "/tmp/vault"
    assert payload["reranking"] == {"enabled": False, "rerank_k": 0}
    assert payload["filters"] == {"tags": ["learning"], "folder": "notes"}
    assert payload["models"] == {
        "embedder": "test-embedder",
        "reranker": "disabled",
    }
    assert payload["results"] == [
        {
            "source_id": 7,
            "path": "notes/memory.md",
            "title": "记忆保持力",
            "content_preview": "间隔重复可以提高长期记忆保持力。",
            "score": 0.875,
            "indegree": 3,
            "line_start": 12,
            "line_end": 14,
        }
    ]


def test_search_parser_defaults_to_auto_rerank_k(monkeypatch):
    captured: dict = {}

    def fake_cmd_search(args):
        captured["rerank_k"] = args.rerank_k

    monkeypatch.setattr(sys, "argv", ["seeklink", "search", "memory"])
    monkeypatch.setattr(cli, "_cmd_search", fake_cmd_search)
    cli.main()

    assert captured == {"rerank_k": "auto"}


def test_search_parser_accepts_no_daemon(monkeypatch):
    captured: dict = {}

    def fake_cmd_search(args):
        captured["no_daemon"] = args.no_daemon

    monkeypatch.setattr(sys, "argv", ["seeklink", "search", "memory", "--no-daemon"])
    monkeypatch.setattr(cli, "_cmd_search", fake_cmd_search)
    cli.main()

    assert captured == {"no_daemon": True}


def test_index_parser_accepts_no_daemon(monkeypatch):
    captured: dict = {}

    def fake_cmd_index(args):
        captured["no_daemon"] = args.no_daemon

    monkeypatch.setattr(sys, "argv", ["seeklink", "index", "note.md", "--no-daemon"])
    monkeypatch.setattr(cli, "_cmd_index", fake_cmd_index)
    cli.main()

    assert captured == {"no_daemon": True}


def test_daemon_parser_accepts_status_json(monkeypatch):
    captured: dict = {}

    def fake_cmd_daemon(args):
        captured["daemon_action"] = args.daemon_action
        captured["json"] = args.json

    monkeypatch.setattr(sys, "argv", ["seeklink", "daemon", "status", "--json"])
    monkeypatch.setattr(cli, "_cmd_daemon", fake_cmd_daemon)
    cli.main()

    assert captured == {"daemon_action": "status", "json": True}


def test_daemon_parser_defaults_to_run(monkeypatch):
    captured: dict = {}

    def fake_cmd_daemon(args):
        captured["daemon_action"] = args.daemon_action
        captured["vault"] = args.vault

    monkeypatch.setattr(sys, "argv", ["seeklink", "daemon", "--vault", "/tmp/vault"])
    monkeypatch.setattr(cli, "_cmd_daemon", fake_cmd_daemon)
    cli.main()

    assert captured == {"daemon_action": "run", "vault": Path("/tmp/vault")}


def test_should_use_daemon_honors_flag_and_env(monkeypatch):
    assert cli._should_use_daemon(argparse.Namespace(vault=None, no_daemon=False))

    assert not cli._should_use_daemon(argparse.Namespace(vault=Path(".")))
    assert not cli._should_use_daemon(argparse.Namespace(vault=None, no_daemon=True))

    monkeypatch.setenv("SEEKLINK_NO_DAEMON", "1")
    assert not cli._should_use_daemon(argparse.Namespace(vault=None))

    monkeypatch.setenv("SEEKLINK_NO_DAEMON", "false")
    assert cli._should_use_daemon(argparse.Namespace(vault=None))


def test_search_result_to_json_truncates_preview():
    result = SearchResult(
        source_id=1,
        chunk_id=10,
        path="notes/long.md",
        title=None,
        content="x" * 250,
        score=0.25,
        indegree=2,
        line_start=4,
        line_end=9,
    )

    payload = cli._search_result_to_json(result)

    assert payload["title"] == ""
    assert payload["content_preview"] == "x" * 200
    assert payload["source_id"] == 1
    assert payload["line_start"] == 4
    assert payload["line_end"] == 9


def test_status_json_subprocess(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "seeklink",
        "status",
        "--vault",
        str(tmp_path),
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["json_schema_version"] == 1
    assert payload["vault"] == str(tmp_path)
    assert payload["database"]["schema_version"] == 4
    assert payload["index"] == {
        "metadata": {},
        "compatibility": {
            "compatible": True,
            "state": "empty",
            "mismatches": {},
        },
    }
    assert payload["stats"] == {
        "notes_total": 0,
        "notes_unprocessed": 0,
        "chunks_total": 0,
        "links_total": 0,
        "suggestions_pending": 0,
    }
    assert isinstance(payload["database"]["wal_bytes"], int)
    assert payload["freshness"] == {
        "checked": True,
        "fresh": True,
        "suspect_files": 0,
    }
    assert payload["models"]["embedder"] == "jinaai/jina-embeddings-v2-base-zh"
    assert payload["models"]["reranker"] == "mlx-community/Qwen3-Reranker-0.6B-mxfp8"


def test_doctor_json_subprocess(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "seeklink",
        "doctor",
        "--vault",
        str(tmp_path),
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["json_schema_version"] == 1
    assert payload["vault"] == str(tmp_path)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["python"]["ok"] is True
    assert checks["sqlite"]["ok"] is True
    assert checks["database"]["ok"] is True
    assert checks["index_compatibility"]["ok"] is True
    assert checks["mlx_lm"]["required"] is False
    assert isinstance(payload["daemon"]["running"], bool)
    assert payload["daemon"]["socket"].endswith("seeklink.sock")


def test_daemon_status_json_not_running(capsys, monkeypatch):
    from seeklink import cli_client

    monkeypatch.setattr(
        cli_client,
        "probe_status",
        lambda: {"ok": False, "error": "socket missing"},
    )
    monkeypatch.setattr(cli_client, "SOCKET_PATH", Path("/tmp/seeklink.sock"))

    cli._cmd_daemon_status(argparse.Namespace(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "json_schema_version": 1,
        "daemon": {
            "running": False,
            "socket": "/tmp/seeklink.sock",
        },
    }


def test_daemon_status_json_running(capsys, monkeypatch):
    from seeklink import cli_client

    monkeypatch.setattr(
        cli_client,
        "probe_status",
        lambda: {
            "ok": True,
            "result": {
                "pid": 123,
                "socket": "/tmp/seeklink.sock",
                "vault": "/tmp/vault",
                "embedder": "embedder-a",
                "reranker": "disabled",
                "started_at": 1778000000.0,
                "uptime_s": 12.5,
                "idle_s": 2.0,
                "idle_timeout_s": 900,
                "requests_served": 4,
                "rss_bytes": 123456,
            },
        },
    )

    cli._cmd_daemon_status(argparse.Namespace(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["daemon"] == {
        "running": True,
        "pid": 123,
        "socket": "/tmp/seeklink.sock",
        "vault": "/tmp/vault",
        "embedder": "embedder-a",
        "reranker": "disabled",
        "started_at": 1778000000.0,
        "uptime_s": 12.5,
        "idle_s": 2.0,
        "idle_timeout_s": 900,
        "requests_served": 4,
        "rss_bytes": 123456,
    }


def test_daemon_stop_json_not_running(capsys, monkeypatch):
    from seeklink import cli_client

    monkeypatch.setattr(
        cli_client,
        "stop_daemon",
        lambda: {"ok": True, "result": {"status": "not_running"}},
    )
    monkeypatch.setattr(cli_client, "SOCKET_PATH", Path("/tmp/seeklink.sock"))

    cli._cmd_daemon_stop(argparse.Namespace(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "json_schema_version": 1,
        "daemon": {
            "running": False,
            "socket": "/tmp/seeklink.sock",
            "status": "not_running",
        },
    }


def test_daemon_pid_outputs_pid(capsys, monkeypatch):
    from seeklink import cli_client

    monkeypatch.setattr(
        cli_client,
        "probe_status",
        lambda: {"ok": True, "result": {"pid": 123, "socket": "/tmp/sock"}},
    )

    cli._cmd_daemon_pid(argparse.Namespace(json=False))

    assert capsys.readouterr().out == "123\n"


def test_search_json_no_rerank_sends_daemon_flag(capsys, monkeypatch):
    def fake_try_daemon(cmd: str, daemon_args: dict) -> dict:
        assert cmd == "search"
        assert daemon_args == {
            "query": "memory",
            "top_k": 3,
            "rerank_k": 7,
            "no_rerank": True,
        }
        return {
            "ok": True,
            "vault": "/tmp/vault",
            "embedder": "test-embedder",
            "reranker": "test-reranker",
            "result": [],
        }

    monkeypatch.setattr(cli, "_try_daemon", fake_try_daemon)
    args = argparse.Namespace(
        query="memory",
        vault=None,
        tags=None,
        folder=None,
        top_k=3,
        rerank_k=7,
        no_rerank=True,
        title_weight=None,
        json=True,
    )

    cli._cmd_search(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["reranking"] == {"enabled": False, "rerank_k": 0}
    assert payload["results"] == []


def test_search_json_auto_rerank_sends_daemon_value(capsys, monkeypatch):
    def fake_try_daemon(cmd: str, daemon_args: dict) -> dict:
        assert cmd == "search"
        assert daemon_args == {
            "query": "memory",
            "top_k": 3,
            "rerank_k": "auto",
        }
        return {
            "ok": True,
            "vault": "/tmp/vault",
            "embedder": "test-embedder",
            "reranker": "test-reranker",
            "result": [],
        }

    monkeypatch.setattr(cli, "_try_daemon", fake_try_daemon)
    args = argparse.Namespace(
        query="memory",
        vault=None,
        tags=None,
        folder=None,
        top_k=3,
        rerank_k="auto",
        no_rerank=False,
        title_weight=None,
        json=True,
    )

    cli._cmd_search(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["reranking"] == {"enabled": True, "rerank_k": "auto"}
    assert payload["results"] == []


def test_search_rejects_invalid_rerank_k(capsys):
    args = argparse.Namespace(
        query="memory",
        vault=None,
        tags=None,
        folder=None,
        top_k=3,
        rerank_k=0,
        no_rerank=False,
        title_weight=None,
        json=False,
    )

    try:
        cli._cmd_search(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("Expected _cmd_search to exit for invalid rerank_k")

    assert "--rerank-k must be a positive integer or 'auto'" in capsys.readouterr().err


def test_parse_rerank_k_accepts_auto_and_positive_integers():
    assert cli._parse_rerank_k("auto") == "auto"
    assert cli._parse_rerank_k("7") == 7


def test_parse_rerank_k_rejects_invalid_values():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_rerank_k("0")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_rerank_k("fast")

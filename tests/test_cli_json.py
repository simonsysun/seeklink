"""Tests for machine-readable CLI JSON output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import seeklink.__main__ as cli
from seeklink.search import SearchResult


def test_search_json_daemon_response(capsys, monkeypatch):
    def fake_try_daemon(cmd: str, daemon_args: dict) -> dict:
        assert cmd == "search"
        assert daemon_args == {
            "query": "记忆保持力",
            "top_k": 1,
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
    assert payload["database"]["schema_version"] == 2
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

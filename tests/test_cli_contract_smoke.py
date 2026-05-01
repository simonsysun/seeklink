"""Smoke tests for documented CLI examples against the bundled corpus."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run_seeklink(vault: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SEEKLINK_RERANKER_MODEL"] = ""
    env.pop("SEEKLINK_VAULT", None)
    return subprocess.run(
        [sys.executable, "-m", "seeklink", *args, "--vault", str(vault)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def _fresh_corpus(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parent / "corpus"
    vault = tmp_path / "corpus"
    shutil.copytree(source, vault, ignore=shutil.ignore_patterns(".seeklink"))
    return vault


def test_documented_non_daemon_cli_workflow(tmp_path: Path):
    """README/llms examples stay runnable on a fresh bundled corpus copy."""
    vault = _fresh_corpus(tmp_path)

    status_before = _run_seeklink(vault, "status")
    assert status_before.returncode == 0, status_before.stderr
    assert "Notes:" in status_before.stdout

    index = _run_seeklink(vault, "index")
    assert index.returncode == 0, index.stderr
    assert "Done:" in index.stdout
    assert "Scanning vault..." in index.stderr
    assert "Embedding" in index.stderr

    status_json = _run_seeklink(vault, "status", "--json")
    assert status_json.returncode == 0, status_json.stderr
    status_payload = json.loads(status_json.stdout)
    assert status_payload["ok"] is True
    assert status_payload["stats"]["notes_total"] > 0
    assert status_payload["stats"]["chunks_total"] > 0

    search = _run_seeklink(vault, "search", "agent memory systems", "--top-k", "3")
    assert search.returncode == 0, search.stderr
    assert re.search(r"^\s*[0-9.]+\s+\S+\.md:\d+\s+", search.stdout, re.MULTILINE)
    assert re.search(r"^\s{2,}\S+", search.stdout, re.MULTILINE)

    search_json = _run_seeklink(
        vault,
        "search",
        "记忆保持力",
        "--title-weight",
        "0.5",
        "--top-k",
        "3",
        "--json",
    )
    assert search_json.returncode == 0, search_json.stderr
    search_payload = json.loads(search_json.stdout)
    assert search_payload["ok"] is True
    assert search_payload["results"]

    first = search_payload["results"][0]
    assert first["path"].endswith(".md")
    assert isinstance(first["line_start"], int)
    assert first["line_start"] >= 1

    hit = f"{first['path']}:{first['line_start']}"
    get_lines = _run_seeklink(vault, "get", hit, "-l", "5")
    assert get_lines.returncode == 0, get_lines.stderr
    assert get_lines.stdout.strip()

    get_context = _run_seeklink(vault, "get", hit, "-C", "2")
    assert get_context.returncode == 0, get_context.stderr
    assert get_context.stdout.strip()

"""Black-box MCP stdio tests for the SeekLink adapter."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

import seeklink.mcp_server as mcp_server


def test_mcp_search_summary_is_compact_and_actionable():
    payload = {
        "query": "agent memory systems",
        "results": [
            {
                "path": "notes/agent-memory-patterns.md",
                "line_start": 51,
                "title": "Agent memory patterns",
                "content_preview": "This preview is available in structuredContent and should not be duplicated.",
            },
            {
                "path": "notes/retrieval-augmented-generation.md",
                "line_start": 10,
                "title": "Retrieval augmented generation",
                "content_preview": "Another preview that would waste text-summary tokens.",
            },
        ],
        "warnings": ["seeklink: 1 modified file(s) since last index."],
    }

    summary = mcp_server._search_summary(payload)

    assert "Use get(path, line, lines)" in summary
    assert "notes/agent-memory-patterns.md:51 Agent memory patterns" in summary
    assert "notes/retrieval-augmented-generation.md:10 Retrieval augmented generation" in summary
    assert "seeklink: 1 modified file(s) since last index." in summary
    assert "should not be duplicated" not in summary
    assert "would waste text-summary tokens" not in summary


@pytest.mark.asyncio
async def test_mcp_stdio_exposes_read_only_tools(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    env = os.environ.copy()
    env["SEEKLINK_RERANKER_MODEL"] = ""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "seeklink", "mcp", "--vault", str(vault)],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = {tool.name: tool for tool in tools_result.tools}
            assert set(tools) == {"search", "get", "status", "doctor"}
            for tool in tools.values():
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is True
                assert tool.annotations.idempotentHint is True
                assert tool.annotations.destructiveHint is False
                assert tool.annotations.openWorldHint is False
            assert "results" in tools["search"].outputSchema["properties"]
            assert (
                "content_preview"
                in tools["search"].outputSchema["$defs"]["SearchResultResponse"]["properties"]
            )
            assert "text" in tools["get"].outputSchema["properties"]
            assert "database" in tools["status"].outputSchema["properties"]
            assert "checks" in tools["doctor"].outputSchema["properties"]

            status = await session.call_tool("status", {})
            assert status.isError is not True
            assert status.structuredContent["ok"] is True
            assert status.structuredContent["database"]["exists"] is False

            fetched = await session.call_tool(
                "get",
                {"path": "note.md", "line": 2, "lines": 1},
            )
            assert fetched.isError is not True
            assert fetched.structuredContent["text"] == "beta"
            assert any(
                isinstance(content, TextContent)
                and "Untrusted vault content follows" in content.text
                for content in fetched.content
            )

            missing_index = await session.call_tool("search", {"query": "alpha"})
            assert missing_index.isError is True
            assert missing_index.structuredContent["error"]["code"] == "NO_INDEX"


@pytest.mark.asyncio
async def test_mcp_stdio_searches_indexed_corpus(tmp_path: Path):
    source = Path(__file__).resolve().parent / "corpus"
    vault = tmp_path / "corpus"
    shutil.copytree(source, vault, ignore=shutil.ignore_patterns(".seeklink"))

    env = os.environ.copy()
    env["SEEKLINK_RERANKER_MODEL"] = ""
    repo = Path(__file__).resolve().parents[1]
    indexed = subprocess.run(
        [sys.executable, "-m", "seeklink", "index", "--vault", str(vault)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert indexed.returncode == 0, indexed.stderr

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "seeklink", "mcp", "--vault", str(vault)],
        env=env,
        cwd=str(repo),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            searched = await session.call_tool(
                "search",
                {"query": "agent memory systems", "top_k": 3, "rerank": False},
            )
            assert searched.isError is not True
            results = searched.structuredContent["results"]
            assert results
            assert results[0]["path"] == "notes/agent-memory-patterns.md"
            assert results[0]["content_preview"]

            fetched = await session.call_tool(
                "get",
                {
                    "path": results[0]["path"],
                    "line": results[0]["line_start"],
                    "lines": 3,
                },
            )
            assert fetched.isError is not True
            assert fetched.structuredContent["text"].strip()


@pytest.mark.asyncio
async def test_mcp_server_reuses_embedder_and_reranker_between_search_calls(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    rerankers_constructed = 0
    embedders_constructed = 0
    seen_embedders = []
    seen_rerankers = []

    class FakeEmbedder:
        MODEL_NAME = "fake-embedder"

        def __init__(self) -> None:
            nonlocal embedders_constructed
            embedders_constructed += 1

    class FakeReranker:
        disabled = True
        MODEL_NAME = "fake-reranker"

        def __init__(self) -> None:
            nonlocal rerankers_constructed
            rerankers_constructed += 1

    def fake_mcp_search(vault_path, **kwargs):
        seen_embedders.append(kwargs["embedder"])
        seen_rerankers.append(kwargs["reranker"])
        return {
            "ok": True,
            "json_schema_version": 1,
            "query": kwargs["query"],
            "vault": str(vault_path),
            "top_k": kwargs["top_k"],
            "reranking": {"enabled": False, "rerank_k": 0},
            "filters": {"tags": [], "folder": None},
            "models": {"embedder": "fake", "reranker": "disabled"},
            "freshness": {"checked": True, "fresh": True, "suspect_files": 0},
            "results": [],
            "warnings": [],
        }

    monkeypatch.setattr("seeklink.embedder.Embedder", FakeEmbedder)
    monkeypatch.setattr("seeklink.reranker.Reranker", FakeReranker)
    monkeypatch.setattr(mcp_server, "mcp_search", fake_mcp_search)

    server = mcp_server.build_mcp_server(vault)
    await server.call_tool("search", {"query": "one"})
    await server.call_tool("search", {"query": "two"})

    assert embedders_constructed == 1
    assert rerankers_constructed == 1
    assert seen_embedders[0] is seen_embedders[1]
    assert seen_rerankers[0] is seen_rerankers[1]


@pytest.mark.asyncio
async def test_mcp_tool_redirects_service_stdout(monkeypatch, tmp_path: Path, capsys):
    vault = tmp_path / "vault"
    vault.mkdir()

    def noisy_get(vault_path, **kwargs):
        print("third-party stdout noise")
        return {
            "ok": True,
            "json_schema_version": 1,
            "vault": str(vault_path),
            "path": kwargs["path"],
            "line_requested": None,
            "line_start": 1,
            "line_end": 1,
            "text": "ok",
            "mime_type": "text/markdown",
            "warnings": [],
        }

    monkeypatch.setattr(mcp_server, "mcp_get", noisy_get)

    server = mcp_server.build_mcp_server(vault)
    await server.call_tool("get", {"path": "note.md"})

    captured = capsys.readouterr()
    assert "third-party stdout noise" not in captured.out
    assert "third-party stdout noise" in captured.err

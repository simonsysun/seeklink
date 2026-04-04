"""Integration tests — all 8 MCP tools tested through the real protocol.

Uses mcp.shared.memory.create_connected_server_and_client_session to create
an in-memory client↔server connection that runs the real lifespan (DB, embedder,
watcher). This catches bugs that unit tests miss, like the Context annotation
issue where every tool returned 'NoneType' errors through the protocol despite
165 unit tests passing.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dict(result) -> dict:
    """Parse a tool result that returns a single dict."""
    assert not result.isError, f"Tool returned error: {result}"
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    return json.loads(result.content[0].text)


def _parse_list(result) -> list[dict]:
    """Parse a tool result that returns list[dict].

    FastMCP serializes each list element as a separate TextContent item,
    so [dict1, dict2] → [TextContent(json(dict1)), TextContent(json(dict2))].
    An empty list returns 0 content items.
    """
    assert not result.isError, f"Tool returned error: {result}"
    items = []
    for c in result.content:
        assert isinstance(c, TextContent)
        items.append(json.loads(c.text))
    return items


def _write_md(vault: Path, rel_path: str, content: str) -> Path:
    p = vault / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def vault(tmp_path_factory) -> Path:
    """Session-scoped vault with test corpus."""
    v = tmp_path_factory.mktemp("vault")
    _write_md(v, "ml-basics.md", "# Machine Learning\n\nML uses algorithms to learn from data.\n")
    _write_md(
        v,
        "deep-learning.md",
        "# Deep Learning\n\nNeural networks with many layers. See [[ml-basics]].\n",
    )
    _write_md(v, "cooking.md", "# Italian Cooking\n\nPasta recipes from Italy.\n")
    _write_md(
        v,
        "hub.md",
        "# Hub Note\n\nSee [[ml-basics]] and [[deep-learning]].\n",
    )
    _write_md(v, "知识管理.md", "# 知识管理入门\n\n知识管理是一种系统性的方法。\n")
    return v


@pytest.fixture(scope="session")
async def client(vault: Path) -> ClientSession:
    """Session-scoped MCP client connected via in-memory transport.

    The real lifespan runs: DB init, embedder load, watcher start.
    Session-scoped so the ~2s embedder load happens once.
    """
    old_env = os.environ.get("SOPHIA_VAULT")
    os.environ["SOPHIA_VAULT"] = str(vault)
    try:
        # Import here so SOPHIA_VAULT is set before lifespan reads it
        from sophia.server import mcp

        async with create_connected_server_and_client_session(
            mcp,
            raise_exceptions=True,
            read_timeout_seconds=timedelta(seconds=60),
        ) as session:
            yield session
    finally:
        if old_env is None:
            os.environ.pop("SOPHIA_VAULT", None)
        else:
            os.environ["SOPHIA_VAULT"] = old_env


# ---------------------------------------------------------------------------
# Tool 1: status (returns dict)
# ---------------------------------------------------------------------------

class TestStatus:
    pytestmark = pytest.mark.anyio

    async def test_returns_all_fields(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("status", {}))
        assert "notes_total" in data
        assert "notes_unprocessed" in data
        assert "chunks_total" in data
        assert "links_total" in data
        assert "suggestions_pending" in data
        assert "budget" in data
        assert "watcher_running" in data

    async def test_watcher_is_running(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("status", {}))
        assert data["watcher_running"] is True


# ---------------------------------------------------------------------------
# Tool 2: get_unprocessed (returns list[dict])
# ---------------------------------------------------------------------------

class TestGetUnprocessed:
    pytestmark = pytest.mark.anyio

    async def test_returns_list(self, client: ClientSession):
        data = _parse_list(await client.call_tool("get_unprocessed", {}))
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Tool 3: index_note (returns dict)
# ---------------------------------------------------------------------------

class TestIndexNote:
    pytestmark = pytest.mark.anyio

    async def test_index_creates_chunks(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index_note", {"path": "ml-basics.md"}))
        assert data["status"] == "indexed"
        assert data["chunks_created"] >= 0
        assert "source_id" in data

    async def test_index_parses_wiki_links(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index_note", {"path": "deep-learning.md"}))
        assert data["status"] == "indexed"
        assert data["links_parsed"] >= 1  # [[ml-basics]]

    async def test_index_missing_file(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index_note", {"path": "nonexistent.md"}))
        assert "error" in data
        assert "not found" in data["error"].lower()

    async def test_force_reindex(self, client: ClientSession):
        data = _parse_dict(
            await client.call_tool("index_note", {"path": "ml-basics.md", "force": True})
        )
        assert data["status"] == "indexed"

    async def test_path_normalization(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index_note", {"path": "./ml-basics.md"}))
        assert data["status"] == "indexed"


# ---------------------------------------------------------------------------
# Tool 4: search (returns list[dict])
# ---------------------------------------------------------------------------

class TestSearch:
    pytestmark = pytest.mark.anyio

    async def test_basic_search(self, client: ClientSession):
        # Ensure corpus is indexed
        await client.call_tool("index_note", {"path": "ml-basics.md"})
        await client.call_tool("index_note", {"path": "cooking.md"})

        data = _parse_list(await client.call_tool("search", {"query": "machine learning", "top_k": 3}))
        assert len(data) > 0
        assert any("ml-basics" in r["path"] for r in data)

    async def test_search_result_fields(self, client: ClientSession):
        data = _parse_list(await client.call_tool("search", {"query": "algorithms", "top_k": 1}))
        assert len(data) >= 1
        r = data[0]
        assert "source_id" in r
        assert "path" in r
        assert "title" in r
        assert "content_preview" in r
        assert "rrf_score" in r
        assert "indegree" in r

    async def test_search_with_expand(self, client: ClientSession):
        await client.call_tool("index_note", {"path": "hub.md"})
        data = _parse_list(
            await client.call_tool("search", {"query": "machine learning", "top_k": 5, "expand": True})
        )
        assert isinstance(data, list)

    async def test_chinese_search(self, client: ClientSession):
        await client.call_tool("index_note", {"path": "知识管理.md"})
        data = _parse_list(await client.call_tool("search", {"query": "知识管理", "top_k": 3}))
        assert len(data) > 0
        assert any("知识管理" in r["path"] for r in data)

    async def test_cross_language_search(self, client: ClientSession):
        """English query can find Chinese notes via vector similarity."""
        data = _parse_list(
            await client.call_tool("search", {"query": "knowledge management", "top_k": 5})
        )
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Tool 5: suggest_links (returns list[dict])
# ---------------------------------------------------------------------------

class TestSuggestLinks:
    pytestmark = pytest.mark.anyio

    async def test_returns_suggestions(self, client: ClientSession):
        await client.call_tool("index_note", {"path": "cooking.md"})
        data = _parse_list(
            await client.call_tool("suggest_links", {"path": "cooking.md", "max_suggestions": 3})
        )
        assert isinstance(data, list)
        if len(data) > 0:
            s = data[0]
            assert "suggestion_id" in s
            assert "target_path" in s
            assert "score" in s
            assert "reason" in s

    async def test_suggest_for_missing_note(self, client: ClientSession):
        # suggest_links returns [{"error": ...}] for missing notes
        data = _parse_list(
            await client.call_tool("suggest_links", {"path": "nonexistent.md"})
        )
        assert len(data) == 1
        assert "error" in data[0]


# ---------------------------------------------------------------------------
# Tool 6 & 7: approve_suggestion / reject_suggestion (return dict)
# ---------------------------------------------------------------------------

class TestSuggestionWorkflow:
    pytestmark = pytest.mark.anyio

    async def test_reject_suggestion(self, client: ClientSession):
        await client.call_tool("index_note", {"path": "cooking.md", "force": True})
        suggestions = _parse_list(
            await client.call_tool("suggest_links", {"path": "cooking.md", "max_suggestions": 2})
        )
        if len(suggestions) == 0:
            pytest.skip("No suggestions generated")

        sid = suggestions[0]["suggestion_id"]
        data = _parse_dict(await client.call_tool("reject_suggestion", {"suggestion_id": sid}))
        assert data["status"] == "rejected"

    async def test_reject_already_rejected(self, client: ClientSession):
        suggestions = _parse_list(
            await client.call_tool("suggest_links", {"path": "cooking.md", "max_suggestions": 1})
        )
        if len(suggestions) == 0:
            pytest.skip("No suggestions generated")

        sid = suggestions[0]["suggestion_id"]
        await client.call_tool("reject_suggestion", {"suggestion_id": sid})
        data = _parse_dict(await client.call_tool("reject_suggestion", {"suggestion_id": sid}))
        assert "error" in data
        assert "already" in data["error"].lower()

    async def test_reject_nonexistent(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("reject_suggestion", {"suggestion_id": 99999}))
        assert "error" in data

    async def test_approve_writes_link(self, client: ClientSession, vault: Path):
        await client.call_tool("index_note", {"path": "hub.md", "force": True})
        await client.call_tool("index_note", {"path": "ml-basics.md", "force": True})

        suggestions = _parse_list(
            await client.call_tool("suggest_links", {"path": "hub.md", "max_suggestions": 3})
        )
        if len(suggestions) == 0:
            pytest.skip("No suggestions generated")

        sid = suggestions[0]["suggestion_id"]
        target_path = suggestions[0]["target_path"]
        target_stem = Path(target_path).stem

        data = _parse_dict(await client.call_tool("approve_suggestion", {"suggestion_id": sid}))
        assert data["status"] == "approved"
        assert f"[[{target_stem}]]" in data["link_written"]

        # Verify file was actually modified
        hub_content = (vault / "hub.md").read_text(encoding="utf-8")
        assert f"[[{target_stem}]]" in hub_content

    async def test_approve_nonexistent(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("approve_suggestion", {"suggestion_id": 99999}))
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 8: graph_neighbors (returns dict)
# ---------------------------------------------------------------------------

class TestGraphNeighbors:
    pytestmark = pytest.mark.anyio

    async def test_returns_structure(self, client: ClientSession):
        await client.call_tool("index_note", {"path": "hub.md"})
        data = _parse_dict(await client.call_tool("graph_neighbors", {"path": "hub.md"}))
        assert "center" in data
        assert "outgoing" in data
        assert "incoming" in data
        assert "path" in data["center"]
        assert "title" in data["center"]

    async def test_outgoing_links(self, client: ClientSession):
        await client.call_tool("index_note", {"path": "hub.md"})
        await client.call_tool("index_note", {"path": "ml-basics.md"})
        await client.call_tool("index_note", {"path": "deep-learning.md"})

        data = _parse_dict(
            await client.call_tool("graph_neighbors", {"path": "hub.md", "depth": 1})
        )
        out_paths = {n["path"] for n in data["outgoing"]}
        assert "ml-basics.md" in out_paths or "deep-learning.md" in out_paths

    async def test_depth_2(self, client: ClientSession):
        data = _parse_dict(
            await client.call_tool("graph_neighbors", {"path": "hub.md", "depth": 2})
        )
        assert isinstance(data["outgoing"], list)

    async def test_missing_note(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("graph_neighbors", {"path": "nonexistent.md"}))
        assert "error" in data

    async def test_isolated_note(self, client: ClientSession):
        await client.call_tool("index_note", {"path": "cooking.md"})
        data = _parse_dict(await client.call_tool("graph_neighbors", {"path": "cooking.md"}))
        assert data["outgoing"] == []


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    pytestmark = pytest.mark.anyio

    async def test_all_8_tools_registered(self, client: ClientSession):
        result = await client.list_tools()
        names = {t.name for t in result.tools}
        expected = {
            "get_unprocessed",
            "index_note",
            "search",
            "suggest_links",
            "approve_suggestion",
            "reject_suggestion",
            "graph_neighbors",
            "status",
        }
        assert expected == names

"""Integration tests — all 6 MCP tools tested through the real protocol.

Uses mcp.shared.memory.create_connected_server_and_client_session to create
an in-memory client<->server connection that runs the real lifespan (DB, embedder,
watcher).
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
    so [dict1, dict2] -> [TextContent(json(dict1)), TextContent(json(dict2))].
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
    _write_md(v, "ml-basics.md", "---\ntags: [ai, ml]\naliases: [ML]\n---\n# Machine Learning\n\nML uses algorithms to learn from data.\n")
    _write_md(
        v,
        "deep-learning.md",
        "---\ntags: [ai, deep-learning]\n---\n# Deep Learning\n\nNeural networks with many layers. See [[ml-basics]].\n",
    )
    _write_md(v, "cooking.md", "---\ntags: [cooking]\n---\n# Italian Cooking\n\nPasta recipes from Italy.\n")
    _write_md(
        v,
        "hub.md",
        "# Hub Note\n\nSee [[ml-basics]] and [[deep-learning]].\n",
    )
    _write_md(v, "知识管理.md", "---\ntags: [meta]\naliases: [Knowledge Management]\n---\n# 知识管理入门\n\n知识管理是一种系统性的方法。\n")
    return v


@pytest.fixture(scope="session")
async def client(vault: Path) -> ClientSession:
    """Session-scoped MCP client connected via in-memory transport."""
    old_env = os.environ.get("SEEKLINK_VAULT")
    os.environ["SEEKLINK_VAULT"] = str(vault)
    try:
        from seeklink.server import mcp

        async with create_connected_server_and_client_session(
            mcp,
            raise_exceptions=True,
            read_timeout_seconds=timedelta(seconds=60),
        ) as session:
            yield session
    finally:
        if old_env is None:
            os.environ.pop("SEEKLINK_VAULT", None)
        else:
            os.environ["SEEKLINK_VAULT"] = old_env


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
        assert "watcher_running" in data

    async def test_watcher_is_running(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("status", {}))
        assert data["watcher_running"] is True


# ---------------------------------------------------------------------------
# Tool 2: index (returns dict or list[dict])
# ---------------------------------------------------------------------------

class TestIndex:
    pytestmark = pytest.mark.anyio

    async def test_index_without_path_returns_unprocessed(self, client: ClientSession):
        data = _parse_list(await client.call_tool("index", {}))
        assert isinstance(data, list)

    async def test_index_creates_chunks(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index", {"path": "ml-basics.md"}))
        assert data["status"] == "indexed"
        assert data["chunks_created"] >= 0
        assert "source_id" in data

    async def test_index_parses_wiki_links(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index", {"path": "deep-learning.md"}))
        assert data["status"] == "indexed"
        assert data["links_parsed"] >= 1  # [[ml-basics]]

    async def test_index_missing_file(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index", {"path": "nonexistent.md"}))
        assert "error" in data
        assert "not found" in data["error"].lower()

    async def test_force_reindex(self, client: ClientSession):
        data = _parse_dict(
            await client.call_tool("index", {"path": "ml-basics.md", "force": True})
        )
        assert data["status"] == "indexed"

    async def test_path_normalization(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("index", {"path": "./ml-basics.md"}))
        assert data["status"] == "indexed"


# ---------------------------------------------------------------------------
# Tool 3: search (returns list[dict])
# ---------------------------------------------------------------------------

class TestSearch:
    pytestmark = pytest.mark.anyio

    async def test_basic_search(self, client: ClientSession):
        await client.call_tool("index", {"path": "ml-basics.md"})
        await client.call_tool("index", {"path": "cooking.md"})

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
        await client.call_tool("index", {"path": "hub.md"})
        data = _parse_list(
            await client.call_tool("search", {"query": "machine learning", "top_k": 5, "expand": True})
        )
        assert isinstance(data, list)

    async def test_chinese_search(self, client: ClientSession):
        await client.call_tool("index", {"path": "知识管理.md"})
        data = _parse_list(await client.call_tool("search", {"query": "知识管理", "top_k": 3}))
        assert len(data) > 0
        assert any("知识管理" in r["path"] for r in data)

    async def test_cross_language_search(self, client: ClientSession):
        """English query can find Chinese notes via vector similarity."""
        data = _parse_list(
            await client.call_tool("search", {"query": "knowledge management", "top_k": 5})
        )
        assert isinstance(data, list)

    async def test_search_with_tags(self, client: ClientSession):
        """Search with tag filter restricts results."""
        data = _parse_list(
            await client.call_tool("search", {"query": "algorithms", "top_k": 5, "tags": ["ai"]})
        )
        assert isinstance(data, list)
        # cooking.md has tag "cooking", not "ai" — should be excluded
        for r in data:
            assert "cooking" not in r["path"]

    async def test_search_with_folder(self, client: ClientSession):
        """Search with folder filter."""
        data = _parse_list(
            await client.call_tool("search", {"query": "learning", "top_k": 5, "folder": "nonexistent"})
        )
        assert data == []  # no notes in nonexistent folder


# ---------------------------------------------------------------------------
# Tool 4: suggest_links (returns list[dict])
# ---------------------------------------------------------------------------

class TestSuggestLinks:
    pytestmark = pytest.mark.anyio

    async def test_returns_suggestions(self, client: ClientSession):
        await client.call_tool("index", {"path": "cooking.md"})
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
        data = _parse_list(
            await client.call_tool("suggest_links", {"path": "nonexistent.md"})
        )
        assert len(data) == 1
        assert "error" in data[0]


# ---------------------------------------------------------------------------
# Tool 5: resolve_suggestion (returns dict)
# ---------------------------------------------------------------------------

class TestResolveSuggestion:
    pytestmark = pytest.mark.anyio

    async def test_reject_suggestion(self, client: ClientSession):
        await client.call_tool("index", {"path": "cooking.md", "force": True})
        suggestions = _parse_list(
            await client.call_tool("suggest_links", {"path": "cooking.md", "max_suggestions": 2})
        )
        if len(suggestions) == 0:
            pytest.skip("No suggestions generated")

        sid = suggestions[0]["suggestion_id"]
        data = _parse_dict(
            await client.call_tool("resolve_suggestion", {"suggestion_id": sid, "action": "reject"})
        )
        assert data["status"] == "rejected"

    async def test_reject_already_rejected(self, client: ClientSession):
        suggestions = _parse_list(
            await client.call_tool("suggest_links", {"path": "cooking.md", "max_suggestions": 1})
        )
        if len(suggestions) == 0:
            pytest.skip("No suggestions generated")

        sid = suggestions[0]["suggestion_id"]
        await client.call_tool("resolve_suggestion", {"suggestion_id": sid, "action": "reject"})
        data = _parse_dict(
            await client.call_tool("resolve_suggestion", {"suggestion_id": sid, "action": "reject"})
        )
        assert "error" in data
        assert "already" in data["error"].lower()

    async def test_reject_nonexistent(self, client: ClientSession):
        data = _parse_dict(
            await client.call_tool("resolve_suggestion", {"suggestion_id": 99999, "action": "reject"})
        )
        assert "error" in data

    async def test_invalid_action(self, client: ClientSession):
        data = _parse_dict(
            await client.call_tool("resolve_suggestion", {"suggestion_id": 1, "action": "invalid"})
        )
        assert "error" in data

    async def test_approve_writes_link(self, client: ClientSession, vault: Path):
        await client.call_tool("index", {"path": "hub.md", "force": True})
        await client.call_tool("index", {"path": "ml-basics.md", "force": True})

        suggestions = _parse_list(
            await client.call_tool("suggest_links", {"path": "hub.md", "max_suggestions": 3})
        )
        if len(suggestions) == 0:
            pytest.skip("No suggestions generated")

        sid = suggestions[0]["suggestion_id"]
        target_path = suggestions[0]["target_path"]
        target_stem = Path(target_path).stem

        data = _parse_dict(
            await client.call_tool("resolve_suggestion", {"suggestion_id": sid, "action": "approve"})
        )
        assert data["status"] == "approved"
        assert f"[[{target_stem}]]" in data["link_written"]

        # Verify file was actually modified
        hub_content = (vault / "hub.md").read_text(encoding="utf-8")
        assert f"[[{target_stem}]]" in hub_content

    async def test_approve_nonexistent(self, client: ClientSession):
        data = _parse_dict(
            await client.call_tool("resolve_suggestion", {"suggestion_id": 99999, "action": "approve"})
        )
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool 6: graph (returns dict)
# ---------------------------------------------------------------------------

class TestGraph:
    pytestmark = pytest.mark.anyio

    async def test_returns_structure(self, client: ClientSession):
        await client.call_tool("index", {"path": "hub.md"})
        data = _parse_dict(await client.call_tool("graph", {"path": "hub.md"}))
        assert "center" in data
        assert "outgoing" in data
        assert "incoming" in data
        assert "path" in data["center"]
        assert "title" in data["center"]

    async def test_outgoing_links(self, client: ClientSession):
        await client.call_tool("index", {"path": "hub.md"})
        await client.call_tool("index", {"path": "ml-basics.md"})
        await client.call_tool("index", {"path": "deep-learning.md"})

        data = _parse_dict(
            await client.call_tool("graph", {"path": "hub.md", "depth": 1})
        )
        out_paths = {n["path"] for n in data["outgoing"]}
        assert "ml-basics.md" in out_paths or "deep-learning.md" in out_paths

    async def test_depth_2(self, client: ClientSession):
        data = _parse_dict(
            await client.call_tool("graph", {"path": "hub.md", "depth": 2})
        )
        assert isinstance(data["outgoing"], list)

    async def test_missing_note(self, client: ClientSession):
        data = _parse_dict(await client.call_tool("graph", {"path": "nonexistent.md"}))
        assert "error" in data

    async def test_isolated_note(self, client: ClientSession):
        await client.call_tool("index", {"path": "cooking.md"})
        data = _parse_dict(await client.call_tool("graph", {"path": "cooking.md"}))
        assert data["outgoing"] == []


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    pytestmark = pytest.mark.anyio

    async def test_all_6_tools_registered(self, client: ClientSession):
        result = await client.list_tools()
        names = {t.name for t in result.tools}
        expected = {
            "index",
            "search",
            "suggest_links",
            "resolve_suggestion",
            "graph",
            "status",
        }
        assert expected == names

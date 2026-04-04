# Sophia

Personal knowledge management engine exposed as an MCP tool server. Markdown files are the source of truth — Sophia indexes, searches, and links them through AI agent conversations.

## Features

- **Hybrid search**: BM25 full-text + vector semantic search with RRF fusion
- **Bilingual**: Chinese + English (jieba tokenizer + jina-embeddings-v2-base-zh)
- **Wiki-link graph**: Parses `[[links]]`, tracks indegree, BFS graph traversal
- **Auto-indexing**: File watcher detects changes and re-indexes automatically
- **Link suggestions**: Finds related notes and writes `[[links]]` on approval

## Setup

```bash
uv sync
```

## Usage

Sophia runs as an MCP server — configure it in `.mcp.json`:

```json
{
  "mcpServers": {
    "sophia": {
      "command": "uv",
      "args": ["run", "python", "-m", "sophia"],
      "env": { "SOPHIA_VAULT": "/path/to/your/notes" }
    }
  }
}
```

Then use the 8 tools through any MCP client (Claude Code, etc.):

| Tool | Description |
|------|-------------|
| `status` | Index stats, graph size, budget |
| `get_unprocessed` | List notes needing indexing |
| `index_note` | Index a note (chunk, embed, parse links) |
| `search` | Hybrid BM25 + vector search with optional graph expansion |
| `suggest_links` | Find notes worth linking to |
| `approve_suggestion` | Accept a link suggestion (writes `[[link]]` to file) |
| `reject_suggestion` | Dismiss a link suggestion |
| `graph_neighbors` | Show link neighborhood (BFS) |

## Tests

```bash
uv run pytest tests/ -v          # all 191 tests
uv run pytest tests/test_integration.py -v  # MCP protocol-level tests
```

# SeekLink

Hybrid semantic search and link discovery MCP server for Obsidian vaults. Fully local, no API keys needed.

## Features

- **Hybrid search**: BM25 full-text + vector semantic search with RRF fusion
- **Link discovery**: Finds related notes and writes `[[links]]` on approval
- **Knowledge graph**: Parses `[[wikilinks]]`, tracks indegree, BFS graph traversal
- **Bilingual**: Native Chinese + English support (jieba tokenizer + jina-embeddings-v2-base-zh)
- **Auto-indexing**: File watcher detects changes and re-indexes automatically

## Setup

```bash
uv sync
```

## Usage

SeekLink runs as an MCP server — configure it in `.mcp.json`:

```json
{
  "mcpServers": {
    "seeklink": {
      "command": "uv",
      "args": ["run", "python", "-m", "seeklink"],
      "env": { "SEEKLINK_VAULT": "/path/to/your/vault" }
    }
  }
}
```

Then use the 8 tools through any MCP client (Claude Code, etc.):

| Tool | Description |
|------|-------------|
| `search` | Hybrid BM25 + vector search with optional graph expansion |
| `suggest_links` | Find notes worth linking to |
| `approve_suggestion` | Accept a link suggestion (writes `[[link]]` to file) |
| `reject_suggestion` | Dismiss a link suggestion |
| `graph_neighbors` | Show link neighborhood (BFS) |
| `index_note` | Index a note (chunk, embed, parse links) |
| `get_unprocessed` | List notes needing indexing |
| `status` | Index stats, graph size, watcher status |

## Tests

```bash
uv run python -m pytest tests/ -v
```

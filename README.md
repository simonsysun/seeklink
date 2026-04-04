# SeekLink

**Let your AI agent manage your Zettelkasten.**

SeekLink is an MCP server that gives AI assistants (Claude Code, Cursor, etc.) deep access to your markdown vault. It searches, discovers missing connections, and writes `[[wikilinks]]` for you — so your knowledge graph grows as you work.

Built for people who take notes seriously and want an AI that understands their knowledge structure, not just their text.

## What it does

```
You:   "What do I know about MCP protocol?"
Agent: searches vault → finds 8 related notes across topics

You:   "What should this note link to?"
Agent: analyzes content → suggests 4 missing connections with relevance scores

You:   "Approve the first two"
Agent: writes [[wikilinks]] directly into your note file
```

**Six MCP tools:**

| Tool | What it does |
|------|-------------|
| `search` | Four-channel hybrid search: keyword (BM25) + semantic (vector) + knowledge graph (indegree) + title/alias. Fused with Reciprocal Rank Fusion. Filter by tags or folder. |
| `graph` | Explore a note's neighborhood — outgoing links, backlinks, configurable depth |
| `suggest_links` | Find notes that should be linked but aren't. Returns scored suggestions |
| `resolve_suggestion` | Approve (writes `[[link]]` to file) or reject a link suggestion |
| `index` | Index a note, or list unprocessed notes |
| `status` | Vault stats: indexed notes, graph size, watcher status |

## Why SeekLink

**Most MCP servers for Obsidian are file managers.** They read, write, and search text. SeekLink understands your knowledge *structure*: it parses `[[wikilinks]]`, builds a knowledge graph, tracks which notes are central (indegree), and finds connections you missed.

**Chinese is a first-class citizen.** jieba tokenization for keyword search + jina-embeddings-v2-base-zh for semantic search. Not "also supports Chinese" — designed for it. Bilingual vaults (Chinese + English) work out of the box.

**Fully local, headless.** Runs on your machine. No Obsidian plugins required, no API keys for search. Works from the terminal with Claude Code, or as MCP server for any client.

## Install

```bash
uv tool install seeklink
# or
pip install seeklink
```

## Setup

### MCP server (for Claude Code, Cursor, etc.)

Add to your MCP config:

```json
{
  "mcpServers": {
    "seeklink": {
      "command": "seeklink",
      "args": ["serve"],
      "env": { "SEEKLINK_VAULT": "/path/to/your/vault" }
    }
  }
}
```

First run indexes your vault automatically. A file watcher keeps the index up to date.

### CLI

```bash
seeklink search "machine learning" --vault /path/to/vault
seeklink search "知识管理" --vault /path/to/vault --tags ai --top-k 5
seeklink index --vault /path/to/vault
seeklink status --vault /path/to/vault
```

## How search works

SeekLink runs four search channels in parallel and merges results with Reciprocal Rank Fusion:

```
Query: "agent memory systems"
        │
        ├── BM25 (FTS5 + jieba) ──── keyword match ──────── weight 1.0
        ├── Vector (jina-v2-zh) ──── semantic similarity ── weight 1.0
        ├── Indegree ─────────────── well-linked = quality ─ weight 0.3
        └── Title/Alias (FTS5) ──── exact name match ────── weight 3.0
        │
        └── RRF Fusion → ranked results
```

- **Tags filter:** `search("query", tags=["ai", "mcp"])` — only return notes with these tags
- **Folder filter:** `search("query", folder="notes/")` — only search within a folder
- **Expand mode:** `search("query", expand=True)` — include graph neighbors of results

## Frontmatter

SeekLink reads `tags` and `aliases` from YAML frontmatter:

```yaml
---
tags: [ai, machine-learning]
aliases: [ML, Machine Learning]
---
```

Both inline (`[a, b]`) and block list formats supported. Aliases are searchable and used for link resolution — if a note has `aliases: [ML]`, then `[[ML]]` resolves to it.

## Architecture

```
                    ┌────────────────────────────────┐
                    │         MCP Client             │
                    │  (Claude Code, Cursor, ...)    │
                    └──────────┬─────────────────────┘
                               │ stdio / SSE
                    ┌──────────▼─────────────────────┐
                    │      FastMCP Server             │
                    │  6 tools, async handlers        │
                    └──────────┬─────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
   ┌─────────────┐    ┌──────────────┐     ┌──────────────┐
   │   Search     │    │   Ingest     │     │   Watcher    │
   │  4-ch RRF    │    │  chunk+embed │     │  watchfiles  │
   └──────┬──────┘    │  +frontmatter│     │  auto-index  │
          │           └──────┬───────┘     └──────────────┘
          ▼                  ▼
   ┌──────────────────────────────────┐
   │          SQLite + Extensions     │
   │  FTS5 (jieba) │ vec0 (768d)     │
   │  sources, chunks, wiki_links    │
   │  source_tags, fts_sources       │
   └──────────────────────────────────┘
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEEKLINK_VAULT` | `.` | Path to vault root |
| `SEEKLINK_SSE_HOST` | `127.0.0.1` | SSE server bind address |
| `SEEKLINK_SSE_PORT` | `8767` | SSE server port |

## Development

```bash
git clone https://github.com/simonsysun/seeklink
cd seeklink
uv sync --dev
uv run python -m pytest tests/ -q --ignore=tests/test_integration.py
```

217 tests. Python 3.11+.

## Roadmap

- [ ] Graph intelligence: orphan detection, cluster analysis, bridge notes, knowledge gap discovery
- [ ] Cross-encoder reranking for top-k results
- [ ] Lightweight embedding model option (~117MB vs 330MB default)
- [ ] PyPI automated publishing

See [TODOS.md](TODOS.md) for details.

## License

MIT

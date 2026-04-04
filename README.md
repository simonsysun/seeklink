# SeekLink

[![PyPI](https://img.shields.io/pypi/v/seeklink)](https://pypi.org/project/seeklink/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/simonsysun/seeklink/actions/workflows/test.yml/badge.svg)](https://github.com/simonsysun/seeklink/actions)

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
| `search` | Four-channel hybrid search with tag/folder filtering and graph expansion |
| `graph` | Explore a note's neighborhood — outgoing links, backlinks, configurable depth |
| `suggest_links` | Find notes that should be linked but aren't. Returns scored suggestions |
| `resolve_suggestion` | Approve (writes `[[link]]` to file) or reject a suggestion |
| `index` | Index a note, or list unprocessed notes |
| `status` | Vault stats: indexed notes, graph size, watcher status |

## Why not Obsidian's built-in search?

Obsidian's search is keyword-only. It finds exact matches, not related ideas.

SeekLink finds notes that are *semantically similar*, even if they use different words. It knows your knowledge graph: notes that many other notes link to rank higher. And it discovers missing connections — notes that *should* be linked but aren't.

Unlike **Smart Connections** (Obsidian plugin), SeekLink:
- Runs headless — works from Claude Code, Cursor, or any MCP client, without Obsidian open
- Builds a real knowledge graph from your `[[wikilinks]]`, not just vector similarity
- Handles Chinese/English bilingual vaults natively (jieba + jina-embeddings-v2-base-zh)
- Uses four-channel fusion (BM25 + vector + graph + title) instead of vector-only search

## Requirements

- Python 3.11+
- ~330 MB disk for the embedding model (downloaded on first run)
- No API keys needed — everything runs locally

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
- **Expand mode:** `search("query", expand=True)` — follow links from top results for deeper recall

## Frontmatter

SeekLink reads `tags` and `aliases` from YAML frontmatter:

```yaml
---
tags: [ai, machine-learning]
aliases: [ML, Machine Learning]
---
```

Both inline (`[a, b]`) and block list formats supported. Aliases are searchable and used for link resolution — if a note has `aliases: [ML]`, then `[[ML]]` resolves to it.

## How it stores data

Everything lives in `.seeklink/seeklink.db` inside your vault — a single SQLite database with:
- FTS5 full-text index (jieba-tokenized for CJK)
- sqlite-vec for 768-dim vector similarity search
- A wikilink graph (parsed from `[[links]]` in your notes)
- Link suggestion tracking with approve/reject state

Notes are chunked (~400 tokens), embedded with jina-embeddings-v2-base-zh, and indexed incrementally. Delete `.seeklink/` to rebuild from scratch.

While built for personal knowledge bases, SeekLink's search and graph tools work with any corpus of markdown files — including agent-generated notes and shared knowledge bases.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEEKLINK_VAULT` | `.` | Path to vault root |
| `SEEKLINK_SSE_HOST` | `127.0.0.1` | SSE server bind address |
| `SEEKLINK_SSE_PORT` | `8767` | SSE server port |

For SSE transport (e.g. Docker or remote access): `seeklink serve --sse`

## Contributing

```bash
git clone https://github.com/simonsysun/seeklink
cd seeklink
uv sync --dev
uv run python -m pytest tests/ -q --ignore=tests/test_integration.py
```

## License

MIT

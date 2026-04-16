# SeekLink

[![PyPI](https://img.shields.io/pypi/v/seeklink)](https://pypi.org/project/seeklink/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Hybrid semantic search for your markdown vault.**

SeekLink searches your personal knowledge base using four channels in parallel — keyword matching, semantic similarity, knowledge graph, and title/alias lookup — then fuses the results for high-recall, high-precision retrieval. Optional cross-encoder reranking via MLX gives an extra precision boost on Apple Silicon.

Built for people who take notes seriously and want an AI that understands their knowledge structure, not just their text.

## What it does

```
You:   seeklink search "agent memory systems" --vault ~/notes
       → 8 related notes across topics, ranked by relevance

You:   seeklink search "记忆保持力" --vault ~/notes --title-weight 0.5
       → surfaces raw log entries alongside polished articles

You:   seeklink daemon --vault ~/notes
       → resident mode: first search ~2s, every search after ~10ms
```

## Architecture

```
Query: "agent memory systems"
        │
        ├── BM25 (FTS5 + jieba) ──── keyword match ──────── weight 1.0
        ├── Vector (jina-v2-zh) ──── semantic similarity ── weight 1.0
        ├── Indegree ─────────────── well-linked = quality ─ weight 0.3
        └── Title/Alias (FTS5) ──── exact name match ────── weight 1.5
        │
        └── RRF Fusion → top candidates
                │
                └── [optional] Qwen3-Reranker-0.6B (MLX) → precision boost
                        │
                        └── ranked results
```

Four-channel Reciprocal Rank Fusion, with optional cross-encoder reranking for Apple Silicon. Everything runs locally — no API keys, no cloud.

## Requirements

- Python 3.11+
- ~330 MB disk for the embedding model (downloaded on first run)
- ~700 MB disk for the reranker model (if enabled; Apple Silicon recommended)

## Install

```bash
uv tool install seeklink
# or
pip install seeklink
```

## Quick start

```bash
# Search your vault
seeklink search "machine learning" --vault /path/to/vault

# Check index health
seeklink status --vault /path/to/vault

# Index a new or changed file
seeklink index path/to/note.md --vault /path/to/vault

# Full vault rebuild
seeklink index --vault /path/to/vault

# Start the resident daemon (keeps models in memory for fast queries)
seeklink daemon --vault /path/to/vault
```

## CLI reference

### `seeklink search`

```
seeklink search "query" --vault PATH [options]

Options:
  --top-k N          Number of results (default: 10)
  --tags TAG [TAG]   Filter by tags (AND semantics)
  --folder PREFIX    Filter by folder (e.g. "notes/")
  --title-weight F   Override title channel weight (default: 1.5)
                     Raise toward 3.0 for "find the article" queries;
                     lower toward 0.5 for "surface raw moments" queries.
```

### `seeklink daemon`

Starts a Unix-socket daemon that keeps the embedding model (and reranker, if enabled) resident in memory. First query after startup takes ~2s (model warmup); subsequent queries return in ~10ms without reranker or ~2s with reranker.

The daemon auto-spawns when needed — you don't have to start it manually. It never auto-exits; kill it with `kill` or restart your machine.

```
seeklink daemon --vault PATH
```

### `seeklink index`

```
seeklink index [PATH] --vault VAULT

Without PATH: full vault re-index (detects unchanged files via content hash).
With PATH:    index a single file.
```

### `seeklink status`

```
seeklink status --vault PATH
```

Shows index stats and freshness warnings. If files have changed since last index, prints a warning to stderr.

## How search works

SeekLink runs four search channels in parallel and merges results with Reciprocal Rank Fusion:

- **BM25** (FTS5 + jieba): keyword match on chunk content. Handles CJK natively via jieba tokenization.
- **Vector** (jina-embeddings-v2-base-zh): semantic similarity. Finds conceptually related notes even when they use different words or languages.
- **Indegree**: notes that many other notes link to rank higher — a lightweight quality signal from your knowledge graph.
- **Title/Alias** (FTS5): matches against note titles and `aliases` frontmatter. Weight 1.5 gives a modest boost without overwhelming content matches.

### Optional: cross-encoder reranking

When enabled (default on Apple Silicon), the top-20 RRF candidates are re-scored by Qwen3-Reranker-0.6B running on MLX (Metal GPU). This reads each (query, passage) pair with full cross-attention — more accurate than vector similarity alone, at the cost of ~1-2s per query.

Disable with: `export SEEKLINK_RERANKER_MODEL=""`

## Frontmatter

SeekLink works with any markdown file — no special formatting required.

If your notes have YAML frontmatter, SeekLink uses it for extra features:

```yaml
---
tags: [ai, machine-learning]
aliases: [ML, Machine Learning]
---
```

- **Tags** enable filtered search: `seeklink search "query" --tags ai`
- **Aliases** are searchable and used for wikilink resolution — `[[ML]]` resolves to the note with that alias

## How it stores data

Everything lives in `.seeklink/seeklink.db` inside your vault — a single SQLite database with:
- FTS5 full-text index (jieba-tokenized for CJK)
- sqlite-vec for 768-dim vector similarity search
- A wikilink graph (parsed from `[[links]]` in your notes)

Notes are chunked (~400 tokens), embedded with jina-embeddings-v2-base-zh, and indexed incrementally. Delete `.seeklink/` to rebuild from scratch.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEEKLINK_VAULT` | `.` | Path to vault root |
| `SEEKLINK_EMBEDDER_MODEL` | `jinaai/jina-embeddings-v2-base-zh` | Embedding model (fastembed-supported) |
| `SEEKLINK_RERANKER_MODEL` | `mlx-community/Qwen3-Reranker-0.6B-mxfp8` | Reranker model (set to `""` to disable) |

## What changed in v0.2

- **CLI-first**: MCP server removed. All interaction via `seeklink search/index/status/daemon`.
- **Daemon mode**: Unix-socket resident server with auto-spawn. Models stay loaded for fast queries.
- **Reranker**: Qwen3-Reranker-0.6B via MLX on Apple Silicon. Optional, default enabled.
- **Freshness check**: bidirectional mtime scan replaces the file watcher. Warns on stale/new/deleted files.
- **Title weight 1.5**: down from 3.0, so log entries and journal notes compete fairly with titled permanent notes.
- **Leaner deps**: `mcp` and `watchfiles` removed. 4 runtime dependencies instead of 6.

## Contributing

```bash
git clone https://github.com/simonsysun/seeklink
cd seeklink
uv sync --dev
uv run python -m pytest tests/ -q
```

## License

MIT

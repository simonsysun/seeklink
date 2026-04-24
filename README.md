# SeekLink

[![PyPI](https://img.shields.io/pypi/v/seeklink)](https://pypi.org/project/seeklink/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Tests](https://github.com/simonsysun/seeklink/actions/workflows/test.yml/badge.svg)](https://github.com/simonsysun/seeklink/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Hybrid semantic search for your Obsidian-compatible markdown vault.**

SeekLink searches your personal knowledge base using four channels in parallel — keyword matching, semantic similarity, knowledge graph, and title/alias lookup — then fuses the results for high-recall, high-precision retrieval. Optional cross-encoder reranking via MLX gives an extra precision boost on Apple Silicon.

Built for people who take notes seriously and want an AI that understands their knowledge structure, not just their text.

**Ideal for:** Obsidian power users, Zettelkasten / second-brain practitioners, bilingual English + Chinese note-takers, and anyone building local-first RAG on top of a markdown vault. Works with any folder of `.md` files — no Obsidian plugin required.

## What it does

```
You:   seeklink search "agent memory systems" --vault ~/notes
       → 8 related notes across topics, ranked by relevance

You:   seeklink search "记忆保持力" --vault ~/notes --title-weight 0.5
       → surfaces raw log entries alongside polished articles

You:   seeklink daemon --vault ~/notes
       → resident mode: first search ~2s, every search after ~10ms
```

## When to use SeekLink

- Your knowledge base is a folder of markdown files, with or without Obsidian.
- You want search that understands *meaning*, not just exact words — and that surfaces short log entries on equal footing with titled permanent notes.
- You want it to work offline, with no API keys, no cloud, and no data leaving your machine.
- You write in English, Chinese, or both. CJK is a first-class tokenization path (`jieba` as a custom FTS5 tokenizer), not an afterthought.
- You want a CLI an agent can shell out to. Any tool that can `exec()` a binary can use SeekLink — no MCP client or vendor lock-in.

## When **not** to use SeekLink

- Your notes are not markdown (Notion export, Bear, Apple Notes native format, Roam). Convert them first, or pick a tool built for that format.
- You want a hosted, synced, multi-user search service. SeekLink is single-machine, single-user.
- You want a GUI inside Obsidian. SeekLink is a CLI + daemon — there's no Obsidian plugin (yet).
- You need sub-millisecond search over millions of notes. SeekLink targets personal vaults (thousands to low tens-of-thousands of notes). At that scale it's fast; beyond, you want a real search service (Typesense, Meilisearch, Elastic).
- You're on Windows. macOS and Linux are tested; Windows should mostly work (Python is portable, the Unix-socket daemon isn't) but is not a supported path.

## How it compares

| Tool | Format | Semantic search | CJK | Local-only | Knowledge-graph signal |
|---|---|---|---|---|---|
| **SeekLink** | any `.md` vault | ✅ BM25 + vector + title, RRF-fused, optional MLX reranker | ✅ jieba tokenizer | ✅ | ✅ wikilink indegree |
| Obsidian core search | Obsidian vault | ❌ (keyword only) | partial | ✅ | ❌ |
| Obsidian semantic-search plugins (Smart Connections, Copilot, etc.) | Obsidian vault | ✅ (vector only, mostly OpenAI API) | varies | usually ❌ (API) | ❌ |
| `ripgrep` | any text | ❌ (keyword only, literal regex) | ✅ | ✅ | ❌ |
| Build-your-own RAG (LlamaIndex, LangChain) | any | ✅ (you wire it) | depends on embedder | depends | depends |

SeekLink's niche: **markdown-native, hybrid keyword + semantic, CJK-first, fully local, with the knowledge-graph signal your wikilinks already encode** — available as a single `pip install`.

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

## Support & limitations

| | Supported | Notes |
|---|---|---|
| Python | 3.11, 3.12, 3.13, 3.14 | Tested in CI |
| OS (core + daemon) | macOS, Linux | Unix socket at `~/.rhizome/seeklink.sock` |
| OS (Windows) | unsupported | Python code is portable; Unix-socket daemon is not |
| File format | `.md` (markdown) | Frontmatter optional |
| Wikilink syntax | `[[note]]`, `[[alias]]`, Obsidian-compatible | |
| Embedder | `jina-embeddings-v2-base-zh` (default, 330 MB) | Swap via `SEEKLINK_EMBEDDER_MODEL` (fastembed-supported) |
| Reranker | Qwen3-Reranker-0.6B via MLX (700 MB) | Apple Silicon only. Disable with `SEEKLINK_RERANKER_MODEL=""` |
| Multi-vault daemon | single-vault only | Pass `--vault` to run a one-shot against a different vault (forces cold-start) |
| Concurrent access | one daemon per machine | Multiple CLI clients may share one daemon |

## Install

```bash
uv tool install seeklink
# or
pip install seeklink
```

## Quick start

```bash
# One-shot: search a specific vault (always cold-start, no daemon)
seeklink search "machine learning" --vault /path/to/vault

# Full vault rebuild (cold-start)
seeklink index --vault /path/to/vault
```

**Recommended for daily use** — set your default vault once, then every search auto-uses the fast daemon:

```bash
export SEEKLINK_VAULT=/path/to/vault
seeklink search "machine learning"
# First call after a cold boot: ~2s (spawns the daemon, loads the embedder).
# Every call after that: ~10ms without reranker, ~0.5s with reranker.
```

The daemon stays resident across terminal sessions until you `kill` it or restart. No manual startup needed — `seeklink search / index / status` auto-spawn it when missing.

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

**You almost never run this directly.** When you run `seeklink search / index / status` without `--vault`, the CLI tries the daemon socket first and auto-spawns a daemon on cold machines. The daemon uses `SEEKLINK_VAULT` (or cwd) as its vault. It never auto-exits — kill it with `kill` or restart your machine.

Passing `--vault` always uses cold-start instead of the daemon, because the daemon binds to a single vault at startup. Multi-vault daemon support is tracked in TODOS.md.

```
seeklink daemon --vault PATH    # foreground, for debugging
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

### `seeklink get`

Print a line range of a vault file directly to stdout. Designed for agents that have a search hit like `notes/fsrs.md:42` and want to read a precise window without fetching the whole file.

```
seeklink get PATH[:LINE] [-l N] [--vault PATH]

  seeklink get notes/fsrs.md              # entire file
  seeklink get notes/fsrs.md:120          # 100 lines starting at line 120
  seeklink get notes/fsrs.md:120 -l 40    # 40 lines starting at line 120
  seeklink get notes/fsrs.md -l 50        # first 50 lines
```

Line numbers match `search` output. CRLF files print with universal newlines. Path escapes (`../..`) are rejected.

## How search works

SeekLink runs four search channels in parallel and merges results with Reciprocal Rank Fusion:

- **BM25** (FTS5 + jieba): keyword match on chunk content. Handles CJK natively via jieba tokenization.
- **Vector** (jina-embeddings-v2-base-zh): semantic similarity. Finds conceptually related notes even when they use different words or languages.
- **Indegree**: notes that many other notes link to rank higher — a lightweight quality signal from your knowledge graph.
- **Title/Alias** (FTS5): matches against note titles and `aliases` frontmatter. Weight 1.5 gives a modest boost without overwhelming content matches.

### Why title weight is 1.5 (not higher)

Many personal knowledge bases contain a mix of **titled articles** (permanent notes, literature reviews) and **untitled process notes** (daily logs, journal entries, quick captures). A high title weight systematically buries untitled content — even when it's the most relevant result for the query. The default of 1.5 keeps title matching useful for precise `[[alias]]` lookups while letting content-based matches compete on their own merits. Override with `--title-weight` per query if needed.

### Title-gated rerank blending (v0.3+)

When the reranker is enabled, a cross-encoder (`Qwen3-Reranker-0.6B` on MLX, ~1-2s per query) re-scores the top-20 RRF candidates for precision. SeekLink applies **title-gated position blending** on top of this:

- **If the title channel's best match is in the candidate pool**, blend `alpha · normalized_rrf + (1 - alpha) · rerank_score` with `alpha = 0.60/0.50/0.40` by rank bucket. This protects exact title / alias hits from being demoted by a content-focused reranker.
- **Otherwise** (no strong title signal), the reranker score is used directly — same as pre-v0.3 behavior. This lets the reranker correct poor first-stage ordering.

On the built-in 22-query blind test, this improved mean MRR from 0.932 to 0.977 vs pure-reranker-override, with zero regressions. See `tests/blind/` for the methodology.

Disable reranking entirely with: `export SEEKLINK_RERANKER_MODEL=""`

### Results carry line numbers

Every `search` result returns `path:line_start-line_end` pointing at the best chunk within the current on-disk file. Agents can pipe that into `seeklink get` for a precise window read — no need to slurp entire files just to see context around a hit.

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

## What changed in v0.3

- **Title-gated rerank blending**: when an exact title / alias hit drives rank 1, protect it from reranker demotion; otherwise fall back to pure reranker. Measured MRR gain of +4.5 pp over v0.2 on a 22-query blind test, with no regressions. See "How search works" above.
- **Line-range retrieval**: `search` results now include `line_start` / `line_end`, and a new `seeklink get PATH[:LINE] -l N` command prints line-precise windows. Agents can find-then-read without slurping whole files.
- **Cold-start / daemon parity fix**: cold-start `seeklink search` now constructs a `Reranker()` and passes it to the search pipeline. Previously the same query returned different rankings depending on whether the daemon was running.
- **Frontmatter-aware line mapping**: chunk offsets (stored against frontmatter-stripped body) are remapped to full-file line numbers, so `search` + `get` report lines the way you'd see them in a text editor.
- **Blind-test framework** at `tests/blind/`: 32-file corpus + 22 ground-truth queries + runner that measures Recall@10 / MRR / latency. Used to validate v0.3 before tagging; gates v0.4 (query expansion) the same way.

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

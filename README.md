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
       → resident mode: first search ~2s; warm reranker-on ~1-2s,
         reranker-disabled ~10ms
```

## When to use SeekLink

- Your knowledge base is a folder of markdown files, with or without Obsidian.
- You want search that understands *meaning*, not just exact words — and that surfaces short log entries on equal footing with titled permanent notes.
- You want it to work offline, with no API keys, no cloud, and no data leaving your machine.
- You write in English, Chinese, or both. CJK is a first-class tokenization path (`jieba` as a custom FTS5 tokenizer), not an afterthought.
- You want a CLI an agent can shell out to. Anything that can `exec()` a binary — editor plugins, shell scripts, LLM agents — can use SeekLink.

## When **not** to use SeekLink

- Your notes are not markdown (Notion export, Bear, Apple Notes native format, Roam). Convert them first, or pick a tool built for that format.
- You want a hosted, synced, multi-user search service. SeekLink is single-machine, single-user.
- You want a GUI inside Obsidian. SeekLink is a CLI + daemon; there is no Obsidian plugin.
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
# 1. Build the index first (searching an un-indexed vault returns no hits).
seeklink index --vault /path/to/vault

# 2. Search it.
seeklink search "machine learning" --vault /path/to/vault
```

**Recommended for daily use** — set your default vault once, then every `search` / `index` auto-uses the fast daemon:

```bash
export SEEKLINK_VAULT=/path/to/vault
seeklink index                         # first run: builds the index
seeklink search "machine learning"
# First search after a cold boot: ~2s (spawns the daemon, loads the embedder).
# Warm reranker-on path (default): ~1-2s per query.
# Warm reranker-disabled path: ~10ms per query.
```

The daemon stays resident across terminal sessions until you `kill` it or restart. `seeklink search` and `seeklink index` auto-spawn it when missing; `seeklink status` is always cold-start (it only reads SQLite stats, no model load) and `seeklink get` is a direct filesystem read (no daemon involved either).

## For agents

SeekLink is designed to be shelled out to by LLM agents and RAG pipelines, not just typed by humans. The contract:

1. **Index once, then search.** Agents should check `seeklink status --vault PATH` before first use; if `Notes: 0` or stale, run `seeklink index --vault PATH`.
2. **Search.** `seeklink search "query" --vault PATH --top-k N`. Each result line is:
   ```
     SCORE  PATH[:LINE]  TITLE
              <indented content preview, up to 120 chars, one line>
   ```
   `PATH` is relative to the vault root. `:LINE` is 1-indexed and points at the best chunk's first line in the current on-disk file; omitted when the match is title-only on a stale file. `SCORE` is `rerank_blended` when reranker is active, else raw RRF (scales differ, don't compare across configurations).
3. **Read the window.** `seeklink get PATH:LINE -l N` prints `N` lines starting at `LINE`; `seeklink get PATH:LINE -C N` prints `N` lines before and after `LINE`. No DB lookup, no daemon round-trip — direct filesystem read with universal-newline translation. This is how you avoid slurping whole files after a search hit.
4. **Query shape.** Raw CJK is fine (jieba handles segmentation). `[[alias]]` / exact-title queries get title-gated protection — if your query IS a note title or alias, results anchor on it. No need for wildcards; BM25 + vector + title fuse automatically.
5. **Exit codes.** `0` on success (including "no results"). `1` on vault-resolution error, missing file (`get`), or unrecoverable config mismatch. No other codes used.
6. **Structured output.** Use `seeklink search "query" --json` or `seeklink status --json` when an agent needs machine-readable stdout. Search JSON includes `path`, `title`, `content_preview`, `score`, `indegree`, `line_start`, and `line_end` per result. For hot loops, agents can still connect to the daemon's Unix socket at `~/.rhizome/seeklink.sock` directly (length-prefixed JSON protocol, see `seeklink/daemon.py`).

See also `llms.txt` at the repo root for a compressed version of this contract.

## CLI reference

### `seeklink search`

```
seeklink search "query" --vault PATH [options]

Options:
  --top-k N          Number of results (default: 10)
  --tags TAG [TAG]   Filter by tags (AND semantics)
  --folder PREFIX    Filter by folder (e.g. "notes/")
  --json             Emit machine-readable JSON instead of text output
  --title-weight F   Override title channel weight (default: 1.5)
                     Raise toward 3.0 for "find the article" queries;
                     lower toward 0.5 for "surface raw moments" queries.
```

### `seeklink daemon`

Starts a Unix-socket daemon that keeps the embedding model (and reranker, if enabled) resident in memory. First query after startup takes ~2s for model warmup; warm queries return in ~1-2s with the reranker on (default) or ~10ms with `SEEKLINK_RERANKER_MODEL=""`.

**You almost never run this directly.** `seeklink search` and `seeklink index` auto-spawn a daemon on cold machines when `--vault` is not passed. `seeklink status` is always cold-start (no model load). `seeklink get` is a direct filesystem read (no daemon). The daemon uses `SEEKLINK_VAULT` (or cwd) as its vault and never auto-exits — kill it with `kill` or restart your machine.

Passing `--vault` always uses cold-start instead of the daemon, because the daemon binds to a single vault at startup.

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
seeklink status --vault PATH [--json]
```

Shows index stats and freshness warnings. If files have changed since last index, prints a warning to stderr. `--json` keeps those warnings on stderr and writes one machine-readable object to stdout.

### `seeklink get`

Print a line range of a vault file directly to stdout. Designed for agents that have a search hit like `notes/fsrs.md:42` and want to read a precise window without fetching the whole file.

```
seeklink get PATH[:LINE] [-l N | -C N] [--vault PATH]

  seeklink get notes/fsrs.md              # entire file
  seeklink get notes/fsrs.md:120          # 100 lines starting at line 120
  seeklink get notes/fsrs.md:120 -l 40    # 40 lines starting at line 120
  seeklink get notes/fsrs.md:120 -C 20    # 20 lines before and after line 120
  seeklink get notes/fsrs.md -l 50        # first 50 lines
```

Line numbers match `search` output. `-C/--context` requires `PATH:LINE` and cannot be combined with `-l/--lines`. CRLF files print with universal newlines. Path escapes (`../..`) are rejected.

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

On the bundled 22-query pilot (see `tests/blind/`), mean MRR moved from 0.932 to 0.977 vs pure-reranker-override with no per-query regressions. Sample size is a pilot, not a statistically powered benchmark — contributions of larger labeled corpora are welcome.

Disable reranking entirely with: `export SEEKLINK_RERANKER_MODEL=""`

### Results carry line numbers

Every `search` result is anchored to a specific line range in the current on-disk file. Two surfaces expose this:

- **CLI text output** (`seeklink search ...`): each line is `SCORE  PATH:LINE_START  TITLE`, followed by an indented content preview. Feed `PATH:LINE_START` straight into `seeklink get` to read the window around the hit.
- **CLI / daemon JSON** (`seeklink search --json` or one request via the Unix socket): each result also carries `line_end` and `indegree` fields for callers that want the full span.

Line numbers are mapped back through the frontmatter strip that happens at index time, so they match what you'd see in a text editor on disk.

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

## Release history

See [CHANGELOG.md](CHANGELOG.md).

## Contributing

```bash
git clone https://github.com/simonsysun/seeklink
cd seeklink
uv sync --dev
uv run python -m pytest tests/ -q
```

## License

MIT

# SeekLink

[![PyPI](https://img.shields.io/pypi/v/seeklink)](https://pypi.org/project/seeklink/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Tests](https://github.com/simonsysun/seeklink/actions/workflows/test.yml/badge.svg)](https://github.com/simonsysun/seeklink/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SeekLink is a local semantic search CLI for Markdown vaults. It indexes a folder
of `.md` files, searches with hybrid keyword + vector retrieval, and returns
line-anchored results that humans and agents can read with simple shell
commands.

It is built for personal knowledge bases, Obsidian-compatible vaults, bilingual
English/Chinese notes, and local agent workflows. It is also a useful search
layer for Markdown wiki patterns such as Andrej Karpathy's
[llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):
an agent can search existing pages, read precise line windows, then update the
wiki without sending the vault to a hosted service.

Everything runs locally. No API key. No cloud search service. No Obsidian plugin
required.

## Install

```bash
uv tool install seeklink
# or
pip install seeklink
```

## Quick Start

```bash
# 1. Build the index first.
seeklink index --vault /path/to/vault

# 2. Search it.
seeklink search "machine learning" --vault /path/to/vault
```

Daily use is simpler if you set a default vault:

```bash
export SEEKLINK_VAULT=/path/to/vault
seeklink index
seeklink search "agent memory systems"
seeklink get notes/agent-memory-patterns.md:1 -C 20
```

`seeklink search` and `seeklink index` auto-use a resident daemon when
`SEEKLINK_VAULT` is set and `--vault` is not passed. The daemon keeps the
embedder and optional reranker in memory. `seeklink status` and `seeklink get`
always stay cold-start: status only reads SQLite metadata, and get reads the
file directly from disk.

## Output

Text search output is stable:

```text
  SCORE  PATH[:LINE]  TITLE
           <content preview, one line, up to 120 chars>
```

- `PATH` is relative to the vault root.
- `LINE` is 1-indexed and points to the best matching chunk in the current file.
- Exit code is `0` for success, including no results, and `1` for vault/config
  errors or missing files.
- Scores are useful for sorting within one query. Do not compare scores across
  reranker-enabled and reranker-disabled runs.

Use JSON when an agent needs structured output:

```bash
seeklink search "agent memory systems" --vault PATH --json
seeklink status --vault PATH --json
```

## Common Commands

### Search

```bash
seeklink search "query" --vault PATH [options]
```

Options:

```text
--top-k N          Number of results. Default: 10.
--json             Emit one machine-readable JSON object.
--tags TAG [TAG]   Filter by tags. AND semantics.
--folder PREFIX    Filter by vault-relative folder prefix.
--rerank-k N|auto  Rerank candidate budget. Default: auto.
--no-rerank        Skip cross-encoder reranking for this query.
--title-weight F   Override title/alias/heading channel weight. Default: 1.5.
```

### Get

Read a precise file window without using the database or daemon:

```bash
seeklink get notes/spaced-repetition.md
seeklink get notes/spaced-repetition.md:12
seeklink get notes/spaced-repetition.md:12 -l 40
seeklink get notes/spaced-repetition.md:12 -C 20
```

`-l/--lines` prints lines starting at `LINE`. `-C/--context` prints lines before
and after `LINE`, grep-style. Path escapes such as `../..` are rejected.

### Status

```bash
seeklink status --vault PATH
seeklink status --vault PATH --json
```

Status reports index counts, model names, SQLite WAL status, and freshness
warnings. It does not load the embedding or reranking models.

### Index

```bash
seeklink index --vault PATH
seeklink index path/to/file.md --vault PATH
```

Full-vault indexing skips unchanged files by content hash. Single-file indexing
updates one Markdown file.

### Daemon

```bash
seeklink daemon --vault PATH
```

You normally do not run this directly. `search` and `index` auto-spawn and
auto-restart the daemon when appropriate. Passing `--vault` to `search` or
`index` forces a one-shot cold-start path because the daemon is bound to one
vault at startup.

## How Search Works

SeekLink fuses four channels with Reciprocal Rank Fusion:

| Channel | Purpose |
|---|---|
| BM25 / FTS5 | Exact words, code terms, acronyms, CJK lexical matches |
| Vector search | Semantic matches across different wording |
| Title / aliases / headings | Exact note and section lookup |
| Wikilink indegree | Small graph-quality prior from existing `[[links]]` |

The default embedder is `jinaai/jina-embeddings-v2-base-zh` through
`fastembed`. CJK full-text search uses a jieba FTS5 tokenizer when the local
Python/SQLite build can safely register it; otherwise SeekLink falls back to
SQLite's built-in trigram tokenizer instead of crashing.

On Apple Silicon, SeekLink can rerank candidates with
`mlx-community/Qwen3-Reranker-0.6B-mxfp8`. Reranking is local and optional. Use
`--no-rerank` for one query or set `SEEKLINK_RERANKER_MODEL=""` to disable it
globally.

## Frontmatter

Markdown frontmatter is optional. When present, SeekLink uses it for tags and
aliases:

```yaml
---
tags: [ai, memory]
aliases: [LLM memory, agent memory]
---
```

- `tags` support filtered search: `seeklink search "memory" --tags ai`
- `aliases` are indexed for search and used when resolving wikilinks

## Storage

SeekLink writes one SQLite database inside the vault:

```text
/path/to/vault/.seeklink/seeklink.db
```

The database contains source metadata, chunks, FTS5 tables, sqlite-vec vectors,
and a wikilink graph. Delete `.seeklink/` and run `seeklink index` to rebuild.

## Supported

| Area | Status |
|---|---|
| Python | 3.11, 3.12, 3.13, 3.14 |
| OS | macOS and Linux |
| Windows | Not supported as a first-class path |
| File format | Markdown `.md` |
| Vault style | Plain folder or Obsidian-compatible vault |
| CJK | Native path via jieba, with trigram fallback on static SQLite builds |
| Reranker | Apple Silicon via MLX; disabled elsewhere |
| Daemon | Single vault per machine |

## Not For

- Hosted or synced multi-user search.
- Non-Markdown sources without conversion.
- A GUI or Obsidian plugin.
- Sub-millisecond search over millions of notes.
- Cloud embedding or reranking APIs.

## Agent Notes

Agents can use SeekLink through ordinary subprocess calls:

```bash
seeklink status --vault PATH
seeklink index --vault PATH
seeklink search "query" --vault PATH --json
seeklink get PATH:LINE -C 20 --vault PATH
```

For hot loops, the daemon exposes a length-prefixed JSON protocol over the Unix
socket at `~/.rhizome/seeklink.sock`. Most agents should prefer the CLI JSON
surface unless they specifically need socket-level latency.

See [llms.txt](llms.txt) for the compact agent contract.

## Evaluation

Search-quality tests live in `tests/blind/`; the method is documented in
[docs/blind-test.md](docs/blind-test.md). Release claims should be backed by
the bundled fixture queries or by clearly labeled private-vault measurements.

## Contributing

```bash
git clone https://github.com/simonsysun/seeklink
cd seeklink
uv sync --dev
uv run python -m pytest tests/ -q
```

Keep runtime dependencies small, keep public docs user-facing, and add a
`CHANGELOG.md` entry for user-visible changes.

## License

MIT

# SeekLink

<!-- mcp-name: io.github.simonsysun/seeklink -->

[English](README.md) · [中文](README.zh.md)

[![PyPI](https://img.shields.io/pypi/v/seeklink)](https://pypi.org/project/seeklink/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Tests](https://github.com/simonsysun/seeklink/actions/workflows/test.yml/badge.svg)](https://github.com/simonsysun/seeklink/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SeekLink is a local semantic search CLI and optional read-only MCP stdio server
for Markdown vaults. It indexes a folder of `.md` files, searches with hybrid
keyword + vector retrieval, and returns line-anchored results that humans and
agents can read with simple shell commands.

It is built for personal knowledge bases, Obsidian-compatible vaults, bilingual
English/Chinese notes, and local agent workflows. MCP clients such as Claude
Code, Cursor, and VS Code can call the same read-only search/get/status/doctor
surface through `seeklink[mcp]`. It is also a useful search layer for Markdown
wiki patterns such as Andrej Karpathy's
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

For Apple Silicon reranking support, install the optional MLX extra:

```bash
uv tool install "seeklink[mlx]"
# or
pip install "seeklink[mlx]"
```

For Model Context Protocol (MCP) clients such as Claude Code, Cursor, or VS
Code, install the optional MCP extra:

```bash
uv tool install "seeklink[mcp]"
# or
pip install "seeklink[mcp]"
```

SeekLink requires Python's `sqlite3` module to be linked against SQLite
3.45 or newer with FTS5 enabled. `seeklink status --vault PATH` checks this and
prints a clear error if the runtime SQLite is too old.

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

`seeklink search` and single-file `seeklink index path/to/file.md` use a
resident daemon when `--vault` is not passed. The daemon keeps the embedder and
optional reranker warm in memory; on macOS this appears as a local `Python`
process. It is local-only, uses a Unix socket, and does not open a network port
or call a cloud service. By default it exits after 15 minutes of inactivity.
Full-vault `seeklink index` runs in-process so progress stays on stderr and the
final `Done:` summary stays on stdout. `seeklink status` and `seeklink get`
always stay cold-start: status only reads SQLite metadata, and get reads the
file directly from disk. Use `--no-daemon`, `SEEKLINK_NO_DAEMON=1`, or an
explicit `--vault PATH` when a script needs a one-shot cold-start path.

MCP users follow the same first step: build the index with
`seeklink index --vault PATH` before registering the MCP server.

## Output

Text search output is stable:

```text
  SCORE  PATH[:LINE]  TITLE
           <content preview, one line, up to 120 chars>
```

- `PATH` is relative to the vault root.
- `LINE` is 1-indexed and points to the best matching chunk in the current file.
- Exit code is `0` for success, including no results; `1` for runtime
  vault/config/file errors detected by SeekLink; and `2` for command-line usage
  errors from argument parsing.
- Scores are useful for sorting within one query. Do not compare scores across
  reranker-enabled and reranker-disabled runs.

Use JSON when an agent needs structured output:

```bash
seeklink search "agent memory systems" --vault PATH --json
seeklink status --vault PATH --json
seeklink doctor --vault PATH --json
seeklink daemon status --json
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
--no-daemon        Force an in-process search instead of using the daemon.
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

Status reports index counts, model names, index-configuration compatibility,
SQLite WAL status, and freshness warnings. It does not load the embedding or
reranking models.

### Doctor

```bash
seeklink doctor --vault PATH
seeklink doctor --vault PATH --json
```

Doctor checks Python, SQLite, the local database, index compatibility, daemon
state, and optional MLX availability. It does not download or load models, but
may initialize the local SeekLink database/schema if missing.

### MCP

The optional Model Context Protocol (MCP) adapter lets agent clients discover
and call SeekLink's read-only tools directly. The CLI keeps working
independently; MCP is another surface for the same retrieval path, not a
replacement.

```bash
seeklink mcp --vault PATH
```

Install it with `seeklink[mcp]`. Build the index with the CLI first:
`seeklink index --vault PATH`. The MCP adapter is read-only and exposes four
tools: `search`, `get`, `status`, and `doctor`. It does not expose `index`,
write notes, use HTTP/OAuth, or route through the Unix-socket daemon. Run one
MCP server per vault. `search` keeps its text summary compact with paths and
line anchors; result previews stay in structured content for agents that need
them. `status` and `doctor` may initialize or migrate the local SeekLink schema
when an existing `.seeklink/seeklink.db` needs it, but they do not index or
modify Markdown notes. If your MCP client does not inherit your shell `PATH`,
use the absolute path from `which seeklink` in the examples below.

Claude Code:

```bash
claude mcp add --transport stdio --scope project seeklink \
  -- seeklink mcp --vault /ABS/PATH/TO/VAULT
```

Cursor `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "seeklink": {
      "type": "stdio",
      "command": "seeklink",
      "args": ["mcp", "--vault", "/ABS/PATH/TO/VAULT"]
    }
  }
}
```

VS Code `.vscode/mcp.json`:

```json
{
  "servers": {
    "seeklink": {
      "type": "stdio",
      "command": "seeklink",
      "args": ["mcp", "--vault", "/ABS/PATH/TO/VAULT"]
    }
  }
}
```

### Index

```bash
seeklink index --vault PATH
seeklink index path/to/file.md --vault PATH
```

Full-vault indexing skips unchanged files by content hash unless the stored
index was built with a different embedder, vector dimension, or chunker
configuration, in which case SeekLink rebuilds the derived index contents.
Single-file indexing updates one Markdown file only when the existing index
configuration is compatible.

### Daemon

```bash
seeklink daemon status
seeklink daemon stop
seeklink daemon restart
seeklink daemon pid
seeklink daemon run --vault PATH
```

You normally do not need to start the daemon manually. `search` and single-file
`index` auto-spawn and auto-restart it when appropriate, then it exits after
`SEEKLINK_DAEMON_IDLE_TIMEOUT` seconds of inactivity. The default is 900 seconds
(15 minutes); set it to `0`, `off`, `false`, or `no` to keep the daemon warm
until stopped.

Full-vault `index` still runs in-process for progress output. Passing `--vault`
to `search` or single-file `index` forces a one-shot cold-start path because the
daemon is bound to one vault at startup. `--no-daemon` and
`SEEKLINK_NO_DAEMON=1` also force the same cold-start path. Use
`seeklink daemon status` to inspect the warm process and `seeklink daemon stop`
to release its memory immediately.

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

The default vector dimension is 768. Advanced custom-embedder experiments can
set `SEEKLINK_EMBEDDING_DIM`, but it must match the embedder output and requires
a full `seeklink index` rebuild.

On Apple Silicon, SeekLink can rerank candidates with
`mlx-community/Qwen3-Reranker-0.6B-mxfp8` when installed with `seeklink[mlx]`.
Reranking is local and optional; if MLX is unavailable, SeekLink falls back to
first-stage hybrid RRF ranking. Use `--no-rerank` for one query or set
`SEEKLINK_RERANKER_MODEL=""` to disable it globally.

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
| SQLite | Python `sqlite3` linked against SQLite 3.45+ with FTS5 |
| OS | macOS and Linux |
| Windows | Not supported as a first-class path |
| File format | Markdown `.md` |
| Vault style | Plain folder or Obsidian-compatible vault |
| CJK | Native path via jieba, with trigram fallback on static SQLite builds |
| Reranker | Optional `seeklink[mlx]` extra on Apple Silicon; disabled elsewhere |
| Daemon | Single vault per machine |
| MCP | Optional `seeklink[mcp]` stdio adapter, one server per vault |

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

MCP clients can use the optional read-only adapter:

```bash
seeklink mcp --vault PATH
```

To make an agent choose SeekLink for a Markdown vault, add this to the
project's `AGENTS.md`, `CLAUDE.md`, or editor rules:

```text
When you need to search or inspect this Markdown vault, use SeekLink for
semantic retrieval:

1. Run `seeklink status --vault PATH --json`.
2. If no index exists or files changed, run `seeklink index --vault PATH`.
3. Run `seeklink search "QUERY" --vault PATH --json`.
4. Read exact context with `seeklink get PATH:LINE -C 20 --vault PATH`.

If SeekLink is registered as an MCP server in this client, prefer the
`search`, `get`, `status`, and `doctor` MCP tools over shelling out to the CLI.

Prefer SeekLink for conceptual, cross-language, tag/folder-filtered, or
Obsidian-style note searches. Use rg for exact literal searches.
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

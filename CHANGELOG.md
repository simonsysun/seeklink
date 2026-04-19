# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Daemon-first CLI dispatch: `seeklink search / index / status` auto-spawn the daemon on first invocation when `--vault` is not passed, then serve subsequent calls in ~10ms. Pass `--vault` to force cold-start.
- PyPI `keywords` for discoverability.
- `Issues` and `Changelog` project URLs.

### Changed
- CI matrix expanded to Python 3.11, 3.12, 3.13, 3.14 to match declared classifier support.
- CI "Verify install" step no longer masks failures with `|| true`; it now exercises `seeklink --help` and `seeklink status` against a temp vault.

### Fixed
- README claimed daemon auto-spawn but the CLI actually went direct to cold-start on every invocation. Behavior now matches docs.

## [0.2.0] - 2026-04-16

### Added
- Unix-socket daemon mode with eager-loaded embedder and optional MLX reranker (`seeklink daemon`). Models stay resident between queries for ~10ms round-trips.
- Optional cross-encoder reranking via Qwen3-Reranker-0.6B on MLX (Apple Silicon). Default-enabled, disable with `SEEKLINK_RERANKER_MODEL=""`.
- Freshness check: bidirectional mtime scan reports stale, new, and deleted files on cold-start `status` / `search`.
- Configurable title-channel RRF weight via `--title-weight` flag per query.

### Changed
- **CLI-first architecture.** MCP server (`seeklink serve`) removed. All interaction is via CLI subcommands or Unix-socket daemon.
- Title-channel default weight lowered from 3.0 to 1.5 so untitled content (daily logs, journal notes) competes fairly with titled articles.
- Runtime dependencies trimmed from 6 to 4 (removed `mcp`, `watchfiles`).

### Removed
- MCP server transport. Agents that used MCP should invoke the CLI via `subprocess` or connect to the daemon socket via `seeklink.cli_client`.

## [0.1.0] - 2026-04-04

### Added
- Initial public release.
- Four-channel hybrid search: BM25 (FTS5 + jieba) + vector (jina-embeddings-v2-base-zh) + knowledge-graph indegree + title/alias FTS, fused via Reciprocal Rank Fusion.
- SQLite-backed storage (`.seeklink/seeklink.db`) with sqlite-vec for 768-dim vectors and FTS5 for keyword and title search.
- Wikilink parser for Obsidian-style `[[note]]` and `[[alias]]` graph edges.
- Native CJK tokenization via jieba registered as a custom FTS5 tokenizer.
- MCP server transport (`seeklink serve`) — removed in v0.2.0.

[Unreleased]: https://github.com/simonsysun/seeklink/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.0
[0.1.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.1.0

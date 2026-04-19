# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-04-19

### Fixed
- PyPI build failed because `pyproject.toml` carried both the SPDX expression `license = "MIT"` and the legacy `License :: OSI Approved :: MIT License` classifier, which modern setuptools rejects under PEP 639. v0.2.1 was tagged on GitHub but never published to PyPI. v0.2.2 is the first release in this line that downstream users can actually `pip install`. No functional changes from v0.2.1 — same daemon-first dispatch, same vault/model guards, same metadata.

## [0.2.1] - 2026-04-18 — **tagged only, not on PyPI**

> This tag was published on GitHub but never made it to PyPI: a duplicate license declaration broke the build. Everything described below shipped in **[0.2.2]**. Do not pin to `seeklink==0.2.1`.

### Added
- Daemon-first CLI dispatch: `seeklink search` and `seeklink index` auto-spawn the daemon on first invocation when `--vault` is not passed, then serve subsequent calls in ~10ms. Pass `--vault` to force cold-start.
- `cli_client.call()` preflights the daemon's vault and model config (embedder + reranker) before reusing it, so a stale daemon bound to a different `SEEKLINK_VAULT` / `SEEKLINK_EMBEDDER_MODEL` / `SEEKLINK_RERANKER_MODEL` cannot silently serve or mutate the wrong database.
- `seeklink status` now prints the configured embedder and reranker names (computed from env, without importing the heavy modules).
- PyPI `keywords` and richer classifiers for discoverability.
- `Issues` and `Changelog` project URLs.
- README: "Ideal for" tagline, "When to use / When not to use" sections, "How it compares" table, and "Support & limitations" matrix.
- `llms.txt` at the repo root for LLM-assisted discovery.
- `CHANGELOG.md` (this file).

### Changed
- CI matrix expanded to Python 3.11, 3.12, 3.13, 3.14 to match declared classifier support.
- CI "Verify install" step no longer masks failures with `|| true`; it now exercises `seeklink --help` and `seeklink status` against a temp vault.
- `seeklink status` now always uses the cold-start path. It only reads SQLite stats + freshness, so routing it through the daemon was wasting a full embedder + reranker warmup (up to a 700MB download on first ever run) just to print a few numbers.
- PyPI `description` rewritten to name Obsidian compatibility explicitly.

### Fixed
- README claimed daemon auto-spawn but the CLI actually went direct to cold-start on every invocation. Behavior now matches docs.
- Prevented a stale daemon bound to a different vault or started with a different embedder/reranker from silently serving incorrect results after the user switched `SEEKLINK_VAULT` / `SEEKLINK_EMBEDDER_MODEL` / `SEEKLINK_RERANKER_MODEL`. On mismatch the CLI falls back to cold-start; an auto-respawn follow-up is tracked in TODOS.md.

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

[Unreleased]: https://github.com/simonsysun/seeklink/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.2
[0.2.1]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.1
[0.2.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.0
[0.1.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.1.0

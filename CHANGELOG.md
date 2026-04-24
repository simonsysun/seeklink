# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-23

### Added
- **Title-gated rerank blending.** When the title-channel's best match is in the rerank candidate pool, blend `alpha · normalized_rrf + (1 − alpha) · rerank_score` with `alpha = 0.60/0.50/0.40` by rank bucket. This protects confident exact-title / alias hits (e.g. searching `Zettelkasten`, `RRF`, `遗忘曲线`) from being demoted by a content-focused reranker. When no title hit is present, the reranker takes over fully — same as pre-v0.3 behavior — so poor first-stage ordering (e.g. `把文档切块放进向量库` where the correct answer is at RRF rank 11) is still recoverable. Measured on a 22-query blind test vs the same baseline: mean MRR 0.932 → 0.977 (+4.5 pp), mean Recall@10 unchanged, zero regressions. See `docs/v0.3-plan.md` for the iteration history (Options A / B / C) and `tests/blind/results/` for the raw JSON.
- **Line-range retrieval end-to-end.**
  - `SearchResult` now carries `line_start` and `line_end` (1-indexed, inclusive), computed by mapping chunk `char_start` / `char_end` back through the frontmatter strip to on-disk line numbers.
  - Daemon search responses include `line_start` / `line_end`.
  - CLI `_print_search_results` displays `path:line_start  title` so `path:LINE` can be piped straight into `seeklink get`.
  - New `seeklink get PATH[:LINE] [-l N]` command reads the current on-disk file with universal-newline translation and prints the requested line range. Defaults: whole file (no `:LINE`), 100 lines starting at `LINE` (no `-l`), N lines (`-l`). Rejects path escapes, warns on beyond-EOF and `LINE < 1`.
  - Helper `body_offset_to_file_line(full_text, body_char_offset) → int` handles the frontmatter offset; also correct when the frontmatter was deleted from disk after indexing.
- **Blind-test framework** at `tests/blind/`: 32-file CJK+EN corpus (`tests/corpus/`), 22 ground-truth queries (`tests/blind/queries.yaml`), runner (`tests/blind/run.py`) that cold-starts seeklink once per invocation, warms the reranker, measures `recall_at_10` / `mrr` / `latency_ms` / `p95`. Three configurations: A (baseline), B (v0.4 query expansion — not yet implemented), C (hand-crafted expansion, RRF-fused; upper bound). Used to validate this release; gates v0.4.
- **v0.3 plan + blind-test framework docs** at `docs/v0.3-plan.md` and `docs/blind-test.md`.
- **FRONTMATTER_RE** is now a public export from `seeklink.ingest` so the search layer can reuse the same regex for offset mapping.

### Fixed
- **Cold-start vs daemon parity.** Cold-start `seeklink search` (the path triggered when `--vault` is passed or the daemon is unreachable) now constructs a `Reranker()` and passes it to `search()`, matching the daemon's behavior. Previously the same query returned different rankings depending on whether a daemon happened to be running — a silent correctness bug. `Reranker()` construction is safe on platforms without MLX (Linux, Intel macOS) because the instance self-disables at model-load time.
- **Line-range accounting for newline-terminated files.** `seeklink get file:LINE` on a file that ends with `\n` no longer miscounts the trailing newline as an extra logical line. Line 6 of a 5-line (newline-terminated) file now correctly emits the `beyond-EOF` warning instead of returning a blank line.
- **Title-only match with deleted file.** When a search result references a source whose file has been removed from disk (title-only match via alias to a stale source), `compute_lines_for_results` no longer returns `line_start=1` — it degrades to `0/0` so agents aren't handed a `path:1` that won't resolve. Consistent with other missing-file paths.

### Dev
- PyYAML added as a dev dependency (required by `tests/blind/run.py`).
- Test suite: 185 → 203 tests (18 new). 3 for position-aware blending, 13 for `get` command + `body_offset_to_file_line` helper, 3 for end-to-end `SearchResult.line_start/line_end` population, 1 for trailing-newline EOF accounting. All green.

### Deferred to v0.3.1+
- `SEEKLINK_DEBUG=1` blended-score logging (proposed in v0.3 plan, skipped to avoid scope creep).
- Per-result `mtime > indexed_at` drift warnings on the daemon path (cold-start already warns globally via `check_freshness`). Daemon-side follow-up tracked in `TODOS.md`.
- Linux reranker via llama.cpp / GGUF (`QuantFactory/Qwen3-Reranker-0.6B-GGUF` exists; wiring it into seeklink lives on after v0.3).

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

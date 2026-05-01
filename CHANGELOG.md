# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Full-vault indexing now embeds in smaller batches to reduce long-tail embedding stalls on real Markdown vaults.
- `seeklink index --vault PATH` now prints full-vault progress to stderr while keeping the final `Done:` summary on stdout.

## [0.4.0] - 2026-04-29

### Added
- `seeklink search --json` and `seeklink status --json` emit stable machine-readable stdout for agents that should not scrape the human text format.
- `seeklink get PATH:LINE -C N` prints a grep-style context window around a search hit while preserving direct filesystem reads and path-escape protection.
- `seeklink search --rerank-k N`, `--rerank-k auto`, and `--no-rerank` let callers trade precision for latency per query without changing the global reranker configuration.
- Source-level metadata search now indexes Markdown headings alongside note titles and frontmatter aliases, improving section-name queries without changing the output format.

### Changed
- Full-vault indexing now embeds chunks in length-sorted batches instead of one file at a time, improving first-run indexing throughput on real Markdown vaults while preserving single-file indexing behavior and the existing SQLite schema.
- The MLX reranker now caps each passage to the first 200 tokens before scoring, reducing warm-query latency on long chunks while preserving the full result preview and `seeklink get` output.
- `seeklink search` now defaults to `--rerank-k auto`, using a smaller reranker budget for ordinary lookups while preserving deeper reranking for filtered and technical CJK queries.
- README, `llms.txt`, and search-evaluation docs now focus on concise usage, agent contracts, and release-quality measurement guidance instead of product positioning or internal experiment notes.
- Existing indexes migrate to schema v3 and mark sources unprocessed so the next `seeklink index` pass can populate heading metadata.

### Fixed
- Python builds that compile `_sqlite3` as a built-in module with hidden SQLite symbols now fall back to SQLite's built-in trigram FTS tokenizer instead of letting `sqlitefts` cross SQLite library boundaries and segfault.
- Filtered searches now rank BM25 and source-metadata candidates inside the requested tag/folder scope, so relevant filtered notes are not dropped just because unfiltered notes filled the global first-stage limit.
- Exact title, alias, and heading lookups now keep the source-metadata winner at rank 1 after reranking, while broader heading matches still allow the content reranker to reorder results.
- `seeklink search --rerank-k N` now limits the number of candidates passed to the cross-encoder even when `N` is lower than `--top-k`; the remaining results keep first-stage RRF order.
- `seeklink search` and `seeklink index` now auto-restart a stale daemon when its vault, embedder, or reranker config no longer matches the caller, avoiding repeated cold-start fallbacks after switching vaults or model settings.

### Dev
- Added a CLI contract smoke test that runs the documented status, index, search, JSON, and get workflow against the bundled `tests/corpus` vault before release.
- The blind-test runner now records nDCG@10, Precision@5, MAP@10, reranker-budget metadata, and first-stage channel diagnostics for config A.
- Refreshed `tests/blind/results/` with v0.4 release-quality snapshots only. On the bundled 22-query fixture, config A reports mean Recall@10 0.985, MRR 0.977, nDCG@10 0.901, and p95 latency 2124 ms on a local Apple Silicon run.

## [0.3.2] - 2026-04-23

Repository cleanup pass. No code changes affecting runtime behavior
from 0.3.1; this release removes internal R&D artifacts from the public
surface so the repo reads as a shipped tool rather than a work log.

### Changed
- Consolidated the 0.3.0 / 0.3.1 narrative into a single release entry (this one). The earlier entries described the same code twice with process detail that did not belong in public release notes.
- Trimmed `tests/blind/results/` to release-quality baseline, shipping, and upper-bound measurements. Intermediate iteration results removed.
- Tightened internal code comments and test docstrings so they describe current behavior rather than the iteration history that produced it.
- README metric claims explicitly labeled as "pilot" with sample size.

### Removed
- `docs/v0.3-plan.md` — internal design scratch that should not have shipped in the public repo. The shipped design is documented in the "How search works" README section and in code comments.

## [0.3.1] - 2026-04-23

### Added
- **Title-gated rerank blending.** When the title / alias channel produces a confident match in the rerank candidate pool, SeekLink blends a normalized first-stage score with the reranker output so exact title or alias hits (`Zettelkasten`, `RRF`, `遗忘曲线`, `[[alias]]`) are preserved at rank 1 instead of being demoted by a content-focused reranker. When no title signal is present, the reranker takes over fully — same behavior as v0.2.x — so poor first-stage ordering is still recoverable. On the bundled 22-query pilot (see `tests/blind/`): mean MRR 0.932 → 0.977, mean Recall@10 unchanged, no per-query regressions. Sample size is intentionally a pilot; larger labeled corpora are welcome.
- **Line-range retrieval.** `SearchResult` now carries 1-indexed inclusive `line_start` / `line_end` fields mapped through the indexer's frontmatter strip back to on-disk line numbers. CLI `search` prints `SCORE  PATH:LINE  TITLE` so agents can pipe the hit into a precise window read. A new `seeklink get PATH[:LINE] [-l N]` command performs that window read directly from the filesystem — no DB round-trip, no daemon involvement, universal-newline translation, path-escape rejection.
- **Cold-start `search` reranker parity.** `seeklink search --vault PATH` (the cold-start path) now constructs a reranker and passes it to the search pipeline, matching the daemon. Before this change, the same query returned different rankings depending on whether a daemon happened to be running.
- **Agent-first documentation.** New "For agents" section in the README (minimum workflow, output contract, exit codes, query-shape hints, daemon JSON fallback). `llms.txt` rewritten as an explicit contract.
- **Blind-test framework** at `tests/blind/`: 32-file bilingual (CJK + English) fixture corpus (`tests/corpus/`), 22 ground-truth queries (`tests/blind/queries.yaml`), runner that cold-starts once per invocation and measures `recall_at_10` / `mrr` / `latency_ms` / `p95`. Three configurations: `A` (current baseline), `B` (planned query expansion — not yet shipped), `C` (hand-crafted expansion, RRF-fused; upper bound). Used to gate this release.

### Fixed
- **`seeklink get` trailing-newline accounting.** `get FILE:LINE` on a newline-terminated file no longer counts the trailing `\n` as an extra logical line. `get FILE:6` on a 5-line file correctly emits the beyond-EOF warning instead of returning a blank line.
- **Title-only match with missing file.** If a search surfaces a title-only match whose file has been deleted from disk, `SearchResult.line_start` / `line_end` now remain at `0` rather than returning a misleading `path:1`.

### Changed
- **`SearchResult` gains `line_start` and `line_end` (default `0`).** Backward compatible for existing callers; populated when `search()` is called with `vault_root`.
- **`FRONTMATTER_RE`** is now a public export from `seeklink.ingest` (was `_FRONTMATTER_RE`), so the search layer can reuse it for offset mapping. The underscore-prefixed name still aliases it for backward compatibility.

### Dev
- PyYAML added as a dev dependency (required by the blind-test runner).
- Test suite: 185 → 204 tests (19 new).

### Deferred
- `SEEKLINK_DEBUG=1` blended-score logging.
- Per-result `mtime > indexed_at` drift warnings on the daemon path (cold-start already warns globally via `check_freshness`).
- Linux reranker via llama.cpp / GGUF.

### Superseded
- This release supersedes the same-day `0.3.0` tag, which had the same code but shipped with inaccurate README content (quick-start ordering, latency numbers, `seeklink status` description). If you are pinning a version, use `0.3.1`.

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

[Unreleased]: https://github.com/simonsysun/seeklink/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/simonsysun/seeklink/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/simonsysun/seeklink/releases/tag/v0.3.2
[0.3.1]: https://github.com/simonsysun/seeklink/releases/tag/v0.3.1
[0.3.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.3.0
[0.2.2]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.2
[0.2.1]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.1
[0.2.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.2.0
[0.1.0]: https://github.com/simonsysun/seeklink/releases/tag/v0.1.0

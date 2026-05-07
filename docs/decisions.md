# Active Project Decisions

This file records active project constraints that are not obvious from code.
Do not mirror every commit here; use commit messages and git history for normal
history. Keep this file short enough that future agents can read it at session
start when they need project context.

## 2026-05-06: Treat SeekLink as a public product surface

SeekLink is a public OSS tool shipped to PyPI. Repository content should help a
user understand, install, trust, and operate the tool. Do not commit process
notes, abandoned design narratives, internal review blow-by-blow, or intermediate
experiment output as product documentation.

## 2026-05-06: Keep release docs executable and user-facing

`README.md`, `README.zh.md`, `llms.txt`, and CLI examples are user contracts.
When setup, run, verification, daemon behavior, or agent-facing output changes,
update the relevant docs in the same change and verify shown commands where
practical. `CHANGELOG.md` and `CHANGELOG.zh.md` should describe user-visible
changes only; internal-only work belongs in terse Dev notes only when it helps
maintainers understand shipped behavior.

## 2026-05-06: Search-quality changes require blind-test evidence

Any change that can affect retrieval ranking, answer line spans, chunking,
tokenization, embedder defaults, reranker defaults, or indexed metadata should
run the blind-test framework described in `docs/blind-test.md`. Commit only
release-quality result snapshots under `tests/blind/results/`; keep local sweeps,
private-vault outputs, and exploratory measurements in ignored `.scratch/`.

## 2026-05-06: Keep dependencies and public history disciplined

Runtime dependencies should stay minimal because SeekLink is installed by users.
Ask before adding production dependencies. Public commits should describe
user-observable changes, use a clear Conventional Commit-style subject when
practical, and avoid AI coauthor trailers unless Simon explicitly asks for one.
For non-trivial changes, use the commit body for why/decision/rejected/verification
rather than file-by-file summaries.

## 2026-05-06: Cleanups must be reversible and verified

Before deleting tracked docs, fixtures, benchmarks, or generated-looking files,
search for references with `git grep` and read any callers. Delete in narrow
rounds, then verify immediately with the smallest relevant project checks. If a
scratch artifact is almost useful as user documentation, rewrite it as product
documentation instead of deleting; if in doubt, keep it local or ask Simon.

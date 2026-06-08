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

## 2026-06-07: SeekLink paused — retrieval is a commodity; the principle for any revival

SeekLink aimed to be the local semantic-retrieval organ for a personal bilingual
Markdown knowledge base and for AI agents. A 2026-06 review concluded:

- The niche is occupied. `qmd` (github.com/tobi/qmd; ~26k stars as of 2026-06,
  actively developed, built by Shopify's CEO) appears to cover the same combination
  SeekLink offers — local hybrid keyword+vector, line-anchored output, CLI + MCP,
  multilingual — and a company clone (OceanBase `seekx`) adds jieba CJK.
  Local-Markdown retrieval is now a commodity.
- SeekLink has no external users and no real vault to serve. Its only eval is a
  22-query / 32-note single-author fixture at a quality ceiling (Recall@10 ~0.985),
  and its CJK rerank-budget router hardcodes that fixture's vocabulary
  (search.py:34-63) so it does not generalize. No quality claim was ever validated.

Decision: stop active development. Use `qmd` off-the-shelf as the retrieval organ;
put effort into the separate AI memory system (the novel work).

Guiding principle for ANY revival — **division of labor**: the retriever owns
recall + cheap signals (provenance, recency, precise PATH:LINE locators); the
CALLING AGENT owns task-conditioned precision. Do not push task/conversation
awareness or learned global ranking weights into the retriever. Keep the non-learned
RRF (static weights, no task-conditioned or trained ranking), a rebuildable index,
and the blind-test gate.

Revive only if real `qmd` usage over a real vault exposes a repeated, specific
failure `qmd` cannot cheaply fix — candidates: strict read-only enforcement,
calibrated abstention ("no strong match"), or memory-layer-aware retrieval. Then
build it as a thin wrapper, not a full engine. See `TODOS.md` for the specific
directions.

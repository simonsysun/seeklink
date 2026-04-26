# Query expansion blind test framework

## Purpose

Validate that query expansion actually improves real search quality
against a CJK-first personal vault **before** committing to shipping it.
Query expansion introduces:

- A new generative model (Qwen3-0.6B, ~500 MB quantized).
- A new runtime dependency (MLX or `llama.cpp`).
- Extra query-time latency (dependent on backend and prompt).

The cost is only justified if the win over baseline is real and
consistent. If the test shows "indistinguishable from baseline" or
"regresses on too many queries", the feature should be cancelled rather
than shipped.

## Three configurations

| ID | Pipeline | What it measures |
|----|----------|------------------|
| **A** | seeklink search + reranker (daemon-path behavior) | Baseline. Must match product behavior — the runner constructs a real `Reranker()` and passes it to `search()`, same as `daemon.py` does. |
| **B** | seeklink + Qwen3-0.6B expansion | Ship candidate |
| **C** | seeklink + hand-crafted expansion, RRF-fused | Upper bound |

A and C are fixed points. B's distance from C tells us how much the small
model is leaving on the table; B's distance from A tells us whether
shipping B is worth it at all.

**Important**: config A is *not* "raw seeklink search without reranker".
The runner constructs a real `Reranker()` and passes it to `search()`,
so A reproduces what a user gets from the default `seeklink search`
invocation.

## Runtime requirement

`tests/blind/run.py` imports `yaml`. Install dev dependencies with
`uv sync --dev` before running.

## Test data format

`tests/blind/queries.yaml`:

```yaml
- query: "记忆保持力"
  intent: "find notes about long-term memory retention techniques"
  expected_paths:
    - "notes/fsrs-algorithm.md"
    - "notes/spaced-repetition.md"
    - "logs/2026-W15.md"
  relevance:
    "notes/fsrs-algorithm.md": 3
    "notes/spaced-repetition.md": 3
    "logs/2026-W15.md": 2
    "notes/forgetting-curve.md": 1
  tags: [cjk, common]
  expansion:
    - "间隔重复 遗忘曲线 FSRS"
    - "how to retain memory long term"
    - "通过间隔算法优化长期记忆保留"

- query: "Zettelkasten vs 卡片盒笔记"
  intent: "compare methodology in user's literature review"
  expected_paths:
    - "notes/zettelkasten.md"
  tags: [cjk-en-mixed]
  expansion:
    - "atomic notes permanent notes"
    - "卡片盒笔记法"
```

### How to build this file

**20-30 queries total.** Fewer than 15 and single-query noise dominates the
averages.

1. Real-user queries only. Pull from shell history, your own notes, or
   memory. No synthetic queries.
2. For each, list 2-5 `expected_paths` you'd be annoyed if not in top 10.
   Hard must-hit semantics — not "would be nice".
3. Optionally add `relevance:` grades for nDCG: `3` = direct answer,
   `2` = strong supporting context, `1` = related but insufficient.
   `expected_paths` default to grade `3`; extra relevance labels do not
   affect Recall@10 or MRR.
4. **Skip queries where a substring of the query exactly matches a note
   title.** Those hit the title channel trivially and test nothing about
   expansion. Prefer queries where notes use different vocabulary than the
   query itself.
5. Fill in `expansion:` with 2-3 hand-crafted alternates: lexical form,
   semantic paraphrase, hypothetical answer sentence (HyDE style).
6. Tag each query for slicing: `cjk`, `english`, `cjk-en-mixed`, `long`,
   `short`, `ambiguous`, `technical`, `common`.

**Ground-truth stability**: commit `queries.yaml` alongside a vault-state
marker (e.g. the vault's git commit SHA if it's versioned). If you re-run
against an edited vault, note the drift.

## Metrics

For each `(query, config)` pair (recorded by the runner):

- `hits` — top-10 result paths in rank order
- `titles` — top-10 titles (for the human blind scorer)
- `snippets` — top-10 content previews (for the human blind scorer)
- `scores` — fused scores (not directly compared across configs)
- `relevance` — graded labels used for nDCG (`expected_paths` default to
  grade `3`)
- `latency_ms` — wall-clock for the full query call chain (model load
  excluded — runner initializes once and warms up)
- `rerank_k` — first-stage candidate count passed to the reranker (`0`
  when reranking is disabled, or `"auto"` for query-sensitive routing)
- `resolved_rerank_k` — actual numeric candidate budget used for this query
  (`5` or `20` for `auto`, `0` when reranking is disabled)
- `rerank_k_reason` — why `auto` chose that budget (`title`,
  `cjk_technical`, `filter`, `default`, or `fixed`)
- `recall_at_10` — fraction of `expected_paths` in top-10
- `mrr` — reciprocal rank of first expected hit in top-10 (0 if none)
- `ndcg_at_10` — graded nDCG when `relevance:` labels exist, otherwise
  binary nDCG over `expected_paths`

Aggregates:

- Mean `recall_at_10`, mean `mrr`, mean `latency_ms`, p95 `latency_ms`
- Per-query delta (`B - A`, `C - A`, `C - B`) → find where expansion hurts
- Per-tag breakdown (computed offline from `results` JSON) — especially
  `cjk` vs `english` to catch asymmetric wins/regressions

## Runner

`tests/blind/run.py` loads `queries.yaml`, runs each query against one
config, writes results JSON. Invocation:

```bash
# Baseline — works today
python tests/blind/run.py \
    --config A \
    --queries tests/blind/queries.yaml \
    --vault /path/to/vault \
    --out tests/blind/results/A.json

# Ship candidate — requires v0.4 expansion hook (runner raises until then)
python tests/blind/run.py --config B ...

# Upper bound — uses hand-crafted `expansion:` field, fuses by RRF
python tests/blind/run.py --config C ...

# Diagnostic: baseline without reranker (NOT the official baseline)
python tests/blind/run.py --config A --no-rerank ...

# Diagnostic: latency / quality sweep for reranker budget
python tests/blind/run.py --config A --rerank-k 5  --out tests/blind/results/A_rerank5.json ...
python tests/blind/run.py --config A --rerank-k 10 --out tests/blind/results/A_rerank10.json ...
python tests/blind/run.py --config A --rerank-k 20 --out tests/blind/results/A_rerank20.json ...
python tests/blind/run.py --config A --rerank-k auto --out .scratch/rerank-sweep/A_auto.json ...
```

Runner:

- Initializes `init_app(vault)` exactly **once** per invocation (before the
  query loop). Warms the embedder, FTS tokenizer, and when enabled the
  reranker with dummy calls so the first measured latency isn't
  model/cache/tokenizer startup.
- Passes `--rerank-k` through to `search()`. Default `20` matches product
  behavior; lower values and `auto` are diagnostic latency / quality probes.
- Records the per-query resolved reranker budget so `auto` sweeps can be
  audited without guessing which queries used 5 vs. 20 candidates.
- Closes the DB once, in a `finally` block.
- Records per-query latency using `time.perf_counter()`. Model-load time
  is excluded by warmup.

For a human blind-scoring pass, run a small shuffling script over
`results/A.json`, `results/B.json`, `results/C.json`: per query, present
one query plus 5 results (path + title + snippet) at a time without
labels, and record a 1-5 score per config.

## Acceptance criteria for shipping B (query expansion feature)

**All five must hold for B to ship:**

1. **Mean Recall@10 of B ≥ Recall@10 of A + 0.10** (at least +10 pp lift)
2. **B regresses on ≤ 20% of queries** (Recall@10(B) < Recall@10(A))
3. **Per-tag protection**: for each of the following tag buckets, B's mean
   Recall@10 within that bucket must be ≥ A's mean within that bucket − 0.05:
   - `cjk` (pure Chinese queries)
   - `cjk-en-mixed`
   - `english`
   - `short` (≤ 2 tokens)
   - `long` (≥ 6 tokens)
   This catches "B crushes English queries, destroys CJK" — not OK for a
   CJK-first vault.
4. **Human blind score mean of B ≥ A + 0.5** on 1-5 scale
5. **`p95(B) ≤ min(3 × p95(A), 2500 ms)`** — whichever bound is lower
   binds. On current M3 + reranker-on hardware, `p95(A)` is ~1-2 s, so
   `3 × p95(A)` is 3-6 s and the 2500 ms **absolute ceiling** is the real
   gate. The `3×` term only starts binding if A itself gets faster (e.g.
   future reranker optimization drops A's p95 below ~833 ms). Writing both
   bounds protects against either regression direction.

**Cancel criteria** (any one triggers "do not ship B"):

- B's Recall@10 is within noise of A (`|B - A| < 0.05` on mean, and no tag
  bucket shows `> 0.10` improvement)
- Per-tag failure: any tag bucket regresses by `> 0.05` on Recall@10
- Latency p95 exceeds either the relative or absolute ceiling
- Human score shows mixed signal: B is higher on some queries and lower on
  others with no tag-level explanation

**Sanity ceiling check**: if C is also indistinguishable from A
(`|C - A| < 0.05`), expansion is not the problem — retrieval or embedder
is. Look at the embedder or retrieval channels instead.

## Notes on setup

- **Ground truth scope.** Use hard must-hit semantics only. Weaker
  signal is more subjective and blurs the numbers.
- **Expansion prompt template.** Base Qwen3-0.6B is not fine-tuned for
  query rewriting, so config B will need a few-shot prompt (variants
  produced, one per line, bounded length). Draft once and commit
  alongside `queries.yaml`. Optionally constrain output format via a
  GBNF grammar if using llama.cpp.
- **Inference backend for B.** MLX (macOS) or llama.cpp (cross-platform)?
  → Run both, pick the one that hits the p95 budget. Record which.
- **Randomness.** Qwen3 at temperature 0.7 is non-deterministic. Propose:
  temperature 0.3, no manual seed, but log each query's actual expansions
  in the `expansions_used` field for reproducibility. For B's final
  acceptance run, consider `N=3` and report median.

## Out of scope for this framework

- Automated labeling (ground truth is labeled by hand).
- CI-integrated regression (this is a pre-release gate, not a
  continuous monitor).
- Cross-tool comparison (different corpora are incommensurable).

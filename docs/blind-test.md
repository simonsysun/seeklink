# Search Quality Evaluation

SeekLink includes a small blind-test framework for retrieval changes. It is not
a public leaderboard. It is a release guard: when search behavior changes, run
the same labeled queries before and after the change and compare quality and
latency.

The bundled fixture lives in:

```text
tests/corpus/              # small bilingual Markdown vault
tests/blind/queries.yaml   # labeled queries
tests/blind/run.py         # runner
tests/blind/results/       # release-quality reference outputs only
```

Use `.scratch/` for local sweeps, downloaded datasets, temporary runs, and
private-vault measurements. Do not commit intermediate experiment output.

## Configs

| Config | Meaning |
|---|---|
| `A` | Current product behavior: hybrid search plus the default reranker path when available |
| `B` | Candidate query-expansion path, reserved for future experiments |
| `C` | Hand-written expansion upper bound using the `expansion:` field in `queries.yaml` |

Config `A` is the release baseline. Config `C` answers whether expansion has
headroom. Config `B` should not ship unless it beats `A` on quality without
breaking the latency budget.

## Query Format

Each query has hard expected paths and optional graded relevance:

```yaml
- query: "记忆保持力"
  intent: "find notes about long-term memory retention techniques"
  expected_paths:
    - "notes/fsrs-algorithm.md"
    - "notes/spaced-repetition.md"
  relevance:
    "notes/fsrs-algorithm.md": 3
    "notes/spaced-repetition.md": 3
    "logs/2026-W15.md": 2
  tags: [cjk, common]
  expansion:
    - "间隔重复 遗忘曲线 FSRS"
    - "how to retain memory long term"
```

Rules:

- Use real user queries when possible.
- `expected_paths` are hard must-hit labels for Recall/MRR.
- `relevance` grades are optional and used for nDCG.
- Grades: `3` direct answer, `2` strong supporting context, `1` related,
  `0` irrelevant.
- Tags should identify slices such as `cjk`, `english`, `mixed`, `short`,
  `long`, `technical`, `alias`, `filtered`, or `logs`.

## Metrics

The runner records per-query:

- `hits`, `titles`, `snippets`, and `scores`
- `recall_at_10`
- `mrr`
- `precision_at_5`
- `average_precision_at_10`
- `ndcg_at_10`
- `last_expected_rank`
- `latency_ms`
- reranker budget metadata
- first-stage channel diagnostics for config `A`

The aggregate output includes mean Recall@10, MRR, nDCG@10, latency, and p95
latency.

Use the diagnostics to separate two failure types:

- Candidate-generation failure: the expected note never appears in first-stage
  candidates.
- Ranking failure: the expected note appears but is pushed down by fusion or
  reranking.

## Running It

Install dev dependencies first:

```bash
uv sync --dev
```

Run the current product baseline:

```bash
uv run python tests/blind/run.py \
  --config A \
  --queries tests/blind/queries.yaml \
  --vault tests/corpus \
  --out .scratch/blind/A_current.json
```

Run without reranking for diagnostics:

```bash
uv run python tests/blind/run.py \
  --config A \
  --no-rerank \
  --queries tests/blind/queries.yaml \
  --vault tests/corpus \
  --out .scratch/blind/A_no_rerank.json
```

Run a reranker-budget sweep:

```bash
uv run python tests/blind/run.py \
  --config A \
  --rerank-k 5 \
  --queries tests/blind/queries.yaml \
  --vault tests/corpus \
  --out .scratch/blind/A_rerank5.json
```

Run the hand-written expansion upper bound:

```bash
uv run python tests/blind/run.py \
  --config C \
  --queries tests/blind/queries.yaml \
  --vault tests/corpus \
  --out .scratch/blind/C_upper_bound.json
```

Only copy a result into `tests/blind/results/` when it is the final
release-quality measurement you want users and contributors to read.

## When To Run

Run the blind test when a change can affect ranking or line spans:

- `seeklink/search.py`
- `seeklink/ingest.py`
- `seeklink/chunker.py`
- `seeklink/tokenizer.py`
- embedding model defaults
- reranker model defaults or reranker scoring behavior
- schema changes that affect indexed text or metadata

For pure CLI formatting, daemon lifecycle, or docs changes, targeted pytest
coverage is usually enough.

## Shipping Criteria For Expansion

Query expansion is not part of the default product path. If a future expansion
candidate uses config `B`, require all of the following before shipping it:

1. Mean Recall@10 improves by at least 10 percentage points over config `A`.
2. Fewer than 20% of queries regress on Recall@10.
3. Main query slices do not regress by more than 5 percentage points.
4. Human blind review prefers `B` by at least 0.5 points on a 1-5 scale.
5. p95 latency is at most `min(3 * p95(A), 2500ms)`.

If config `C` is also close to `A`, expansion probably is not the right lever;
look at chunking, metadata, filters, or the embedder instead.

## Public vs Private Results

Public repo:

- fixture vault
- labeled fixture queries
- runner code
- final baseline / shipping / upper-bound reference results

Private or local only:

- downloaded external datasets
- large public-vault mirrors used for stress tests
- private-vault labels and outputs
- intermediate sweeps
- exploratory research notes

Use `.scratch/` for the private/local side.

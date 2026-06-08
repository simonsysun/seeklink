# Deferred work

Known limitations and possible future work. Not commitments — items here
ship if and when they become worth the cost.

> Status: SeekLink is paused (see `docs/decisions.md`, 2026-06). If you fork or
> revive it, read `docs/decisions.md` first — the recall-vs-precision division of
> labor there should constrain everything below.

## Highest-value directions (if revived)

These came from designing SeekLink as the retrieval organ for a separate,
file-first AI memory system, then adversarially stress-testing the plan. Ordered
by honesty, not excitement.

1. **Prove it on a real vault before building anything.** The only eval is a
   22-query / 32-note single-author fixture at a ceiling (Recall@10 ~0.985), so it
   cannot detect improvement or regression. Before tuning, build a labeled set on a
   *real* vault (50-100+ queries) with the four slices that matter: cross-lingual
   (EN query → 中文 note and vice versa), conceptual / no-shared-keyword,
   recency/staleness, and abstention (queries with no good answer). A public or
   synthetic corpus is a weak proxy and tends to manufacture false confidence.

2. **Surface provenance/metadata; do not hardcode a schema.** `SearchResult`
   (search.py:300-313) exposes path/title/content/score/indegree and a line range,
   but no source-provenance, layer, verification, or validity fields. A consuming
   agent needs those to prefer trustworthy + current evidence. Do it as a *generic
   passthrough* of declared frontmatter fields (and replace the hand-rolled regex
   frontmatter reader, `_parse_frontmatter` in ingest.py, with a real YAML parse —
   note that adds a runtime dependency; `pyyaml` is dev-only today) — do NOT bake a
   specific layer vocabulary (e.g. raw/card/profile) into the engine. Recency is
   nearly free: `fs_modified_at`/`indexed_at` are already stored (models.py).

3. **A calibrated "no strong match" / abstention signal.** No competitor (incl.
   `qmd`) surfaces "nothing here is strongly relevant," and an agent needs it to
   say "I don't know" rather than use a weak hit. Honest caveat: this is NOT cheap —
   RRF scores are not calibrated, so a trustworthy signal needs score calibration
   the engine lacks plus a labeled no-answer set; and it only has value once a
   consumer (an agent loop / memory system) exists to act on it. Defer until then.

4. **Layer-scoped retrieval.** Make folder=layer a documented convention so a
   caller can default to summary "cards" and pull raw transcripts only on demand
   via `get PATH:LINE`. Mind this known limitation: sqlite-vec filters AFTER the
   KNN (`db.search_vec`); `_resolve_vector_limit` (search.py) mitigates it with an
   adaptive global K but can still drop hits beyond the K/cap (default 5000) on a
   large vault — and this path gets hotter under layer-scoping.

5. **Delete the overfit rerank-budget router** (search.py:34-63). It hardcodes the
   test fixture's own vocabulary (fsrs/bm25/rrf/anki/向量/嵌入…); on any other corpus
   it silently degrades, and it actively misleads measurement. Replace with a
   uniform rerank budget. Worth doing even if the project stays paused.

6. **Default the cross-encoder reranker OFF on the agent path.** It is Apple-only
   (MLX, ~700MB per reranker.py), its marginal benefit was never demonstrated on the
   fixture (no committed reranker-off baseline to compare), and it re-does precision
   the calling agent does better with full task context. Keep it as an opt-in for the
   human CLI path only; this also makes the agent path portable (e.g. a Linux box).

## Search quality and features

### Cross-encoder performance optimization
The MLX reranker (`Qwen3-Reranker-0.6B`) is still the main warm-query
latency cost on realistic vault chunks. Passage text is now capped before
reranking; remaining possible reductions:

- Hardware-specific batching or sequence-classification reranker probes.
  Gate on real blind-test latency because MLX batch throughput depends on
  prompt length and padding.
- Better query routing so only ambiguous queries pay the full rerank budget.

### Additional CLI subcommands
Helpers exist inside `seeklink/app.py` but are not exposed on the CLI:

- `seeklink suggest-links <path>` — find notes that should be linked.
- `seeklink resolve-suggestion <id> approve|reject` — accept / reject a
  suggestion, writing `[[link]]` into the source on approval.
- `seeklink graph <path> --depth N` — show the link neighborhood of a note.

### Embedder upgrade path
Current default: `jinaai/jina-embeddings-v2-base-zh` (~330 MB, 768-dim).
Stronger multilingual alternatives (`Qwen3-Embedding-0.6B`, `BGE-M3`,
`jina-v3`) are >2 GB in ONNX form and not currently fastembed-supported.
Revisit when a <500 MB multilingual embedder with better CJK scores
becomes available in `fastembed`. `SEEKLINK_EMBEDDER_MODEL` is already
configurable; a swap requires a full re-index.

### Ingesting non-markdown sources
A vault `sources/` folder could hold raw external content (PDFs, papers,
textbooks) for semantic search. Requires a PDF→markdown pipeline
(`markitdown` / `marker` / `docling`), optional LLM metadata extraction,
and probably chapter-level chunking instead of the current fixed-size
chunks.

## Daemon and platform

### Daemon freshness integration
Cold-start `seeklink search` / `seeklink status` emit stderr warnings when
indexed files have drifted on disk. The daemon path does not propagate
those warnings back to clients. Adding a `warnings` field to the daemon
JSON response would let `cli_client` surface them in the same shape.

### Multi-vault daemon support
The daemon binds to a single socket (`~/.rhizome/seeklink.sock`) regardless
of vault. For multiple vaults to run concurrent daemons, hash the vault
path into the socket name. Deferred until multi-vault is a real user need.

### Linux reranker
The MLX reranker is Apple Silicon only. On Linux it self-disables. Options
to restore reranking on Linux:

- Port the scoring loop to `onnxruntime` (CUDA or CPU).
- Run a GGUF build of the same model via `llama.cpp` (e.g.
  `Mungert/Qwen3-Reranker-0.6B-GGUF`).

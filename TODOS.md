# Deferred work

Known limitations and possible future work. Not commitments — items here
ship if and when they become worth the cost.

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

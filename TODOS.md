# TODOs

Deferred work for future releases.

## v0.3 candidates

### Cross-encoder performance optimization
The MLX reranker (Qwen3-Reranker-0.6B) runs at ~60ms per pair on M3 Air Metal GPU, totaling ~1.2-2.7s for 20 candidates with realistic vault chunks. Potential paths to reduce this:
- Batch inference (process all 20 pairs in one forward pass instead of sequentially)
- Passage truncation (cap at 200 tokens for reranking, use full text only for final display)
- Configurable `rerank_k` via CLI flag (reduce candidate pool when speed matters)

### `suggest_links` and `graph` CLI subcommands
The helper functions exist in `seeklink/app.py` (moved from the deleted MCP server) but have no CLI exposure. Wire them as:
- `seeklink suggest-links <path>` — find notes that should be linked
- `seeklink resolve-suggestion <id> approve|reject` — approve writes `[[link]]`
- `seeklink graph <path> --depth N` — show link neighborhood

### Embedder upgrade path
Current: jina-embeddings-v2-base-zh (330MB, 768-dim). All 2026 multilingual alternatives (Qwen3-Embedding-0.6B, BGE-M3, jina-v3) are >2GB in ONNX form and not fastembed-supported. Revisit when a <500MB multilingual embedder with better CJK scores becomes available in fastembed. The `SEEKLINK_EMBEDDER_MODEL` env var is ready for a drop-in swap; a full re-index (`seeklink index --force`) is required after switching.

### Sources folder as RAG data source
The vault's `sources/` folder could store raw external content (textbooks, papers, PDFs converted to markdown) for semantic search. Requires:
- An ingest pipeline (MarkItDown / marker / docling for PDF→markdown)
- LLM metadata extraction pass (title, aliases, tags, chapter structure)
- Possibly chunk-by-chapter instead of chunk-by-400-tokens

### Daemon freshness integration
Currently freshness warnings only appear in the cold-start CLI path (`seeklink search/status`). The daemon doesn't propagate warnings back to clients. Add a `warnings` field to daemon JSON responses so `cli_client` can print them.

### Daemon auto-respawn on config mismatch
`cli_client.call()` refuses to reuse a daemon bound to a different vault or started with a different embedder/reranker (P1 correctness fix), but falls back to cold-start on every subsequent CLI call after a switch until the user manually kills the stale daemon. Codex rated this P2: add a `shutdown` command to the daemon protocol, have the client shutdown + respawn on mismatch so the auto-spawn workflow keeps working across vault/model switches.

## Infrastructure

### PyPI v0.2 publishing
Publish v0.2.0 to PyPI. v0.1.0 is already there (manually published).
When setting this up, prefer a [Trusted Publisher (OIDC)](https://docs.pypi.org/trusted-publishers/) flow via a SHA-pinned `pypa/gh-action-pypi-publish` action instead of a long-lived API token. One-time config; removes the "what if the token leaks" supply-chain class entirely.

### Multi-vault daemon support
Current daemon binds to a single socket (`~/.rhizome/seeklink.sock`) regardless of vault. If multiple vaults need concurrent daemons, hash the vault path into the socket name. Deferred until multi-vault is a real use case.

### Linux deployment
The MLX reranker is Apple Silicon only. On Linux, either:
- Disable reranker (`SEEKLINK_RERANKER_MODEL=""`) and rely on RRF only
- Port to onnxruntime with CUDA (if GPU available)
- Port to llama.cpp via GGUF format (`Mungert/Qwen3-Reranker-0.6B-GGUF`)

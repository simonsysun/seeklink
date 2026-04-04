# TODOs

Deferred work for future releases. Contributions welcome.

## v0.2 candidates

### Cross-encoder reranking
Re-rank top results with a cross-encoder for better precision. Needs latency benchmarking to ensure the quality gain justifies the added latency (~100-300ms per query).

**Context:** The current RRF fusion (BM25 + vector + indegree + title) is already strong. Cross-encoder would be an optional second pass on the top-k results. Competitor "Hybrid Search" uses multilingual-e5 reranking.

### Lightweight embedding model option
Add `SEEKLINK_MODEL` environment variable to choose between:
- `jinaai/jina-embeddings-v2-base-zh` (default, 330MB, best CJK)
- A smaller multilingual model (~117MB, 100+ languages)

**Context:** Deferred because adding model choice increases decision burden for new users. 330MB is acceptable in 2026. Add when community requests it.

### Agent knowledge layer
Extend SeekLink as a knowledge/memory layer for AI agents and multi-agent systems. Current API is note-centric (paths, frontmatter, wikilinks). Agent memory would need: programmatic note creation tool, entity/fact-level retrieval, session memory, and possibly a different metadata model. The search + graph infrastructure is already there.

## Infrastructure

### PyPI automated publishing
GitHub Actions workflow to publish to PyPI on tag push. Currently v0.1.0 is published manually.

### Integration test fixture cleanup
Session-scoped async MCP client fixture hangs during teardown when all integration tests run together. Individual test classes pass fine. Likely a `create_connected_server_and_client_session` cleanup issue with the watcher task.

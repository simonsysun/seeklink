---
tags: [ml, rag, retrieval]
aliases: [RAG, retrieval-augmented]
---
# Retrieval-augmented generation

RAG is the pattern of augmenting an LLM's prompt with retrieved documents
at query time, so the model can answer from up-to-date or private
information it wasn't trained on. Pipeline:

1. **Index**: chunk your corpus, embed each chunk, store in a vector DB.
2. **Retrieve**: at query time, embed the user's question, find the
   top-K similar chunks.
3. **Augment**: inject those chunks into the prompt alongside the question.
4. **Generate**: LLM answers using the provided context.

## Why this beats fine-tuning for most use cases

- **Fresh data**: re-indexing is cheap; re-training isn't.
- **Attribution**: you can point to exactly which chunks the answer
  came from.
- **Access control**: different users can see different subsets of the
  index.
- **Smaller models work**: the model doesn't need to memorize the
  corpus; it just needs to read + synthesize.

## The hybrid retrieval insight

Pure vector retrieval misses exact keyword matches. Pure keyword
([[BM25]]) misses semantic matches. **Hybrid** combines both channels
via a fusion algorithm, typically [[RRF]] (Reciprocal Rank Fusion).
Many RAG systems now also add a **reranker** as a third stage that
re-scores the top-K candidates using a cross-encoder for precision
([[attention-mechanism]] discusses why this works).

seeklink is a hybrid retrieval tool — it doesn't include the "generate"
step; it's designed to be shelled out to by an LLM agent or embedded in
a larger pipeline.

## Failure modes

- **Chunking too coarse**: 2000-token chunks retrieve but the model has
  to find the answer inside a wall of text.
- **Chunking too fine**: 100-token chunks lose context.
- **No deduplication**: three chunks of the same document crowd out
  others.
- **Query mismatch**: user asks in casual English, notes written in
  formal Chinese — vector helps, but query expansion helps more.

## Related
- [[BM25]] — keyword retrieval channel
- [[RRF]] — fusion
- [[vector-embeddings]] — semantic channel
- [[attention-mechanism]] — reranker internals
- [[agent-memory-patterns]] — RAG as memory substrate

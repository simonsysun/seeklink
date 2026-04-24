---
tags: [ml, retrieval, embeddings]
aliases: [embeddings, 向量嵌入, dense-retrieval]
---
# Vector embeddings

An embedding is a fixed-length numeric vector that represents a piece
of text (or image, or code) in a geometric space where **semantic
similarity maps to geometric proximity**. Two paragraphs about the same
topic end up close in the space; unrelated paragraphs end up far.

Modern text embedders (jina-embeddings-v2-base-zh, BGE-M3, Qwen3-Embedding)
are 500-1500 MB Transformer models that output 512-1024 dimensional
vectors. You feed in tokenized text, take the `[CLS]` token or mean-pool
the last hidden states, L2-normalize, and that's your embedding.

## How similarity is computed

Cosine similarity:

```
sim(a, b) = (a · b) / (|a| * |b|)
```

Range: -1 to 1. In practice for normalized embeddings, this simplifies
to the dot product. Values typically cluster between 0.5-0.9 for related
text.

## Bi-encoder vs cross-encoder

- **Bi-encoder**: embed query and document separately, compare via dot
  product. Fast (~ms per query over millions of docs via approximate
  nearest neighbor).
- **Cross-encoder**: feed `(query, document)` pair into a model that
  outputs a single relevance score. Accurate but 100-1000× slower;
  used as a reranker on top of bi-encoder candidates (see
  [[attention-mechanism]]).

seeklink uses a bi-encoder (`jina-embeddings-v2-base-zh`) for first-stage
retrieval and a cross-encoder (`Qwen3-Reranker-0.6B`) for reranking.

## CJK-specific concerns

Many multilingual embedders underperform on Chinese because training
data skews English. CJK-specialized models (jina-v2-zh, bge-large-zh,
Qwen3) fix this with balanced training data. When the vault is CJK-first,
the choice of embedder matters more than any other single design
decision in the retrieval stack.

## Re-indexing cost

Every vault document must be re-embedded when you change the embedder,
because embeddings from different models live in incompatible vector
spaces. This is why seeklink doesn't swap embedders lightly — a swap
means a full re-index.

## Related
- [[BM25]] — sparse retrieval counterpart
- [[attention-mechanism]] — what the embedder is doing internally
- [[retrieval-augmented-generation]] — end-to-end system

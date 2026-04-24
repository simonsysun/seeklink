---
tags: [ml, transformers, architecture]
aliases: [self-attention, scaled-dot-product-attention]
---
# Attention mechanism

The operation at the core of Transformer architectures. Given a sequence
of tokens, attention lets every position look at every other position
with learnable weights. For the output at position `i`, you mix all the
values at other positions, weighted by a similarity between position
`i`'s query and each position's key.

Scaled dot-product form:

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

where `Q`, `K`, `V` are linear projections of the input (queries, keys,
values), and `d_k` is the key dimension (scaling prevents softmax
saturation at large dims).

## Why it works

- **Parallel**: unlike RNNs, all positions compute simultaneously.
- **Long range**: the softmax over all positions means a token 2000
  away can still influence the output if its key matches the query.
- **Learned routing**: the model learns which positions to attend to
  for which queries. Different attention heads learn different patterns
  (syntactic, semantic, positional).

## Multi-head attention

Run `h` parallel attentions with different learned projections, then
concatenate. Each head can specialize: one might track subject-verb
agreement, another might track anaphora, etc. Interpretability
research has extracted some of these patterns from trained models.

## The quadratic cost

The softmax is over the full sequence length `n`, giving `O(n^2)` time
and memory. For long contexts, this dominates training cost. Recent
work (FlashAttention, Mamba-style state-space models) tries to avoid
the quadratic wall.

## Relevance to retrieval

A **cross-encoder reranker** (like Qwen3-Reranker-0.6B) uses attention
across the concatenated `(query, passage)` pair — that's what makes it
more accurate than a bi-encoder (which encodes `query` and `passage`
independently and compares via dot product). The extra accuracy comes
from the learned attention between query tokens and passage tokens.

See [[vector-embeddings]] for the bi-encoder side, and
[[retrieval-augmented-generation]] for the end-to-end pipeline.

## Related
- [[vector-embeddings]] — embeddings are built from attention outputs
- [[retrieval-augmented-generation]] — where reranking happens

---
tags: [ir, retrieval, bm25]
aliases: [BM25, Okapi-BM25]
---
# BM25 intuition

BM25 is the classic keyword-matching ranking function from 1990s IR
research. It scores a (query, document) pair based on how often the
query's terms appear in the document, with two key refinements:

1. **Term frequency saturation** — the first occurrence of a term
   matters a lot, the tenth not much more. BM25 uses:
   ```
   tf_scaled = (tf * (k1 + 1)) / (tf + k1)
   ```
   This plateaus, unlike plain TF.

2. **Length normalization** — long documents naturally contain more of
   every term. BM25 divides by a length factor so short docs aren't
   unfairly penalized.

3. **Inverse document frequency** — rare query terms score higher than
   common ones. "the" gets almost zero weight; "spaced" gets a lot.

Full formula:

```
BM25(q, d) = Σ IDF(q_i) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
```

where `k1 ≈ 1.2-2.0` and `b ≈ 0.75`.

## Why it still matters in 2026

Vector embedders are strong at semantic match, but they lose on exact
vocabulary. "FSRS" is a proper noun that means one thing; if the
embedder maps it near general "spaced repetition" notes, it will drown
the exact hit. BM25 will rank the document that literally contains
"FSRS" at the top regardless of semantics.

This is why production RAG systems use **hybrid** retrieval: BM25 +
vector, fused by [[RRF]]. See [[retrieval-augmented-generation]].

## BM25 on CJK

Stock BM25 expects word boundaries (whitespace). Chinese has no
whitespace, so naive BM25 indexes character n-grams and performs
terribly. The fix: a CJK-aware tokenizer like **jieba** that splits on
semantic word boundaries. seeklink wires jieba as an FTS5 custom
tokenizer.

## Related
- [[RRF]] — how BM25 and vector channels are combined
- [[sqlite-fts5-cjk]] — seeklink's implementation
- [[vector-embeddings]] — semantic counterpart

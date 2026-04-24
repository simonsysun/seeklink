---
tags: [card, ir]
---
# Card: RRF formula

**Q**: How does Reciprocal Rank Fusion combine rankings from multiple
retrieval channels?

**A**:
```
score(c) = Σ_i w_i / (k + rank_i(c))
```

For each candidate `c`, sum (per-channel weight) ÷ (k + rank) across
all channels. `k ≈ 60` is a smoothing constant. Scale-free — no need
to normalize raw scores across BM25/vector/etc.

See [[RRF]] for context.

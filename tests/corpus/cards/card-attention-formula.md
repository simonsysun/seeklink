---
tags: [card, ml]
---
# Card: Attention formula

**Q**: Write the scaled dot-product attention formula.

**A**:
```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

Query `Q`, key `K`, value `V` are linear projections of the input.
Scaling by `sqrt(d_k)` prevents softmax saturation at high dimension.

See [[attention-mechanism]] for why each piece is there.

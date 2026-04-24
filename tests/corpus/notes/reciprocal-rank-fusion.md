---
tags: [ir, fusion, retrieval]
aliases: [RRF, reciprocal-rank-fusion]
---
# Reciprocal Rank Fusion (RRF)

A simple algorithm for combining ranked lists from multiple retrieval
systems into a single ranked list. Given `N` channels, each producing
its own ranking of the same candidates, RRF assigns each candidate:

```
score(c) = Σ_i w_i / (k + rank_i(c))
```

where `rank_i(c)` is `c`'s 1-indexed rank in channel `i` (or ∞ if
absent), `w_i` is a per-channel weight, and `k` is a smoothing constant
(typically 60).

## Why RRF works

- **Scale-free**: BM25 returns raw scores in [0, 30]; vector similarity
  returns [0, 1]. Can't average these directly. RRF only uses ranks,
  which are always 1, 2, 3...
- **Rank-sensitive**: a rank-1 hit contributes 1/61 ≈ 0.0164; rank-20
  contributes 1/80 ≈ 0.0125. Top hits matter more but not dominantly.
- **Tolerant of bad channels**: if one channel produces garbage, its
  contribution is small compared to consensus among good channels.

See [[bm25-intuition]] and [[vector-embeddings]] for the two main
channels in hybrid retrieval.

## Variant: weighted RRF

With channel weights, `w_i = 1.5` for "title" channel (match on title
should count more) or `w_i = 0.3` for "indegree" channel (well-linked
notes are a weak quality signal). seeklink uses:

```
bm25:    1.0
vector:  1.0
indegree: 0.3
title:   1.5
```

The title weight 1.5 is deliberate — high enough to amplify exact
alias hits, low enough that untitled log entries still compete. See
seeklink README's "why title weight is 1.5".

## RRF isn't the only fusion

- **CombSUM**: sum of (normalized) scores across channels. Requires
  score normalization, which is fragile.
- **CombMNZ**: CombSUM multiplied by the number of channels the
  candidate appeared in. Favors consensus, penalizes rare channels.
- **Learning-to-rank (LTR)**: use a model (gradient boosted trees or
  neural) to learn fusion weights from labeled data. Best accuracy,
  but requires labels.

For personal vaults without labeled data, RRF is the pragmatic choice.

## Related
- [[bm25-intuition]]
- [[vector-embeddings]]
- [[retrieval-augmented-generation]]

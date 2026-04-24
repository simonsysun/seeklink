---
tags: [learning-science, spaced-repetition, algorithm]
aliases: [FSRS, free-spaced-repetition-scheduler]
---
# FSRS algorithm

FSRS (Free Spaced Repetition Scheduler) is a memory model that replaces
Anki's SM-2 as the default scheduling algorithm in modern flashcard apps.
It predicts how long you'll remember a card based on three latent
variables: **difficulty** (how hard the card is for this specific
reviewer), **stability** (how long the memory will last before it decays
below retrievability threshold), and **retrievability** (the current
probability of recall).

The model is trained on millions of real review logs and outputs, for each
card, the optimal next interval that will land your recall probability at
a target (typically 90%).

## Why FSRS beats SM-2

SM-2 uses a fixed formula tied to a user-reported "ease factor" per card.
It doesn't learn from population data, and its intervals tend to grow
too fast for hard cards and too slow for easy ones. FSRS models the
**forgetting curve** explicitly (see [[forgetting-curve]]) and fits the
curve's parameters from data.

On the same review history, FSRS achieves ~30% lower workload for the
same retention target, or equivalently ~10% better retention for the same
workload.

## Core equations (simplified)

Stability update after review with rating `g ∈ {1, 2, 3, 4}`:

```
S_new = S * (1 + exp(w_8) * (11 - D) * S^(-w_9) * (exp((1-R) * w_10) - 1))
```

where `D` is difficulty, `R` is current retrievability computed from
elapsed time, and `w_0..w_16` are the 17 free parameters fitted from
training data.

The practitioner doesn't need to care about the equation — what matters
is that FSRS shortens intervals for cards you nearly forgot and lengthens
them for cards you remember effortlessly.

## Related
- [[spaced-repetition]] — the practice that FSRS optimizes
- [[forgetting-curve]] — the phenomenon being modeled
- [[note-to-card-workflow]] — how permanent notes become reviewable cards

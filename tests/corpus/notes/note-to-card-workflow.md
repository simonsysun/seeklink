---
tags: [pkm, spaced-repetition, workflow]
aliases: [笔记转卡片, zettel-to-anki]
---
# Note-to-card workflow

The bridge between [[zettelkasten]] (where you *think*) and
[[spaced-repetition]] (where you *retain*). Permanent notes are too
long to review daily. Flashcards are too atomic to think with. The
workflow distills one into the other.

## The distillation step

For each permanent note worth retaining:

1. Identify the **claim** the note argues for. One sentence.
2. Write it as a question whose answer is that claim. Front side.
3. Write the answer — the claim itself, plus the one reasoning step
   that makes it stick. Back side.
4. Tag the card with the note's aliases so you can trace back.

Cards are disposable. Notes are durable. If the card needs an update,
you change the note first, then regenerate the card. The note is
the source of truth.

## Example

**Note** ([[FSRS]]): argues FSRS beats SM-2 because it models
population-fitted stability instead of per-card ease.

**Card front**: "Why does FSRS outperform SM-2 at the same retention
target?"

**Card back**: "FSRS fits stability parameters from population review
data, adjusting intervals to the *actual* forgetting curve shape
instead of SM-2's fixed formula + user-reported ease factor. ~30%
lower workload at equal retention."

## Common failure: over-carding

Don't make a flashcard for every sentence of your notes. The goal is
**review-worthy claims**, not "does this look like a fact". Three
cards per permanent note is plenty; one strong card beats five weak
ones.

See [[feynman-technique]] — if you can't formulate the card-back as a
plain-language answer, the note isn't ready yet.

## Related
- [[FSRS]] — scheduler that processes the cards
- [[atomic-notes]] — what you're distilling from
- [[spaced-repetition]] — why distillation is worth doing

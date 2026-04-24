---
tags: [learning-science, memory]
aliases: [SRS, 间隔重复]
---
# Spaced repetition

Spaced repetition is the practice of reviewing material at increasing
intervals, timed to catch you right before you'd forget. It's the single
most evidence-backed intervention in memory research: it works across
age groups, subject matter, and cultures.

The underlying phenomenon is the **spacing effect**: two 10-minute
review sessions separated by a day produce dramatically better long-term
retention than one 20-minute massed session, for the same total time.

## How it works in practice

Apps like Anki, SuperMemo, and Mochi implement spaced repetition via
flashcards. You create a card (question front, answer back), and the app
schedules it. You rate each review: "again" (failed), "hard" (just
barely), "good" (normal), or "easy" (trivial). The rating feeds back into
the schedule. Harder cards come back sooner; easy cards stretch out.

See [[FSRS]] for the modern scheduling algorithm, or [[forgetting-curve]]
for the empirical basis.

## The 记忆保持力 problem

The core question of spaced repetition is: **how do we maximize memory
retention per hour of study?** Daily re-reading wastes most of the time
on material you already know. One-pass study loses almost everything
within a month. Well-spaced review lands somewhere much better — typical
numbers are 80-90% retention at 10% of the time cost of re-reading.

## Why it's under-used

Spaced repetition is boring. It demands honest self-assessment ("did I
really remember, or did I recognize the answer?"), it punishes you for
missed days, and it surfaces your weakest material exactly when you're
tired. Most people bounce off after a week. The tooling — cards,
schedulers, syncing — adds friction.

The meta-skill is [[note-to-card-workflow]]: learning to distill reading
into atomic questions worth reviewing.

## Related
- [[FSRS]] — state-of-the-art scheduler
- [[forgetting-curve]] — what the spacing fights against
- [[feynman-technique]] — complementary practice for deep understanding
- [[zettelkasten]] — a note-taking method that pairs well with SRS

---
source: synthesized intro article
access_date: 2026-04-15
tags: [source, anki, tools, imported]
---
# Anki and flashcards — a practical intro

*A concise synthesized primer, 2026.*

Anki is the most widely used open-source flashcard application. Core
loop: you create cards (front / back), Anki schedules them based on
your self-rating of each review (Again / Hard / Good / Easy), and you
see each card at intervals designed to catch you right before you'd
forget.

## How to make a good card

1. **One claim per card.** If the back side has "AND", split it.
2. **Question, not prompt.** "Explain how attention works" is a
   prompt; "Why does multi-head attention outperform single-head?"
   is a question — it admits exactly one correct answer.
3. **Short answer.** 1-3 sentences. A three-paragraph back side isn't
   a flashcard, it's a note you're pretending is a flashcard.
4. **Source reference.** Include a link to where you learned this,
   either as a tag or a trailing `src:` line. You'll need it.

## Common mistakes

- **Cloze overuse.** Cloze cards (`The capital of France is {{c1::Paris}}`)
  are quick to make but often test recognition, not recall.
- **Same-card images.** If you're memorizing a diagram by always
  seeing the same diagram, you're memorizing pixels, not content.
- **Reviewing while distracted.** The schedule assumes honest
  self-assessment. If you half-know an answer but rate "good" because
  you're on a commute, you're poisoning the schedule.

## Algorithms

Anki's default scheduler has been **SM-2** (derived from SuperMemo's
1985 algorithm) for most of its history. Since 2024, **FSRS** is
shipped as an optional (now default on 2.1.66+) replacement. FSRS
models the forgetting curve empirically and typically achieves the
same retention target with 20-30% fewer reviews.

Most users will see a noticeable improvement the first week after
switching to FSRS, as cards they kept hitting repeatedly get longer
intervals and cards they were over-confident on get shorter ones.

## Ecosystem

- **AnkiWeb** (free sync)
- **AnkiMobile** (iOS, paid, funds development)
- **AnkiDroid** (Android, free)
- Shared decks: anki.space, large community of shared decks for
  medicine, language, law, etc.
- Add-ons: heatmap, image occlusion, hierarchical tags, FSRS helper.

## When not to use Anki

- If you haven't understood the material yet. Anki entrenches; it
  doesn't teach. Understand first (Feynman), then memorize.
- If the material has a 2-week shelf life. Use a note for that, not a
  card.
- If you resent reviewing. Cards you dread become noise; you'll
  either abandon Anki or fail the cards reflexively. Prune ruthlessly.

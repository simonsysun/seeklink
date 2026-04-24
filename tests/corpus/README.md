# seeklink test corpus

A self-contained fixture vault for the blind-test framework (see
`docs/blind-test.md`). Not the user's personal vault — everything here is
hand-authored or synthesized for testing purposes, so the corpus can be
committed, re-indexed, and compared across seeklink versions without
touching real notes.

## Layout

| Folder | Purpose | Style |
|---|---|---|
| `notes/` | Permanent notes | Strong frontmatter (tags, aliases), clear titles, wikilinks between notes, 300-800 words |
| `logs/` | Weekly/daily logs | No frontmatter title, first-person stream, references notes via wikilinks |
| `sources/` | Simulated external material | Longer, article-style, single `source:` field in frontmatter |
| `recipes/` | Off-topic content (cooking) | Tests that domain-unrelated queries don't leak results |
| `cards/` | Atomic Q/A flashcard notes | Very short, single concept each |

## Languages

Intentionally bilingual. ~40% English-dominant, ~30% Chinese-dominant,
~30% mixed. This exercises the jieba tokenizer (BM25 channel), the
jina-v2-zh embedder (vector channel), and cross-language recall.

## Topics

Three semantic clusters:

1. **Learning science / PKM** — FSRS, spaced repetition, Zettelkasten,
   Feynman technique, forgetting curve. Heavily cross-linked.
2. **ML / IR / agents** — attention, embeddings, RAG, BM25, RRF. Also
   cross-linked but weaker overlap with cluster 1.
3. **Off-topic** — recipes, to confirm nothing from these leaks into
   on-topic query results.

## Regenerating the index

From the seeklink project root:

```bash
SEEKLINK_VAULT=tests/corpus python -m seeklink index
```

Or via the blind-test runner (see `tests/blind/run.py`), which cold-starts
against a specified `--vault`.

## Licensing

Everything in this folder is original or paraphrased for testing purposes.
Files in `sources/` that cite Wikipedia are summaries, not verbatim copies.

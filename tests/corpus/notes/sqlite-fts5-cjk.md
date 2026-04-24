---
tags: [technical, sqlite, fts5, cjk]
aliases: [fts5-jieba, sqlite-fts-chinese]
---
# SQLite FTS5 + CJK (jieba)

SQLite's FTS5 extension provides full-text search out of the box, but
its default tokenizers (`simple`, `porter`, `unicode61`) all assume
whitespace or Unicode word boundaries. Chinese has neither — a sentence
like `语义搜索的研究` is one unbroken string to `unicode61`, which
then indexes it as a single 7-character token. Searches for `语义` or
`搜索` fail to match.

## The fix: jieba as custom tokenizer

[jieba](https://github.com/fxsjy/jieba) is the standard Chinese word
segmenter. It uses a dictionary + HMM to split `语义搜索的研究` into
`[语义, 搜索, 的, 研究]`. Plugging jieba into FTS5 as a custom
tokenizer makes BM25 work on Chinese.

seeklink uses the `sqlitefts` Python package to register jieba.
Registration happens once at connection open; after that, `CREATE
VIRTUAL TABLE ... USING fts5(content, tokenize='jieba')` just works.

## Gotchas

- **Tokenizer must be registered in every connection** that reads the
  FTS5 index. Forgetting this gives silent "0 results" on Chinese
  queries.
- **Mixed content**: jieba tokenizes Chinese correctly but also works
  on English by falling back to whitespace. Good enough for bilingual
  vaults.
- **Index size**: jieba tokens are longer than character n-grams, so
  the FTS5 index is smaller — usually ~1.5× the raw text size vs 5×
  for character-level.
- **Updates**: jieba's dictionary is fixed per install. If your vault
  has heavy jargon (character names, domain terms), you can load a
  custom dict into jieba before indexing.

## Related
- [[bm25-intuition]] — BM25 is what runs on this index
- [[RRF]] — how the BM25 channel gets combined with vector

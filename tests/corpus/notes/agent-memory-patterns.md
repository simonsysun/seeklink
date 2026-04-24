---
tags: [agents, memory, architecture]
aliases: [agent-memory, LLM-memory-systems]
---
# Agent memory patterns

LLMs are stateless between calls. Any "memory" an agent exhibits comes
from something external that persists across invocations. A few patterns
in current practice:

## 1. Conversation window

The simplest: keep the last N turns in the context. Works for chat.
Fails for anything long-lived because the window truncates old facts.

## 2. Summary + recent

Compress old turns into a running summary; prepend summary + keep recent
full turns. Loses detail but bounded context. Most "long memory"
chatbots do this.

## 3. Retrieval-based (RAG)

Store turns, notes, and tool outputs as documents; index them with
[[vector-embeddings]] + [[BM25]]; retrieve relevant ones at each turn.
See [[retrieval-augmented-generation]]. This scales but depends heavily
on retrieval quality — a bad index means the agent "forgets" things
that are technically stored.

## 4. File-based scratchpad

Agent writes to and reads from files as needed. Simpler than vector DB;
works well when the agent knows file paths. Claude Code's CLAUDE.md
files and the memory file system are this pattern.

## 5. Knowledge graph

Parse facts out of interactions; store as `(subject, relation, object)`
triples; query by relation pattern. Much more work to build, harder
than RAG for most use cases. A few specialized agents (medical, legal)
use this.

## Hybrid patterns in production

Most capable agents today combine several:
- Conversation summary for recent context
- RAG over long-term document store
- File-based scratchpad for working memory
- Sometimes a separate "facts about the user" k/v store

## Search quality dominates

In all RAG-based memory systems, **the agent is only as smart as its
search layer**. If seeklink returns the wrong three chunks, the agent
answers wrong with confidence. Reranking ([[attention-mechanism]]),
[[RRF]] fusion, and well-calibrated channel weights matter much more
than the LLM size.

## Related
- [[retrieval-augmented-generation]]
- [[vector-embeddings]]
- [[RRF]]

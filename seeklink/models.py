"""Frozen dataclasses for all SeekLink entity types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    id: int
    uid: str
    path: str
    title: str | None
    content_hash: str | None
    status: str
    indegree: int
    fs_modified_at: str | None
    indexed_at: str | None
    created_at: str
    updated_at: str
    aliases: str = "[]"  # JSON array of alias strings


@dataclass(frozen=True, slots=True)
class Chunk:
    id: int
    source_id: int
    content: str
    chunk_index: int
    char_start: int | None
    char_end: int | None
    token_count: int | None
    created_at: str


@dataclass(frozen=True, slots=True)
class WikiLink:
    id: int
    source_note_id: int
    target_note_id: int | None
    target_path: str
    created_at: str


@dataclass(frozen=True, slots=True)
class Suggestion:
    id: int
    source_note_id: int
    target_note_id: int
    score: float
    reason: str | None
    status: str
    created_at: str
    resolved_at: str | None



"""Markdown chunker — splits text into semantically coherent chunks.

Invariant: chunk.text == source_text[chunk.char_start:chunk.char_end]
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCED_CODE = re.compile(r"(?m)^(`{3,})[^\n]*\n(.*?)^\1\s*$", re.DOTALL)
_HEADER = re.compile(r"(?m)^#{1,6}\s")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    text: str
    char_start: int
    char_end: int
    token_count: int


def estimate_tokens(text: str) -> int:
    """Estimate token count using a CJK-aware heuristic.

    ~len/4 for Latin, ~len/1.5 for CJK, blended for mixed.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff")
    non_cjk = len(text) - cjk
    return max(1, int(non_cjk / 4 + cjk / 1.5))


def chunk_markdown(text: str, target_tokens: int = 400) -> list[ChunkSpan]:
    """Split markdown into chunks respecting semantic boundaries.

    Splitting hierarchy:
    1. Fenced code blocks → atomic chunks (never split)
    2. Headers → section boundaries
    3. Paragraph boundaries (blank lines)
    4. Sentence boundaries
    5. Single sentences kept whole even if over target

    Invariant: chunk.text == text[chunk.char_start:chunk.char_end]
    """
    if not text or not text.strip():
        return []

    # Step 1: Extract fenced code blocks as atomic segments
    segments: list[tuple[int, int, bool]] = []  # (start, end, is_code)
    last_end = 0
    for m in _FENCED_CODE.finditer(text):
        if m.start() > last_end:
            segments.append((last_end, m.start(), False))
        segments.append((m.start(), m.end(), True))
        last_end = m.end()
    if last_end < len(text):
        segments.append((last_end, len(text), False))

    chunks: list[ChunkSpan] = []

    for seg_start, seg_end, is_code in segments:
        if is_code:
            seg_text = text[seg_start:seg_end]
            if seg_text.strip():
                chunks.append(ChunkSpan(
                    text=seg_text,
                    char_start=seg_start,
                    char_end=seg_end,
                    token_count=estimate_tokens(seg_text),
                ))
            continue

        # Step 2: Split prose at headers
        header_spans = _split_at_headers(text, seg_start, seg_end)

        for h_start, h_end in header_spans:
            # Step 3: Split at paragraph boundaries
            para_spans = _split_paragraphs(text, h_start, h_end)

            # Step 4: Accumulate paragraphs until near target
            _accumulate(para_spans, target_tokens, chunks, text)

    return [c for c in chunks if c.text.strip()]


def _split_at_headers(
    text: str, start: int, end: int
) -> list[tuple[int, int]]:
    """Split region at header lines, returning (start, end) sub-spans."""
    region = text[start:end]
    split_points = [0]
    for m in _HEADER.finditer(region):
        if m.start() > 0:
            split_points.append(m.start())
    spans: list[tuple[int, int]] = []
    for i in range(len(split_points)):
        s = start + split_points[i]
        e = start + split_points[i + 1] if i + 1 < len(split_points) else end
        if text[s:e].strip():
            spans.append((s, e))
    return spans


def _split_paragraphs(
    text: str, start: int, end: int
) -> list[tuple[int, int]]:
    """Split region at blank lines, returning trimmed content (start, end) spans."""
    region = text[start:end]
    spans: list[tuple[int, int]] = []
    raw_parts = re.split(r"(\n\s*\n)", region)
    pos = 0
    for part in raw_parts:
        if part.strip():
            lstrip_count = len(part) - len(part.lstrip())
            rstrip_count = len(part) - len(part.rstrip())
            cs = start + pos + lstrip_count
            ce = start + pos + len(part) - rstrip_count
            if cs < ce:
                spans.append((cs, ce))
        pos += len(part)
    return spans


def _accumulate(
    para_spans: list[tuple[int, int]],
    target_tokens: int,
    out: list[ChunkSpan],
    text: str,
) -> None:
    """Accumulate paragraph spans into chunks near target_tokens.

    Chunk text is the exact original substring text[buf_start:buf_end].
    """
    buf_start: int | None = None
    buf_end: int = 0
    buf_tokens: int = 0

    def flush() -> None:
        nonlocal buf_start, buf_end, buf_tokens
        if buf_start is not None:
            chunk_text = text[buf_start:buf_end]
            out.append(ChunkSpan(
                text=chunk_text,
                char_start=buf_start,
                char_end=buf_end,
                token_count=estimate_tokens(chunk_text),
            ))
            buf_start = None
            buf_end = 0
            buf_tokens = 0

    for para_start, para_end in para_spans:
        para_text = text[para_start:para_end]
        para_tokens = estimate_tokens(para_text)

        # Single paragraph exceeds target → try splitting at sentences
        if para_tokens > target_tokens and buf_start is None:
            _split_sentences(para_start, para_end, target_tokens, out, text)
            continue

        # Would exceed target → flush first
        if buf_start is not None and buf_tokens + para_tokens > target_tokens:
            flush()

        if buf_start is None:
            buf_start = para_start
        buf_end = para_end
        buf_tokens += para_tokens

    flush()


def _split_sentences(
    start: int,
    end: int,
    target_tokens: int,
    out: list[ChunkSpan],
    text: str,
) -> None:
    """Split a large paragraph span at sentence boundaries.

    Chunk text is the exact original substring.
    """
    region = text[start:end]
    separators = list(_SENTENCE_END.finditer(region))

    # Build sentence spans: each ends right after punctuation (before separator whitespace)
    sentence_spans: list[tuple[int, int]] = []
    prev_end = 0
    for sep in separators:
        sentence_spans.append((start + prev_end, start + sep.start()))
        prev_end = sep.end()
    if prev_end < len(region):
        sentence_spans.append((start + prev_end, end))

    buf_start: int | None = None
    buf_end: int = start
    buf_tokens: int = 0

    def flush() -> None:
        nonlocal buf_start, buf_end, buf_tokens
        if buf_start is not None:
            chunk_text = text[buf_start:buf_end]
            out.append(ChunkSpan(
                text=chunk_text,
                char_start=buf_start,
                char_end=buf_end,
                token_count=estimate_tokens(chunk_text),
            ))
            buf_start = None
            buf_end = start
            buf_tokens = 0

    for s_start, s_end in sentence_spans:
        sent = text[s_start:s_end]
        s_tokens = estimate_tokens(sent)

        if buf_start is not None and buf_tokens + s_tokens > target_tokens:
            flush()

        if buf_start is None:
            buf_start = s_start
        buf_end = s_end
        buf_tokens += s_tokens

    flush()

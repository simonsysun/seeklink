"""Jieba-based FTS5 tokenizer for bilingual Chinese+English text."""

from __future__ import annotations

import re
import sqlite3
import unicodedata

import jieba
from sqlitefts import fts5


class JiebaTokenizer(fts5.FTS5Tokenizer):
    """FTS5 tokenizer using jieba for Chinese segmentation.

    - Chinese text: jieba.tokenize(mode='search') for compound-word support
    - Non-Chinese text: simple alphanumeric regex extraction
    - NFKC normalization for fullwidth character handling
    - Pre-computed byte offset array for O(1) offset lookup
    """

    _re_han = re.compile(
        r"(["
        r"\u3400-\u4dbf"  # CJK Extension A
        r"\u4e00-\u9fff"  # CJK Unified Ideographs
        r"\uf900-\ufaff"  # CJK Compatibility Ideographs
        r"]+)"
    )
    _re_alnum = re.compile(r"[a-zA-Z0-9]+")

    def tokenize(self, text: str, flags: int | None = None):
        text = unicodedata.normalize("NFKC", text)

        # Pre-compute character -> byte offset mapping
        byte_offsets: list[int] = []
        b = 0
        for ch in text:
            byte_offsets.append(b)
            b += len(ch.encode("utf-8"))
        byte_offsets.append(b)  # sentinel for end-of-string

        parts = self._re_han.split(text)
        char_offset = 0

        for part in parts:
            if not part:
                continue
            if self._re_han.match(part):
                # Chinese: use jieba search mode for compound-word segmentation
                for token, start, end in jieba.tokenize(part, mode="search"):
                    token = token.strip()
                    if not token:
                        continue
                    global_start = char_offset + start
                    global_end = char_offset + end
                    bs = byte_offsets[global_start]
                    be = byte_offsets[global_end]
                    yield token.lower(), bs, be
            else:
                # Non-Chinese: extract alphanumeric tokens
                for m in self._re_alnum.finditer(part):
                    token_text = m.group()
                    global_char_idx = char_offset + m.start()
                    bs = byte_offsets[global_char_idx]
                    be = byte_offsets[global_char_idx + len(token_text)]
                    yield token_text.lower(), bs, be

            char_offset += len(part)


def register_jieba_tokenizer(conn: sqlite3.Connection) -> None:
    """Register the jieba FTS5 tokenizer on the given connection."""
    tk = fts5.make_fts5_tokenizer(JiebaTokenizer())
    fts5.register_tokenizer(conn, "jieba", tk)

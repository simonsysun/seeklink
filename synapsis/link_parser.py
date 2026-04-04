"""Wiki-link parser — extracts [[target]] links from markdown text."""

from __future__ import annotations

import re

_FENCED_CODE = re.compile(r"(?m)^(`{3,})[^\n]*\n(.*?)^\1\s*$", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]+`")
_WIKI_LINK = re.compile(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]*?)?\]\]")


def extract_wiki_links(text: str) -> list[str]:
    """Extract wiki-link targets from markdown, preserving first-occurrence order.

    Handles [[target]] and [[target|alias]] syntax.
    Skips links inside fenced code blocks and inline code.
    Returns deduplicated list of target strings.
    """
    if not text:
        return []

    # Replace fenced code blocks with spaces (preserves positions)
    cleaned = text
    for m in _FENCED_CODE.finditer(cleaned):
        cleaned = cleaned[: m.start()] + " " * (m.end() - m.start()) + cleaned[m.end() :]

    # Replace inline code with spaces
    for m in _INLINE_CODE.finditer(cleaned):
        cleaned = cleaned[: m.start()] + " " * (m.end() - m.start()) + cleaned[m.end() :]

    # Extract targets, deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in _WIKI_LINK.finditer(cleaned):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            result.append(target)

    return result

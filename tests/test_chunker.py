"""Tests for seeklink.chunker — pure function tests, no DB or embedding."""

from __future__ import annotations

from seeklink.chunker import ChunkSpan, chunk_markdown, estimate_tokens


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_latin(self):
        text = "Hello world, this is a test sentence."
        tokens = estimate_tokens(text)
        assert tokens == len(text) // 4

    def test_chinese(self):
        text = "深度学习使用反向传播算法"
        tokens = estimate_tokens(text)
        # ~len/1.5 for CJK
        assert tokens > 0
        assert tokens == int(len(text) / 1.5)

    def test_mixed(self):
        text = "Hello 你好 world 世界"
        tokens = estimate_tokens(text)
        assert tokens > 0


class TestChunkMarkdown:
    def test_empty_input(self):
        assert chunk_markdown("") == []

    def test_whitespace_only(self):
        assert chunk_markdown("   \n\n  \n") == []

    def test_single_line(self):
        text = "Hello world."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_splits_on_headers(self):
        text = "# Header 1\nContent one.\n\n# Header 2\nContent two."
        chunks = chunk_markdown(text)
        assert len(chunks) == 2
        assert "Header 1" in chunks[0].text
        assert "Header 2" in chunks[1].text

    def test_splits_on_h2_h3(self):
        text = "## Section A\nText A.\n\n### Subsection B\nText B."
        chunks = chunk_markdown(text)
        assert len(chunks) == 2

    def test_paragraph_accumulation(self):
        """Short paragraphs should accumulate into one chunk."""
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_markdown(text)
        # All three are short, should be one chunk
        assert len(chunks) == 1
        assert "Para one." in chunks[0].text
        assert "Para three." in chunks[0].text

    def test_long_paragraph_splits_at_sentences(self):
        """A paragraph exceeding target_tokens should split at sentences."""
        # ~100 chars per sentence, target 20 tokens = ~80 chars
        s1 = "A" * 80 + ". "
        s2 = "B" * 80 + ". "
        s3 = "C" * 80 + "."
        text = s1 + s2 + s3
        chunks = chunk_markdown(text, target_tokens=20)
        assert len(chunks) >= 2

    def test_single_sentence_kept_whole(self):
        """A single long sentence should not be split mid-sentence."""
        text = "A" * 2000
        chunks = chunk_markdown(text, target_tokens=20)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_code_block_atomic(self):
        """Fenced code blocks are never split, even if large."""
        code = "```python\n" + "x = 1\n" * 200 + "```"
        text = "Before.\n\n" + code + "\n\nAfter."
        chunks = chunk_markdown(text, target_tokens=20)
        # Should have: before, code block, after
        code_chunks = [c for c in chunks if "x = 1" in c.text]
        assert len(code_chunks) == 1
        assert "```python" in code_chunks[0].text

    def test_chinese_text(self):
        text = "# 标题\n这是一段中文测试内容。\n\n## 第二节\n更多中文内容。"
        chunks = chunk_markdown(text)
        assert len(chunks) >= 1
        assert any("中文" in c.text for c in chunks)

    def test_mixed_zh_en(self):
        text = "# Mixed Content\nThis is English. 这是中文。\n\nMore text here."
        chunks = chunk_markdown(text)
        assert len(chunks) >= 1

    def test_char_offsets_correct(self):
        text = "# Header\nContent here."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.char_start >= 0
        assert c.char_end <= len(text)
        assert c.char_end > c.char_start

    def test_token_count_populated(self):
        text = "Some content for token counting."
        chunks = chunk_markdown(text)
        assert len(chunks) == 1
        assert chunks[0].token_count > 0

    def test_multiple_code_blocks(self):
        text = (
            "Text before.\n\n"
            "```\ncode block 1\n```\n\n"
            "Text between.\n\n"
            "```js\ncode block 2\n```\n\n"
            "Text after."
        )
        chunks = chunk_markdown(text)
        code_chunks = [c for c in chunks if "code block" in c.text]
        assert len(code_chunks) == 2

    def test_returns_chunk_span_type(self):
        chunks = chunk_markdown("Hello world.")
        assert all(isinstance(c, ChunkSpan) for c in chunks)

    def test_offset_invariant(self):
        """chunk.text == source_text[chunk.char_start:chunk.char_end] for all chunks."""
        texts = [
            "# Header\nContent here.",
            "Para one.\n\nPara two.\n\nPara three.",
            "# H1\nText.\n\n# H2\nMore text.",
            "  para1\n\n    para2\n",
            "```python\ncode\n```\n\nAfter code.",
            "# 标题\n中文内容。\n\n## 第二节\n更多。",
        ]
        for text in texts:
            for c in chunk_markdown(text):
                assert c.text == text[c.char_start : c.char_end], (
                    f"Offset invariant failed for text={text!r}: "
                    f"chunk.text={c.text!r} != slice={text[c.char_start:c.char_end]!r}"
                )

    def test_offset_invariant_with_sentences(self):
        """Offset invariant holds after sentence splitting."""
        s1 = "A" * 80 + ". "
        s2 = "B" * 80 + ". "
        s3 = "C" * 80 + "."
        text = s1 + s2 + s3
        for c in chunk_markdown(text, target_tokens=20):
            assert c.text == text[c.char_start : c.char_end]

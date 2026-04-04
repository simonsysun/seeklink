"""Tests for sophia.link_parser — pure function tests, no DB."""

from __future__ import annotations

from sophia.link_parser import extract_wiki_links


class TestExtractWikiLinks:
    def test_empty_text(self):
        assert extract_wiki_links("") == []

    def test_no_links(self):
        assert extract_wiki_links("Just regular text with no links.") == []

    def test_simple_link(self):
        assert extract_wiki_links("See [[note-name]] for details.") == ["note-name"]

    def test_link_with_spaces(self):
        assert extract_wiki_links("See [[my note]] here.") == ["my note"]

    def test_alias_link(self):
        """[[target|alias]] should extract only the target."""
        assert extract_wiki_links("See [[target|display text]] here.") == ["target"]

    def test_chinese_target(self):
        assert extract_wiki_links("参考[[知识管理]]和[[学习方法]]") == ["知识管理", "学习方法"]

    def test_multiple_links(self):
        text = "Links: [[alpha]], [[beta]], [[gamma]]."
        result = extract_wiki_links(text)
        assert result == ["alpha", "beta", "gamma"]

    def test_deduplication(self):
        text = "See [[note]] and also [[note]] again."
        assert extract_wiki_links(text) == ["note"]

    def test_preserves_first_occurrence_order(self):
        text = "First [[beta]], then [[alpha]], then [[beta]] again."
        assert extract_wiki_links(text) == ["beta", "alpha"]

    def test_skip_fenced_code_block(self):
        text = "Real [[link]] here.\n\n```\n[[fake-link]]\n```\n\nMore text."
        assert extract_wiki_links(text) == ["link"]

    def test_skip_inline_code(self):
        text = "Real [[link]] and `[[inline-code]]` ignored."
        assert extract_wiki_links(text) == ["link"]

    def test_empty_link_ignored(self):
        """[[]] should not produce a result."""
        assert extract_wiki_links("Empty [[]] link.") == []

    def test_nested_brackets_ignored(self):
        """Links with nested brackets should be handled gracefully."""
        text = "Normal [[valid-link]] text."
        assert extract_wiki_links(text) == ["valid-link"]

    def test_multiline_fenced_code(self):
        text = (
            "Before [[real]].\n\n"
            "```python\n"
            "# [[code-link]]\n"
            "x = [[another]]\n"
            "```\n\n"
            "After [[also-real]]."
        )
        assert extract_wiki_links(text) == ["real", "also-real"]

    def test_alias_with_chinese(self):
        text = "See [[知识管理|KM系统]] for more."
        assert extract_wiki_links(text) == ["知识管理"]

    def test_backtick_fence_variants(self):
        """Longer backtick fences should still be stripped."""
        text = "Real [[link]].\n\n````\n[[fake]]\n````\n"
        assert extract_wiki_links(text) == ["link"]

"""Tests for `seeklink get` command and the body-offset to file-line mapping helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from seeklink.search import body_offset_to_file_line


class TestBodyOffsetToFileLine:
    """Verify frontmatter-aware mapping from body char offset → file line."""

    def test_no_frontmatter_basic(self):
        text = "line 1\nline 2\nline 3\n"
        # body offset 0 → line 1
        assert body_offset_to_file_line(text, 0) == 1
        # body offset 7 (start of "line 2") → line 2
        assert body_offset_to_file_line(text, 7) == 2
        # body offset 14 (start of "line 3") → line 3
        assert body_offset_to_file_line(text, 14) == 3

    def test_with_frontmatter(self):
        text = "---\ntags: [x]\n---\nbody line 1\nbody line 2\n"
        # Frontmatter is 3 lines (---, tags, ---), so frontmatter_len = 18
        # Body offset 0 in body → line 4 in file (first body line)
        assert body_offset_to_file_line(text, 0) == 4
        # Body offset 12 (start of "body line 2") → line 5
        body = text.split("---\n", 2)[2]
        assert body.startswith("body line 1\n")
        offset_of_body_line_2 = body.index("body line 2")
        assert body_offset_to_file_line(text, offset_of_body_line_2) == 5

    def test_frontmatter_deleted_from_disk(self):
        """User deleted frontmatter after indexing. Offset is body-relative;
        the current on-disk text has no frontmatter. Mapping should still
        work — treat full file as body."""
        text = "body line 1\nbody line 2\n"
        assert body_offset_to_file_line(text, 0) == 1
        assert body_offset_to_file_line(text, 12) == 2

    def test_cjk_char_boundary(self):
        """Chinese characters must not shift line numbers (code-point offsets)."""
        text = "标题\n内容第一行\n内容第二行\n"
        # "标题\n" = 3 code points. Offset 3 = start of "内容第一行"
        assert body_offset_to_file_line(text, 3) == 2
        # "标题\n内容第一行\n" = 3 + 6 = 9 code points. Offset 9 = start of line 3
        assert body_offset_to_file_line(text, 9) == 3


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    v.mkdir()
    # One file with frontmatter
    (v / "note.md").write_text(
        "---\ntags: [a, b]\n---\nLine 1 of body\nLine 2 of body\nLine 3 of body\n",
        encoding="utf-8",
    )
    # One file without frontmatter
    (v / "plain.md").write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    # One longer file
    (v / "long.md").write_text("".join(f"line {i}\n" for i in range(1, 51)), encoding="utf-8")
    return v


def _run_get(vault: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run `seeklink get` against the vault, returning CompletedProcess."""
    cmd = [sys.executable, "-m", "seeklink", "get", "--vault", str(vault), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestGetCommand:
    def test_whole_file(self, vault: Path):
        r = _run_get(vault, ["plain.md"])
        assert r.returncode == 0
        assert r.stdout == "one\ntwo\nthree\nfour\nfive\n"

    def test_start_line(self, vault: Path):
        r = _run_get(vault, ["long.md:10", "-l", "3"])
        assert r.returncode == 0
        lines = r.stdout.rstrip("\n").split("\n")
        assert lines == ["line 10", "line 11", "line 12"]

    def test_context_window(self, vault: Path):
        r = _run_get(vault, ["long.md:10", "-C", "2"])
        assert r.returncode == 0
        lines = r.stdout.rstrip("\n").split("\n")
        assert lines == ["line 8", "line 9", "line 10", "line 11", "line 12"]

    def test_context_clamps_at_start(self, vault: Path):
        r = _run_get(vault, ["long.md:2", "-C", "5"])
        assert r.returncode == 0
        lines = r.stdout.rstrip("\n").split("\n")
        assert lines == [
            "line 1",
            "line 2",
            "line 3",
            "line 4",
            "line 5",
            "line 6",
            "line 7",
        ]

    def test_context_clamps_at_eof(self, vault: Path):
        r = _run_get(vault, ["long.md:49", "-C", "5"])
        assert r.returncode == 0
        lines = r.stdout.rstrip("\n").split("\n")
        assert lines == [
            "line 44",
            "line 45",
            "line 46",
            "line 47",
            "line 48",
            "line 49",
            "line 50",
        ]

    def test_context_zero_prints_hit_line_only(self, vault: Path):
        r = _run_get(vault, ["long.md:10", "-C", "0"])
        assert r.returncode == 0
        assert r.stdout.rstrip("\n") == "line 10"

    def test_context_requires_line_suffix(self, vault: Path):
        r = _run_get(vault, ["long.md", "-C", "2"])
        assert r.returncode == 1
        assert "requires PATH:LINE" in r.stderr

    def test_context_cannot_combine_with_lines(self, vault: Path):
        r = _run_get(vault, ["long.md:10", "-C", "2", "-l", "3"])
        assert r.returncode == 1
        assert "cannot be combined" in r.stderr

    def test_context_rejects_negative_value(self, vault: Path):
        r = _run_get(vault, ["long.md:10", "-C", "-1"])
        assert r.returncode == 1
        assert "must be >= 0" in r.stderr

    def test_default_lines_is_100(self, vault: Path):
        """With :LINE but no -l, default to 100 lines."""
        r = _run_get(vault, ["long.md:5"])
        assert r.returncode == 0
        # long.md is 50 lines; starting at 5 and asking for 100 should give 46
        assert r.stdout.count("\n") == 46

    def test_first_n_lines(self, vault: Path):
        r = _run_get(vault, ["long.md", "-l", "3"])
        assert r.returncode == 0
        lines = r.stdout.rstrip("\n").split("\n")
        assert lines == ["line 1", "line 2", "line 3"]

    def test_missing_file(self, vault: Path):
        r = _run_get(vault, ["does-not-exist.md"])
        assert r.returncode == 1
        assert "not found" in r.stderr

    def test_line_beyond_eof(self, vault: Path):
        r = _run_get(vault, ["plain.md:999"])
        assert r.returncode == 0
        assert r.stdout == ""
        assert "beyond EOF" in r.stderr

    def test_line_less_than_one(self, vault: Path):
        r = _run_get(vault, ["plain.md:0", "-l", "1"])
        assert r.returncode == 0
        # 0 gets clamped to 1, so we get line 1
        assert r.stdout.startswith("one")
        assert "clamping to 1" in r.stderr

    def test_frontmatter_file_line_numbers(self, vault: Path):
        """Line numbers should reference the full file, not the body."""
        r = _run_get(vault, ["note.md:4", "-l", "1"])
        assert r.returncode == 0
        # Line 4 of note.md (which has 3 lines of frontmatter) = "Line 1 of body"
        assert r.stdout.rstrip("\n") == "Line 1 of body"

    def test_path_escape_rejected(self, vault: Path):
        """Security: paths that escape the vault must be rejected by the
        explicit escape check, not just accidentally by file-not-found.

        We use a target path that `relative_to` will reject (../../..)
        and verify the explicit "escapes" error message is emitted.
        """
        r = _run_get(vault, ["../../../../../../etc/passwd"])
        assert r.returncode == 1
        assert "escapes" in r.stderr, (
            f"Expected explicit escape rejection, got stderr={r.stderr!r}"
        )

    def test_trailing_newline_eof_accounting(self, vault: Path):
        """split('\\n') on newline-terminated text has a trailing empty
        element that does NOT correspond to a real line. Asking for the
        line just past the last real line should emit the beyond-EOF
        warning, not return a blank line."""
        # plain.md is 5 logical lines, all ending with \n.
        r = _run_get(vault, ["plain.md:6"])
        assert r.returncode == 0
        assert "beyond EOF" in r.stderr, (
            f"Expected beyond-EOF warning for line 6 of 5-line file, "
            f"got stderr={r.stderr!r} stdout={r.stdout!r}"
        )
        assert r.stdout == "", (
            f"Expected empty stdout for beyond-EOF, got {r.stdout!r}"
        )

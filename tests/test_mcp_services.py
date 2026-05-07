"""Unit tests for the read-only MCP service layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from seeklink.mcp_services import ServiceError, mcp_get, mcp_status, resolve_mcp_vault


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (root / "plain.txt").write_text("not markdown\n", encoding="utf-8")
    (root / "visible.txt").write_text("visible non-markdown\n", encoding="utf-8")
    (root / ".secret.md").write_text("hidden\n", encoding="utf-8")
    (root / "long.md").write_text("".join(f"line {i}\n" for i in range(1, 2105)), encoding="utf-8")
    return root


def test_resolve_mcp_vault_requires_explicit_vault(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SEEKLINK_VAULT", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ServiceError) as excinfo:
        resolve_mcp_vault(None)

    assert excinfo.value.code == "NO_VAULT"


def test_resolve_mcp_vault_uses_env(monkeypatch, vault: Path):
    monkeypatch.setenv("SEEKLINK_VAULT", str(vault))

    assert resolve_mcp_vault(None) == vault.resolve()


def test_mcp_status_missing_index_does_not_create_seeklink_dir(vault: Path):
    payload = mcp_status(vault)

    assert payload["ok"] is True
    assert payload["database"]["exists"] is False
    assert payload["index"]["compatibility"]["state"] == "missing"
    assert not (vault / ".seeklink").exists()


def test_mcp_get_reads_line_window(vault: Path):
    payload = mcp_get(vault, path="note.md", line=2, lines=1)

    assert payload["ok"] is True
    assert payload["path"] == "note.md"
    assert payload["line_start"] == 2
    assert payload["line_end"] == 2
    assert payload["text"] == "two"


def test_mcp_get_accepts_path_line_suffix(vault: Path):
    payload = mcp_get(vault, path="note.md:2", context=1)

    assert payload["line_start"] == 1
    assert payload["line_end"] == 3
    assert payload["text"] == "one\ntwo\nthree\n"


def test_mcp_get_beyond_eof_is_success_with_warning(vault: Path):
    payload = mcp_get(vault, path="note.md", line=99)

    assert payload["ok"] is True
    assert payload["text"] == ""
    assert payload["warnings"]
    assert "beyond EOF" in payload["warnings"][0]


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("../outside.md", "PATH_OUTSIDE_VAULT"),
        (".secret.md", "ACCESS_DENIED"),
        ("plain.txt", "ACCESS_DENIED"),
    ],
)
def test_mcp_get_rejects_unsafe_paths(vault: Path, path: str, code: str):
    with pytest.raises(ServiceError) as excinfo:
        mcp_get(vault, path=path)

    assert excinfo.value.code == code


def test_mcp_get_rejects_symlink_to_hidden_file(vault: Path):
    link = vault / "linked-secret.md"
    try:
        link.symlink_to(vault / ".secret.md")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlink unavailable: {e}")

    with pytest.raises(ServiceError) as excinfo:
        mcp_get(vault, path="linked-secret.md")

    assert excinfo.value.code == "ACCESS_DENIED"


def test_mcp_get_rejects_symlink_that_escapes_vault(vault: Path, tmp_path: Path):
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    link = vault / "linked-outside.md"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlink unavailable: {e}")

    with pytest.raises(ServiceError) as excinfo:
        mcp_get(vault, path="linked-outside.md")

    assert excinfo.value.code == "PATH_OUTSIDE_VAULT"


def test_mcp_get_rejects_symlink_to_non_markdown_file(vault: Path):
    link = vault / "linked-visible.md"
    try:
        link.symlink_to(vault / "visible.txt")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlink unavailable: {e}")

    with pytest.raises(ServiceError) as excinfo:
        mcp_get(vault, path="linked-visible.md")

    assert excinfo.value.code == "ACCESS_DENIED"


def test_mcp_get_caps_large_file_output(vault: Path):
    payload = mcp_get(vault, path="long.md")

    assert payload["line_start"] == 1
    assert payload["line_end"] == 2000
    assert len(payload["text"].splitlines()) == 2000
    assert any("truncated" in warning for warning in payload["warnings"])

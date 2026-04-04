"""Pytest fixtures for SeekLink database tests."""

from pathlib import Path

import pytest

from seeklink.db import Database


@pytest.fixture
def db():
    """In-memory database for fast tests."""
    d = Database(":memory:")
    d.check_capabilities()
    d.init_schema()
    yield d
    d.close()


@pytest.fixture
def db_file(tmp_path: Path):
    """File-backed database for WAL / pragma tests."""
    db_path = tmp_path / "test.db"
    d = Database(db_path)
    d.check_capabilities()
    d.init_schema()
    yield d
    d.close()

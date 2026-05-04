"""Tests for persisted index-configuration compatibility checks."""

from __future__ import annotations

import pytest

from seeklink.db import Database
from seeklink.index_config import (
    compatibility_state,
    ensure_index_compatible_for_search,
    expected_index_metadata,
)


def test_empty_index_without_metadata_is_compatible(db: Database):
    expected = expected_index_metadata("model-a")

    state = compatibility_state(
        stored=db.get_index_metadata(),
        expected=expected,
        chunks_total=db.get_stats()["chunks_total"],
    )

    assert state == {
        "compatible": True,
        "state": "empty",
        "mismatches": {},
    }


def test_search_rejects_existing_chunks_without_metadata(db: Database):
    source = db.add_source(uid="uid-1", path="note.md")
    db.add_chunk(source.id, "indexed content", 0)

    with pytest.raises(RuntimeError, match="Index configuration does not match"):
        ensure_index_compatible_for_search(db, embedder_model="model-a")


def test_search_accepts_matching_metadata(db: Database):
    source = db.add_source(uid="uid-1", path="note.md")
    db.add_chunk(source.id, "indexed content", 0)
    db.set_index_metadata(expected_index_metadata("model-a"))

    ensure_index_compatible_for_search(db, embedder_model="model-a")


def test_search_rejects_vector_table_dimension_mismatch(db: Database):
    source = db.add_source(uid="uid-1", path="note.md")
    db.add_chunk(source.id, "indexed content", 0)
    db.set_index_metadata(expected_index_metadata("model-a", 384))

    with pytest.raises(RuntimeError, match="vec_chunks_dimension"):
        ensure_index_compatible_for_search(
            db,
            embedder_model="model-a",
            embedding_dim=384,
        )

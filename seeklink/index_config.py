"""Index-configuration metadata helpers.

The vector index is only valid for the embedder/chunker configuration that
created it. These helpers are deliberately lightweight so `seeklink status`
can report compatibility without importing fastembed or MLX.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


DEFAULT_EMBEDDER_MODEL = "jinaai/jina-embeddings-v2-base-zh"
EMBEDDING_BACKEND = "fastembed"
EMBEDDING_DIM = "768"
VECTOR_DISTANCE_METRIC = "cosine"
CHUNKER_VERSION = "1"
INDEX_METADATA_VERSION = "1"

INDEX_METADATA_KEYS = (
    "index_metadata_version",
    "embedder_model",
    "embedding_backend",
    "embedding_dim",
    "vector_distance_metric",
    "chunker_version",
)


def resolve_embedder_model() -> str:
    """Return the configured embedder name without loading the model."""
    return os.environ.get("SEEKLINK_EMBEDDER_MODEL", DEFAULT_EMBEDDER_MODEL)


def expected_index_metadata(embedder_model: str | None = None) -> dict[str, str]:
    """Return the index metadata expected for the active configuration."""
    return {
        "index_metadata_version": INDEX_METADATA_VERSION,
        "embedder_model": embedder_model or resolve_embedder_model(),
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_dim": EMBEDDING_DIM,
        "vector_distance_metric": VECTOR_DISTANCE_METRIC,
        "chunker_version": CHUNKER_VERSION,
    }


def metadata_mismatches(
    stored: Mapping[str, str],
    expected: Mapping[str, str],
) -> dict[str, tuple[str | None, str]]:
    """Return keys whose stored value does not match the expected value."""
    return {
        key: (stored.get(key), expected[key])
        for key in INDEX_METADATA_KEYS
        if stored.get(key) != expected[key]
    }


def describe_mismatches(
    mismatches: Mapping[str, tuple[str | None, str]],
) -> str:
    """Format a concise, user-readable mismatch summary."""
    parts = []
    for key, (stored, expected) in mismatches.items():
        stored_label = "missing" if stored is None else repr(stored)
        parts.append(f"{key}: {stored_label} != {expected!r}")
    return "; ".join(parts)


def compatibility_state(
    *,
    stored: Mapping[str, str],
    expected: Mapping[str, str],
    chunks_total: int,
) -> dict:
    """Return a JSON-serializable compatibility summary."""
    mismatches = metadata_mismatches(stored, expected)
    if not mismatches:
        return {
            "compatible": True,
            "state": "ok",
            "mismatches": {},
        }
    if chunks_total == 0:
        return {
            "compatible": True,
            "state": "empty",
            "mismatches": {},
        }
    return {
        "compatible": False,
        "state": "mismatch" if stored else "missing",
        "mismatches": {
            key: {"stored": stored_value, "expected": expected_value}
            for key, (stored_value, expected_value) in mismatches.items()
        },
    }


def ensure_index_compatible_for_search(
    db,
    *,
    embedder_model: str,
) -> None:
    """Raise RuntimeError if existing vectors do not match active config."""
    stats = db.get_stats()
    expected = expected_index_metadata(embedder_model)
    stored = db.get_index_metadata()
    state = compatibility_state(
        stored=stored,
        expected=expected,
        chunks_total=stats["chunks_total"],
    )
    if state["compatible"]:
        return

    mismatches = metadata_mismatches(stored, expected)
    raise RuntimeError(
        "Index configuration does not match the active embedder/chunker "
        f"settings ({describe_mismatches(mismatches)}). Run full "
        "`seeklink index` for this vault to rebuild the index."
    )

"""Synapsis — personal knowledge management engine."""

from synapsis.chunker import ChunkSpan, chunk_markdown
from synapsis.db import CapabilityError, Database
from synapsis.embedder import Embedder
from synapsis.ingest import ingest_file, ingest_vault
from synapsis.link_parser import extract_wiki_links
from synapsis.models import BudgetEntry, Chunk, Source, Suggestion, WikiLink
from synapsis.search import SearchResult, search

__all__ = [
    "Database",
    "CapabilityError",
    "Source",
    "Chunk",
    "WikiLink",
    "Suggestion",
    "BudgetEntry",
    "Embedder",
    "ChunkSpan",
    "chunk_markdown",
    "extract_wiki_links",
    "ingest_file",
    "ingest_vault",
    "search",
    "SearchResult",
]

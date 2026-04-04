"""Sophia — personal knowledge management engine."""

from sophia.chunker import ChunkSpan, chunk_markdown
from sophia.db import CapabilityError, Database
from sophia.embedder import Embedder
from sophia.ingest import ingest_file, ingest_vault
from sophia.link_parser import extract_wiki_links
from sophia.models import BudgetEntry, Chunk, Source, Suggestion, WikiLink
from sophia.search import SearchResult, search

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

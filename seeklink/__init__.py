"""SeekLink — personal knowledge management engine."""

from seeklink.chunker import ChunkSpan, chunk_markdown
from seeklink.db import CapabilityError, Database
from seeklink.embedder import Embedder
from seeklink.ingest import ingest_file, ingest_vault
from seeklink.link_parser import extract_wiki_links
from seeklink.models import BudgetEntry, Chunk, Source, Suggestion, WikiLink
from seeklink.search import SearchResult, search

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

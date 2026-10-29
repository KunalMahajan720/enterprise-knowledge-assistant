from pathlib import Path
from typing import TypedDict

from src.document_loader import load_documents
from src.text_chunker import chunk_documents


class IngestionResult(TypedDict):
    document_count: int
    chunk_count: int
    documents: list[dict[str, str]]
    chunks: list[dict[str, object]]


def ingest_documents(
    directory: str | Path = "documents",
    chunk_size: int = 500,
    overlap: int = 100,
) -> IngestionResult:
    """
    Load documents and split their text into overlapping chunks.

    Args:
        directory: Folder containing knowledge documents.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters.

    Returns:
        Document and chunk data with summary counts.
    """
    documents = load_documents(directory)

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents": documents,
        "chunks": chunks,
    }
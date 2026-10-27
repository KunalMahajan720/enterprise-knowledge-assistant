from typing import TypedDict


class TextChunk(TypedDict):
    """A text chunk and its source metadata."""

    source: str
    chunk_index: int
    text: str


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """
    Split loaded documents into chunks while preserving filenames.

    Each input document should contain 'source' and 'text' fields.
    """
    chunks: list[TextChunk] = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = split_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk_text in enumerate(document_chunks):
            chunks.append(
                {
                    "source": source,
                    "chunk_index": index,
                    "text": chunk_text,
                }
            )

    return chunks
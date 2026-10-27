






import pytest

from src.text_chunker import chunk_documents, split_text


def test_empty_text_returns_no_chunks() -> None:
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text_returns_one_chunk() -> None:
    text = "Employees receive annual leave."

    chunks = split_text(text, chunk_size=100, overlap=20)

    assert chunks == [text]


def test_long_text_is_split_into_chunks() -> None:
    text = "A" * 1200

    chunks = split_text(text, chunk_size=500, overlap=100)

    assert len(chunks) == 3
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_chunks_overlap_correctly() -> None:
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    chunks = split_text(text, chunk_size=10, overlap=3)

    assert chunks[0] == "ABCDEFGHIJ"
    assert chunks[1] == "HIJKLMNOPQ"
    assert chunks[2] == "OPQRSTUVWX"
    assert chunks[3] == "VWXYZ"


def test_invalid_chunk_size_raises_error() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        split_text("Sample text", chunk_size=0, overlap=0)


def test_invalid_overlap_raises_error() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("Sample text", chunk_size=10, overlap=10)


def test_negative_overlap_raises_error() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("Sample text", chunk_size=10, overlap=-1)


def test_document_metadata_is_preserved() -> None:
    documents = [
        {
            "source": "hr_policy.txt",
            "text": "A" * 1200,
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    assert len(chunks) == 3
    assert chunks[0]["source"] == "hr_policy.txt"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[2]["chunk_index"] == 2


def test_multiple_documents_are_chunked_separately() -> None:
    documents = [
        {"source": "hr.txt", "text": "A" * 600},
        {"source": "it.txt", "text": "B" * 600},
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    assert len(chunks) == 4
    assert chunks[0]["source"] == "hr.txt"
    assert chunks[2]["source"] == "it.txt"
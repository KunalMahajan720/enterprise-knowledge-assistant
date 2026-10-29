
from src import ingestion_pipeline


def test_ingest_documents_returns_counts_and_data(monkeypatch):
    """The pipeline should return loaded documents, chunks, and counts."""
    documents = [
        {
            "source": "hr_policy.txt",
            "text": "Employees receive annual leave.",
        },
        {
            "source": "it_support.txt",
            "text": "Contact IT for password resets.",
        },
    ]

    chunks = [
        {
            "source": "hr_policy.txt",
            "chunk_index": 0,
            "text": "Employees receive annual leave.",
        },
        {
            "source": "it_support.txt",
            "chunk_index": 0,
            "text": "Contact IT for password resets.",
        },
    ]

    monkeypatch.setattr(
        ingestion_pipeline,
        "load_documents",
        lambda directory: documents,
    )

    def fake_chunk_documents(
        received_documents,
        chunk_size,
        overlap,
    ):
        assert received_documents == documents
        assert chunk_size == 500
        assert overlap == 100
        return chunks

    monkeypatch.setattr(
        ingestion_pipeline,
        "chunk_documents",
        fake_chunk_documents,
    )

    result = ingestion_pipeline.ingest_documents("sample_documents")

    assert result["document_count"] == 2
    assert result["chunk_count"] == 2
    assert result["documents"] == documents
    assert result["chunks"] == chunks


def test_ingest_documents_with_no_documents(monkeypatch):
    """An empty input should produce zero documents and zero chunks."""
    monkeypatch.setattr(
        ingestion_pipeline,
        "load_documents",
        lambda directory: [],
    )

    monkeypatch.setattr(
        ingestion_pipeline,
        "chunk_documents",
        lambda documents, chunk_size, overlap: [],
    )

    result = ingestion_pipeline.ingest_documents("empty_documents")

    assert result["document_count"] == 0
    assert result["chunk_count"] == 0
    assert result["documents"] == []
    assert result["chunks"] == []


def test_ingest_documents_passes_custom_chunk_settings(monkeypatch):
    """Custom chunk size and overlap should reach the chunking function."""
    monkeypatch.setattr(
        ingestion_pipeline,
        "load_documents",
        lambda directory: [
            {"source": "example.txt", "text": "Example content"}
        ],
    )

    def fake_chunk_documents(documents, chunk_size, overlap):
        assert chunk_size == 200
        assert overlap == 40
        return [
            {
                "source": "example.txt",
                "chunk_index": 0,
                "text": "Example content",
            }
        ]

    monkeypatch.setattr(
        ingestion_pipeline,
        "chunk_documents",
        fake_chunk_documents,
    )

    result = ingestion_pipeline.ingest_documents(
        "sample_documents",
        chunk_size=200,
        overlap=40,
    )

    assert result["document_count"] == 1
    assert result["chunk_count"] == 1

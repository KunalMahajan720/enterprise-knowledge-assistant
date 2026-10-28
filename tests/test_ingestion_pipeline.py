from pathlib import Path

import pytest

from src.ingestion_pipeline import ingest_documents


def test_ingestion_pipeline_processes_documents(
    tmp_path: Path,
) -> None:
    """The pipeline should load and chunk supported documents."""
    (tmp_path / "hr.txt").write_text(
        "A" * 1200,
        encoding="utf-8",
    )
    (tmp_path / "it.md").write_text(
        "B" * 600,
        encoding="utf-8",
    )

    result = ingest_documents(
        directory=tmp_path,
        chunk_size=500,
        overlap=100,
    )

    assert result["document_count"] == 2
    assert result["chunk_count"] == 5
    assert len(result["documents"]) == 2
    assert len(result["chunks"]) == 5


def test_pipeline_preserves_source_metadata(
    tmp_path: Path,
) -> None:
    """Each chunk should retain its original filename."""
    (tmp_path / "hr.txt").write_text(
        "Annual leave policy.",
        encoding="utf-8",
    )

    result = ingest_documents(
        directory=tmp_path,
        chunk_size=500,
        overlap=100,
    )

    assert result["chunks"][0]["source"] == "hr.txt"
    assert result["chunks"][0]["chunk_index"] == 0


def test_empty_directory_returns_empty_results(
    tmp_path: Path,
) -> None:
    """An empty directory should produce no documents or chunks."""
    result = ingest_documents(directory=tmp_path)

    assert result["document_count"] == 0
    assert result["chunk_count"] == 0
    assert result["documents"] == []
    assert result["chunks"] == []


def test_missing_directory_raises_error(
    tmp_path: Path,
) -> None:
    """A missing directory should raise FileNotFoundError."""
    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ingest_documents(directory=missing_directory)
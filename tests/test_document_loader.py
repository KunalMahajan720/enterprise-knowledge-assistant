from pathlib import Path

import pytest

from src.document_loader import load_document, load_documents


def test_load_text_document(tmp_path: Path) -> None:
    """A text file should load with its filename and content."""
    file_path = tmp_path / "policy.txt"
    file_path.write_text("Employees receive annual leave.", encoding="utf-8")

    document = load_document(file_path)

    assert document["source"] == "policy.txt"
    assert document["text"] == "Employees receive annual leave."


def test_load_markdown_document(tmp_path: Path) -> None:
    """A Markdown file should be supported."""
    file_path = tmp_path / "guide.md"
    file_path.write_text("# IT Guide\nReset your password.", encoding="utf-8")

    document = load_document(file_path)

    assert document["source"] == "guide.md"
    assert "Reset your password." in document["text"]


def test_unsupported_file_format(tmp_path: Path) -> None:
    """Unsupported file formats should raise ValueError."""
    file_path = tmp_path / "data.csv"
    file_path.write_text("name,value", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_document(file_path)


def test_load_documents_from_directory(tmp_path: Path) -> None:
    """The directory loader should load supported files."""
    (tmp_path / "hr.txt").write_text("Annual leave policy.", encoding="utf-8")
    (tmp_path / "it.md").write_text("# IT\nPassword reset policy.", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("not supported", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert [document["source"] for document in documents] == ["hr.txt", "it.md"]


def test_missing_file_raises_error(tmp_path: Path) -> None:
    """A missing file should raise FileNotFoundError."""
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(file_path)
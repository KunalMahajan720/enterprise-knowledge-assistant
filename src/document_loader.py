from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class Document(TypedDict):
    """Represents an extracted knowledge document."""

    source: str
    text: str


def load_document(file_path: Path) -> Document:
    """
    Load a single TXT, Markdown, or PDF document.

    Args:
        file_path: Path to the document.

    Returns:
        A dictionary containing the filename and extracted text.

    Raises:
        ValueError: If the file format is unsupported.
        OSError: If the file cannot be accessed.
    """
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {extension}")

    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if extension == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = file_path.read_text(encoding="utf-8-sig")

    return {
        "source": file_path.name,
        "text": text.strip(),
    }


def load_documents(directory: str | Path) -> list[Document]:
    """
    Load all supported documents from a directory.

    Files are processed in alphabetical order. If an individual
    document fails to load, the error is reported and processing
    continues with the remaining files.
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Documents directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, but received: {directory_path}"
        )

    documents: list[Document] = []

    for file_path in sorted(directory_path.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            document = load_document(file_path)

            if not document["text"]:
                print(f"Warning: No text extracted from {file_path.name}")
                continue

            documents.append(document)

        except Exception as exc:
            print(f"Warning: Could not load {file_path.name}: {exc}")

    return documents
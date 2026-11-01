from pathlib import Path
from typing import Any

from src.embedding_service import EmbeddingService
from src.ingestion_pipeline import ingest_documents
from src.vector_store import VectorStore

class SemanticSearch:
    """Index knowledge documents and search them by semantic similarity."""

    def __init__(
        self,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = "enterprise_knowledge",
        embedding_service: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else VectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        )

    def index_documents(
        self,
        directory: str | Path = "documents",
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> dict[str, int]:
        """Load documents, create embeddings, and store their chunks."""

        ingestion_result = ingest_documents(
            directory=directory,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks = ingestion_result["chunks"]

        if not chunks:
            return {
                "document_count": ingestion_result["document_count"],
                "chunk_count": 0,
                "stored_count": 0,
            }

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_service.embed_texts(texts)

        stored_count = self.vector_store.upsert_chunks(
            chunks,
            embeddings,
        )

        return {
            "document_count": ingestion_result["document_count"],
            "chunk_count": len(chunks),
            "stored_count": stored_count,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant document chunks for a question."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.embedding_service.embed_query(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
from pathlib import Path
from typing import TypedDict

import chromadb


DEFAULT_COLLECTION_NAME = "enterprise_knowledge"


class DocumentChunk(TypedDict):
    source: str
    chunk_index: int
    text: str


class SearchResult(TypedDict):
    id: str
    text: str
    metadata: dict[str, str | int]
    distance: float


class VectorStore:
    """Store document chunks and their embeddings in local ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Insert or update chunks and their embeddings.

        Each chunk must contain source, chunk_index, and text.
        The embeddings list must contain one vector per chunk.

        Returns the number of chunks submitted to ChromaDB.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []

        for chunk in chunks:
            source = chunk["source"]
            chunk_index = chunk["chunk_index"]
            text = chunk["text"]

            if not source.strip():
                raise ValueError("Chunk source cannot be empty.")

            if not text.strip():
                raise ValueError("Chunk text cannot be empty.")

            ids.append(f"{source}::chunk::{chunk_index}")
            documents.append(text)
            metadatas.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                }
            )

        for embedding in embeddings:
            if not embedding:
                raise ValueError("Embeddings cannot be empty.")

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieve the most relevant stored chunks for a query vector.

        Results are ordered by increasing cosine distance.
        A smaller distance indicates a closer match.
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        search_results: list[SearchResult] = []

        for result_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            if document is None or metadata is None or distance is None:
                continue

            search_results.append(
                {
                    "id": result_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )

        return search_results
import pytest

from src import semantic_search


class FakeEmbeddingService:
    """Fake embedding service for testing without a real model."""

    def __init__(self):
        self.received_texts = []
        self.received_queries = []

    def embed_texts(self, texts):
        self.received_texts = texts
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query):
        self.received_queries.append(query)
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    """Fake vector store for testing indexing and search."""

    def __init__(self):
        self.received_chunks = []
        self.received_embeddings = []
        self.received_query_embedding = None
        self.received_top_k = None

    def upsert_chunks(self, chunks, embeddings):
        self.received_chunks = chunks
        self.received_embeddings = embeddings
        return len(chunks)

    def search(self, query_embedding, top_k=5):
        self.received_query_embedding = query_embedding
        self.received_top_k = top_k

        return [
            {
                "id": "hr_policy.txt::chunk::0",
                "text": "Employees receive annual paid leave.",
                "metadata": {
                    "source": "hr_policy.txt",
                    "chunk_index": 0,
                },
                "distance": 0.05,
            }
        ]


@pytest.fixture
def search_service():
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = semantic_search.SemanticSearch(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    return service, embedding_service, vector_store


def test_index_documents_generates_and_stores_embeddings(
    monkeypatch,
    search_service,
):
    service, embedding_service, vector_store = search_service

    chunks = [
        {
            "source": "hr_policy.txt",
            "chunk_index": 0,
            "text": "Employees receive annual paid leave.",
        },
        {
            "source": "it_support.txt",
            "chunk_index": 0,
            "text": "Contact IT to reset your password.",
        },
    ]

    monkeypatch.setattr(
        semantic_search,
        "ingest_documents",
        lambda **kwargs: {
            "document_count": 2,
            "chunk_count": 2,
            "documents": [],
            "chunks": chunks,
        },
    )

    result = service.index_documents("documents")

    assert result == {
        "document_count": 2,
        "chunk_count": 2,
        "stored_count": 2,
    }

    assert embedding_service.received_texts == [
        "Employees receive annual paid leave.",
        "Contact IT to reset your password.",
    ]

    assert len(vector_store.received_chunks) == 2
    assert len(vector_store.received_embeddings) == 2


def test_index_documents_passes_chunk_settings(
    monkeypatch,
    search_service,
):
    service, _, _ = search_service
    received_arguments = {}

    def fake_ingest_documents(**kwargs):
        received_arguments.update(kwargs)

        return {
            "document_count": 0,
            "chunk_count": 0,
            "documents": [],
            "chunks": [],
        }

    monkeypatch.setattr(
        semantic_search,
        "ingest_documents",
        fake_ingest_documents,
    )

    service.index_documents(
        directory="sample_documents",
        chunk_size=200,
        overlap=40,
    )

    assert received_arguments["directory"] == "sample_documents"
    assert received_arguments["chunk_size"] == 200
    assert received_arguments["overlap"] == 40


def test_index_documents_handles_empty_chunks(
    monkeypatch,
    search_service,
):
    service, embedding_service, vector_store = search_service

    monkeypatch.setattr(
        semantic_search,
        "ingest_documents",
        lambda **kwargs: {
            "document_count": 0,
            "chunk_count": 0,
            "documents": [],
            "chunks": [],
        },
    )

    result = service.index_documents("empty_documents")

    assert result == {
        "document_count": 0,
        "chunk_count": 0,
        "stored_count": 0,
    }

    assert embedding_service.received_texts == []
    assert vector_store.received_chunks == []


def test_search_returns_relevant_results(search_service):
    service, embedding_service, vector_store = search_service

    results = service.search(
        "How much annual leave do employees receive?",
        top_k=3,
    )

    assert len(results) == 1

    assert results[0]["text"] == (
        "Employees receive annual paid leave."
    )

    assert results[0]["metadata"]["source"] == "hr_policy.txt"

    assert embedding_service.received_queries == [
        "How much annual leave do employees receive?"
    ]

    assert vector_store.received_query_embedding == [1.0, 0.0, 0.0]
    assert vector_store.received_top_k == 3


def test_search_rejects_empty_query(search_service):
    service, _, _ = search_service

    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.search("   ")


def test_search_rejects_invalid_top_k(search_service):
    service, _, _ = search_service

    with pytest.raises(ValueError, match="top_k"):
        service.search("Find the leave policy", top_k=0)
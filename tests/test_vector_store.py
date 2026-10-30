import pytest

from src.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create an isolated, temporary ChromaDB database for each test."""
    return VectorStore(
        persist_directory=tmp_path / "test_chroma_db",
        collection_name="test_knowledge",
    )


def test_upsert_chunks_stores_documents_and_embeddings(vector_store):
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

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    count = vector_store.upsert_chunks(chunks, embeddings)

    assert count == 2
    assert vector_store.collection.count() == 2


def test_search_returns_most_relevant_chunk_first(vector_store):
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

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    vector_store.upsert_chunks(chunks, embeddings)

    results = vector_store.search(
        query_embedding=[0.99, 0.01, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == (
        "Employees receive annual paid leave."
    )
    assert results[0]["metadata"]["source"] == "hr_policy.txt"
    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[0]["distance"] < results[1]["distance"]


def test_upsert_empty_chunks_returns_zero(vector_store):
    count = vector_store.upsert_chunks([], [])

    assert count == 0
    assert vector_store.collection.count() == 0


def test_upsert_rejects_mismatched_chunk_and_embedding_counts(
    vector_store,
):
    chunks = [
        {
            "source": "hr_policy.txt",
            "chunk_index": 0,
            "text": "Employees receive annual paid leave.",
        }
    ]

    embeddings = []

    with pytest.raises(ValueError, match="number of chunks"):
        vector_store.upsert_chunks(chunks, embeddings)


def test_search_rejects_empty_query_embedding(vector_store):
    with pytest.raises(ValueError, match="Query embedding"):
        vector_store.search([])


def test_search_rejects_invalid_top_k(vector_store):
    with pytest.raises(ValueError, match="top_k"):
        vector_store.search([1.0, 0.0, 0.0], top_k=0)


def test_search_returns_empty_list_for_empty_collection(vector_store):
    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert results == []


def test_upsert_updates_existing_chunk(vector_store):
    original_chunk = [
        {
            "source": "hr_policy.txt",
            "chunk_index": 0,
            "text": "Old policy text.",
        }
    ]

    updated_chunk = [
        {
            "source": "hr_policy.txt",
            "chunk_index": 0,
            "text": "Updated policy text.",
        }
    ]

    vector_store.upsert_chunks(original_chunk, [[1.0, 0.0, 0.0]])
    vector_store.upsert_chunks(updated_chunk, [[1.0, 0.0, 0.0]])

    assert vector_store.collection.count() == 1

    results = vector_store.search([1.0, 0.0, 0.0], top_k=1)

    assert results[0]["text"] == "Updated policy text."

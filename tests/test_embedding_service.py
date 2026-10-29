import pytest

from src.embedding_service import EmbeddingService


class FakeEmbeddingModel:
    """Small mock model for testing without downloading an ML model."""

    def encode(
        self,
        texts,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        assert show_progress_bar is False

        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


@pytest.fixture
def embedding_service(monkeypatch):
    """Create an embedding service with a fake model."""

    monkeypatch.setattr(
        "src.embedding_service.SentenceTransformer",
        lambda model_name: FakeEmbeddingModel(),
    )

    return EmbeddingService()


def test_embed_multiple_texts(embedding_service):
    texts = [
        "Employees receive annual leave.",
        "The IT team resets passwords.",
    ]

    embeddings = embedding_service.embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert len(embeddings[1]) == 3


def test_embed_empty_list_returns_empty_list(embedding_service):
    assert embedding_service.embed_texts([]) == []


def test_embed_query_returns_single_vector(embedding_service):
    embedding = embedding_service.embed_query(
        "How many days of annual leave are available?"
    )

    assert isinstance(embedding, list)
    assert len(embedding) == 3


def test_empty_query_raises_error(embedding_service):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedding_service.embed_query("   ")


def test_empty_passage_raises_error(embedding_service):
    with pytest.raises(ValueError, match="empty passages"):
        embedding_service.embed_texts(["Valid text", "   "])
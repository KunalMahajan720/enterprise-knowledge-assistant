from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Generate text embeddings using a Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        """
        Initialize the embedding model.

        The first initialization may download the model.
        Subsequent runs normally reuse the local model cache.
        """
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert multiple text passages into numerical vectors.

        Args:
            texts: Text passages to encode.

        Returns:
            One embedding vector per input passage.
        """
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Texts must not contain empty passages")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    def embed_query(self, query: str) -> list[float]:
        """Convert a single user question into an embedding."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return self.embed_texts([query])[0]
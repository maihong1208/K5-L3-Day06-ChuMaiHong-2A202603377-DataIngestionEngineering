"""Local multilingual sentence-transformer embeddings; no API key is used."""
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Return the model-tokenizer count for each input string."""
        ...


class MyEmbedder:
    """Embed text locally with a real Sentence Transformers model."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from importlib.metadata import version

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.library_version = version("sentence-transformers")
        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=False,
        )
        first_module = self._model[0]
        auto_model = getattr(first_module, "auto_model", None)
        config = getattr(auto_model, "config", None)
        max_positions = getattr(config, "max_position_embeddings", None)
        if max_positions is not None:
            self._model.max_seq_length = min(256, max_positions)
        dimensions = self._model.get_sentence_embedding_dimension()
        if dimensions is None or dimensions <= 0:
            raise RuntimeError(
                f"Embedding model did not report a valid dimension: {model_name}"
            )
        self.dimensions = dimensions
        self.revision = getattr(config, "_commit_hash", None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("question must not be empty")
        vector = self._model.encode(
            question,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.tolist()

    def count_tokens(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        token_counts = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            encoded = self._model.tokenizer(
                batch,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            token_counts.extend(len(input_ids) for input_ids in encoded["input_ids"])
        return token_counts

"""Persistent local Chroma vector store with cosine similarity and metadata filters."""
import math
import re
from pathlib import Path
from typing import Any

from .models import Chunk


class ChromaVectorStore:
    _FILTER_FIELDS = {
        "document_id",
        "source",
        "page",
        "section",
        "year",
        "version",
        "category",
        "audience",
        "language",
        "document_type",
    }

    def __init__(
        self,
        persist_directory: Path,
        dimensions: int,
        client: Any | None = None,
    ) -> None:
        """Create a persistent Chroma client configured for real vector search."""
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        if client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        else:
            self._client = client
        self._collections: dict[str, Any] = {}

    def create_schema(self) -> None:
        """PersistentClient creates the on-disk schema on first collection write."""
        return None

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace this document's index contents without accumulating stale chunks."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal lengths")
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk IDs must be unique within an upsert")
        for vector in vectors:
            self._validate_vector(vector)

        collection = self._collection(strategy)
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("upsert accepts chunks for exactly one document")

        document_id = next(iter(document_ids))
        existing = collection.get(where={"document_id": document_id})
        existing_ids = set(existing.get("ids", []))

        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                }
            )
            invalid = [
                key
                for key, value in metadata.items()
                if not isinstance(value, (str, int, float, bool))
            ]
            if invalid:
                raise ValueError(
                    f"Chroma metadata must contain scalar values; invalid: {invalid}"
                )
            metadatas.append(metadata)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )
        stale_ids = existing_ids.difference(ids)
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks, optionally constrained by equality filters.

        Scores are cosine similarity (1 - Chroma cosine distance).
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._validate_vector(query_vector)
        where = None
        if filters:
            unknown_fields = set(filters).difference(self._FILTER_FIELDS)
            if unknown_fields:
                raise ValueError(f"Unsupported metadata filter fields: {sorted(unknown_fields)}")
            if any(value is None for value in filters.values()):
                raise ValueError("metadata filters must not contain null values")
            where = {
                key: {"$eq": value}
                for key, value in filters.items()
            }

        results = self._collection(strategy).query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        hits = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": 1.0 - float(distance),
                    "distance": float(distance),
                    "metadata": metadata or {},
                }
            )
        return hits

    def _collection(self, strategy: str) -> Any:
        key = re.sub(r"[^a-zA-Z0-9_-]", "_", strategy.lower()).strip("_-")
        if not key:
            raise ValueError("strategy must contain letters or digits")
        name = f"day6_{key}"
        if len(name) > 63:
            raise ValueError("strategy name is too long for a Chroma collection")
        if name not in self._collections:
            collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            collection_metadata = collection.metadata or {}
            metric = collection_metadata.get("hnsw:space", "cosine")
            if metric != "cosine":
                raise RuntimeError(
                    f"Collection {name} uses {metric!r}, expected cosine distance"
                )
            self._collections[name] = collection
        return self._collections[name]

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match {self.dimensions}"
            )
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("vectors must contain only finite numbers")

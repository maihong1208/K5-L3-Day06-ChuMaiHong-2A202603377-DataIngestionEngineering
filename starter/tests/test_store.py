import gc
import tempfile
import unittest
from pathlib import Path

from chromadb import EphemeralClient

from src.models import Chunk
from src.store import ChromaVectorStore


class ChromaStoreTests(unittest.TestCase):
    def test_persistent_store_upserts_replaces_and_filters_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(
                Path(temp_dir),
                dimensions=2,
                client=EphemeralClient(),
            )
            first = Chunk(
                chunk_id="first",
                document_id="doc-1",
                source="source.pdf",
                page=1,
                text="Older chunk.",
                metadata={"section": "LLM01:2025"},
            )
            second = Chunk(
                chunk_id="second",
                document_id="doc-1",
                source="source.pdf",
                page=2,
                text="Filtered target.",
                metadata={"section": "LLM02:2025"},
            )

            store.upsert("C", [first, second], [[1.0, 0.0], [0.0, 1.0]])
            store.upsert("C", [second], [[0.0, 1.0]])

            hits = store.search(
                "C",
                [0.0, 1.0],
                top_k=3,
                filters={"section": "LLM02:2025"},
            )

        self.assertEqual([hit["chunk_id"] for hit in hits], ["second"])
        self.assertEqual(hits[0]["metadata"]["page"], 2)

    def test_rejects_invalid_vector_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(
                Path(temp_dir),
                dimensions=2,
                client=EphemeralClient(),
            )
            chunk = Chunk(
                chunk_id="chunk",
                document_id="doc-1",
                source="source.pdf",
                page=1,
                text="Text.",
            )
            with self.assertRaises(ValueError):
                store.upsert("A", [chunk], [[1.0]])


if __name__ == "__main__":
    unittest.main()

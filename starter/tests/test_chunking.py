import unittest

from src.chunking import by_paragraph, fixed_size
from src.models import DocumentPage


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.page = DocumentPage(
            document_id="doc-1",
            source="source.pdf",
            page=7,
            text="A" * 1200,
            metadata={"section": "LLM01:2025"},
        )

    def test_fixed_size_respects_size_overlap_and_traceability(self):
        chunks = fixed_size(self.page, size=500, overlap=50)

        self.assertEqual([len(chunk.text) for chunk in chunks], [500, 500, 300])
        self.assertEqual(chunks[1].metadata["char_start"], 450)
        self.assertEqual(chunks[0].page, 7)
        self.assertEqual(chunks[0].metadata["section"], "LLM01:2025")
        self.assertEqual(chunks, fixed_size(self.page, size=500, overlap=50))

    def test_fixed_size_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=100, overlap=100)

    def test_paragraph_chunking_groups_paragraphs_and_splits_long_text(self):
        page = DocumentPage(
            document_id="doc-1",
            source="source.pdf",
            page=8,
            text="Short first paragraph.\n\nShort second paragraph.\n\n" + ("word " * 100),
        )
        chunks = by_paragraph(page, max_size=80)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(0 < len(chunk.text) <= 80 for chunk in chunks))
        self.assertEqual(chunks[0].text, "Short first paragraph.\n\nShort second paragraph.")
        self.assertTrue(all(chunk.metadata["chunk_strategy"] == "paragraph" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()

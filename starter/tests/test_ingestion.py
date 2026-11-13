import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion import load_pdf, normalize_text


class IngestionTests(unittest.TestCase):
    def test_normalize_text_preserves_paragraphs_and_normalizes_whitespace(self):
        self.assertEqual(
            normalize_text("  Cafe\u0301 \t test\r\n\r\n second   line \n\n\n"),
            "Café test\n\nsecond line",
        )

    def test_load_pdf_keeps_page_numbers_and_marks_empty_pages(self):
        fake_pypdf = types.ModuleType("pypdf")

        class FakePage:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class FakeReader:
            def __init__(self, path):
                self.pages = [
                    FakePage(" First page "),
                    FakePage(None),
                    FakePage(" Third page "),
                ]

        fake_pypdf.PdfReader = FakeReader
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "document.pdf"
            pdf_path.touch()
            with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
                with self.assertLogs("src.ingestion", level="WARNING"):
                    pages = load_pdf(pdf_path, "doc-1", "source.pdf")

        self.assertEqual([page.page for page in pages], [1, 2, 3])
        self.assertEqual(pages[0].text, "First page")
        self.assertEqual(pages[0].metadata, {})
        self.assertEqual(pages[1].text, "")
        self.assertEqual(pages[1].metadata["extraction_status"], "empty")
        self.assertEqual(pages[2].text, "Third page")

    def test_load_pdf_selects_original_physical_page_numbers(self):
        fake_pypdf = types.ModuleType("pypdf")

        class FakePage:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class FakeReader:
            def __init__(self, path):
                self.pages = [
                    FakePage(" First "),
                    FakePage(" Second "),
                    FakePage(" Third "),
                ]

        fake_pypdf.PdfReader = FakeReader
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "document.pdf"
            pdf_path.touch()
            with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
                pages = load_pdf(
                    pdf_path,
                    "doc-1",
                    "source.pdf",
                    page_numbers={1, 3},
                )

        self.assertEqual([page.page for page in pages], [1, 3])
        self.assertEqual([page.text for page in pages], ["First", "Third"])

    def test_load_pdf_reports_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_pdf(Path("missing.pdf"), "doc-1", "source.pdf")


if __name__ == "__main__":
    unittest.main()

"""Read and normalize source data."""
import logging
import re
import unicodedata
from collections.abc import Collection
from pathlib import Path

from .models import DocumentPage

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    page_numbers: Collection[int] | None = None,
) -> list[DocumentPage]:
    """Extract each physical page and retain its page number and source.

    Pages without extractable text are kept with an empty-text status so they
    can be reviewed or processed with OCR separately. Optional page numbers
    refer to 1-based physical pages in the original PDF.
    """
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    selected_pages = None if page_numbers is None else set(page_numbers)
    if selected_pages is not None:
        if not selected_pages:
            raise ValueError("page_numbers must not be empty")
        invalid_pages = [
            page for page in selected_pages
            if page < 1 or page > len(reader.pages)
        ]
        if invalid_pages:
            raise ValueError(
                f"Page numbers outside the PDF range: {sorted(invalid_pages)}"
            )

    pages = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        if selected_pages is not None and page_number not in selected_pages:
            continue
        text = normalize_text(pdf_page.extract_text() or "")
        metadata = {}
        if not text:
            metadata["extraction_status"] = "empty"
            logger.warning(
                "No extractable text on %s page %d; OCR may be required",
                source,
                page_number,
            )
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=metadata,
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in text.split("\n")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

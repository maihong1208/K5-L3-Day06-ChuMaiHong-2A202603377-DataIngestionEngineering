"""Fixed-size and paragraph-aware chunking strategies."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _make_chunk(
    page: DocumentPage,
    text: str,
    strategy: str,
    index: int,
    **extra_metadata: int,
) -> Chunk:
    metadata = dict(page.metadata)
    metadata.update(extra_metadata)
    metadata.update({"chunk_strategy": strategy, "chunk_index": index})
    stable_key = "\0".join(
        (page.document_id, page.source, str(page.page), strategy, str(index), text)
    )
    chunk_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split a page into deterministic, overlapping character windows."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text:
        return []

    chunks = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        text = page.text[start:end].strip()
        if text:
            chunks.append(
                _make_chunk(
                    page,
                    text,
                    "fixed",
                    len(chunks),
                    char_start=start,
                    char_end=end,
                )
            )
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def _split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_size:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split()
            word_piece = ""
            for word in words:
                if len(word) > max_size:
                    if word_piece:
                        pieces.append(word_piece)
                        word_piece = ""
                    pieces.extend(
                        word[offset : offset + max_size]
                        for offset in range(0, len(word), max_size)
                    )
                    continue
                candidate = f"{word_piece} {word}".strip()
                if len(candidate) > max_size:
                    pieces.append(word_piece)
                    word_piece = word
                else:
                    word_piece = candidate
            if word_piece:
                pieces.append(word_piece)
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_size:
            if current:
                pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to a limit; split only oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")
    if not page.text:
        return []

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", page.text)
    ]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    segments: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                segments.append(current)
                current = ""
            segments.extend(_split_long_paragraph(paragraph, max_size))
            continue

        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) > max_size:
            if current:
                segments.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        segments.append(current)

    return [
        _make_chunk(
            page,
            text,
            "paragraph",
            index,
            char_count=len(text),
        )
        for index, text in enumerate(segments)
    ]

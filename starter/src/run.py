"""Run the three local ingestion and retrieval benchmark strategies."""
import json
import logging
import os
import re
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _validate_questions(
    questions: list[dict[str, Any]],
    source_id: str,
    pages_by_number: dict[int, Any],
    section_by_page: dict[str, str],
) -> None:
    if len(questions) < 20:
        raise ValueError(f"Benchmark needs at least 20 questions; found {len(questions)}")
    split_counts = {
        split: sum(question.get("split") == split for question in questions)
        for split in ("dev", "test")
    }
    if split_counts["dev"] < 15 or split_counts["test"] < 5:
        raise ValueError(
            "Benchmark must contain at least 15 dev and 5 test questions; "
            f"found {split_counts}"
        )

    ids = [question.get("id") for question in questions]
    if any(not question_id for question_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Benchmark question IDs must be present and unique")

    filter_questions = [question for question in questions if question.get("filter")]
    test_filters = [
        question for question in filter_questions if question["split"] == "test"
    ]
    if len(filter_questions) < 3 or not test_filters:
        raise ValueError(
            "Benchmark needs at least 3 metadata-filter questions, including a test question"
        )

    for question in questions:
        if question.get("gold_source_id") != source_id:
            raise ValueError(f"{question['id']} references an unknown gold source")
        page_number = question.get("gold_page")
        if page_number not in pages_by_number:
            raise ValueError(
                f"{question['id']} gold page {page_number!r} is not in the selected corpus"
            )
        quote = _normalized(question.get("evidence_quote", ""))
        page_text = _normalized(pages_by_number[page_number].text)
        if not quote or quote not in page_text:
            raise ValueError(
                f"{question['id']} evidence_quote was not found on physical page {page_number}"
            )

        question_filter = question.get("filter")
        if question_filter:
            if (
                question_filter.get("field") != "section"
                or question_filter.get("operator") != "eq"
            ):
                raise ValueError(
                    f"{question['id']} uses an unsupported metadata filter"
                )
            expected_section = section_by_page.get(str(page_number))
            if question_filter.get("value") != expected_section:
                raise ValueError(
                    f"{question['id']} filter does not match the gold page section"
                )
            if question_filter["value"] not in question["question"]:
                raise ValueError(
                    f"{question['id']} must state its section filter in the question"
                )


def _chunk_stats(chunks: list[Any]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "total_characters": sum(lengths),
        "min_chunk_characters": min(lengths) if lengths else 0,
        "median_chunk_characters": (
            statistics.median(lengths) if lengths else 0
        ),
        "max_chunk_characters": max(lengths) if lengths else 0,
    }


def _token_stats(token_counts: list[int]) -> dict[str, Any]:
    return {
        "total_tokens": sum(token_counts),
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "median_chunk_tokens": statistics.median(token_counts) if token_counts else 0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
    }


def _build_manual_audit_template(
    traces: list[dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entries = []
    for trace in traces:
        if trace["split"] != "test":
            continue
        question = questions_by_id[trace["question_id"]]
        entries.append(
            {
                "strategy": trace["strategy"],
                "question_id": trace["question_id"],
                "question": trace["question"],
                "retrieval_seconds": trace["retrieval_seconds"],
                "expected_answer": trace["expected_answer"],
                "gold_source_id": trace["gold_source_id"],
                "gold_page": trace["gold_page"],
                "filter": trace["filter"],
                "top_3": [
                    {
                        "chunk_id": hit["chunk_id"],
                        "text": hit["text"],
                        "metadata": hit["metadata"],
                        "score": hit["score"],
                        "source_correct": None,
                        "page_correct": None,
                        "section_correct": None,
                        "evidence_sufficient": None,
                        "notes": "",
                    }
                    for hit in trace["hits"]
                ],
                "grounding_score_0_to_2": None,
                "grounding_reason": "",
                "filter_compliant": None if question.get("filter") else "not_applicable",
            }
        )
    return {
        "instructions": (
            "Fill labels manually after inspecting the source PDF. Do not treat "
            "automatic evidence matching as a human grounding score."
        ),
        "grounding_scale": {
            "0": "No supporting evidence",
            "1": "Some evidence, but missing needed condition or context",
            "2": "Sufficient evidence to answer and trace to the source",
        },
        "items": entries,
    }


def _build_coherence_audit_template(
    pages: list[Any],
    chunks_by_strategy: dict[str, list[Any]],
) -> dict[str, Any]:
    sampled_pages = sorted(pages, key=lambda page: page.page)[:10]
    items = []
    for page in sampled_pages:
        body_lines = [
            line.strip()
            for line in page.text.splitlines()
            if line.strip()
            and "OWASP Top 10 for LLM Applications v2.0" not in line
            and "genai.owasp.org" not in line
            and not re.fullmatch(r"\d+", line.strip())
        ]
        if not body_lines:
            raise ValueError(f"No body text found for coherence sample page {page.page}")
        anchor = body_lines[0][:120]
        strategy_chunks = {}
        for strategy, chunks in chunks_by_strategy.items():
            candidates = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
                for chunk in chunks
                if chunk.document_id == page.document_id
                and chunk.page == page.page
                and anchor in chunk.text
            ]
            strategy_chunks[strategy] = candidates
        items.append(
            {
                "physical_page": page.page,
                "section": page.metadata["section"],
                "source_position_anchor": anchor,
                "candidate_chunks_by_strategy": strategy_chunks,
                "scores_0_to_2": {"A": None, "B": None, "C": None},
                "reasons": {"A": "", "B": "", "C": ""},
            }
        )
    return {
        "sampling_policy": (
            "The 10 positions are the first 10 included physical pages in "
            "manifest order, selected before reviewing retrieval results."
        ),
        "score_scale": {
            "0": "Cut mid-idea or missing context makes it unclear",
            "1": "Partly understandable but missing context",
            "2": "Self-contained and preserves the complete idea",
        },
        "items": items,
    }


def main() -> None:
    """Ingest the manifest-selected PDF pages, build A/B/C, and benchmark retrieval."""
    starter_dir = Path(__file__).resolve().parents[1]
    load_dotenv(starter_dir / ".env")

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest_path = starter_dir / "data" / "manifest.json"
    benchmark_path = starter_dir / "benchmark" / "questions.json"
    manifest_data = _read_json(manifest_path)
    benchmark_data = _read_json(benchmark_path)
    documents = manifest_data.get("documents", [])
    if len(documents) != 1:
        raise ValueError("This runner expects exactly one manifest document")
    manifest = documents[0]
    source_id = manifest["source_id"]
    pdf_path = starter_dir / "data" / manifest["local_file"]
    source = manifest["source_path_or_url"]

    page_numbers = {
        item["page"] for item in manifest.get("included_physical_pages", [])
    }
    if not page_numbers:
        raise ValueError("Manifest does not list any included physical pages")

    pdf_extraction_start = time.perf_counter()
    pages = load_pdf(pdf_path, source_id, source, page_numbers=page_numbers)
    pdf_extraction_seconds = time.perf_counter() - pdf_extraction_start
    section_by_page = manifest.get("section_by_page", {})
    for page in pages:
        section = section_by_page.get(str(page.page))
        if not section:
            raise ValueError(f"Manifest is missing section metadata for page {page.page}")
        page.metadata["section"] = section

    pages_by_number = {page.page: page for page in pages}
    useful_pages = [page for page in pages if len(page.text) >= 150]
    if len(useful_pages) < 30:
        raise ValueError(
            f"Corpus has {len(useful_pages)} useful selected pages; at least 30 are required"
        )

    questions = benchmark_data.get("questions", [])
    _validate_questions(questions, source_id, pages_by_number, section_by_page)
    questions_by_id = {question["id"]: question for question in questions}

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        raise ValueError("EMBEDDING_MODEL must name a real local embedding model")
    device = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"
    embedder = MyEmbedder(model_name=model_name, device=device)

    vector_path = Path(os.getenv("VECTOR_STORE_PATH", "results/chroma"))
    if not vector_path.is_absolute():
        vector_path = starter_dir / vector_path
    store = ChromaVectorStore(vector_path, embedder.dimensions)
    store.create_schema()

    chunking_start = time.perf_counter()
    chunks_a = [
        chunk
        for page in pages
        for chunk in fixed_size(page, size=500, overlap=50)
    ]
    chunking_seconds_a = time.perf_counter() - chunking_start
    chunking_start = time.perf_counter()
    chunks_b = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=800)
    ]
    chunking_seconds_b = time.perf_counter() - chunking_start
    chunk_sets = {"A": chunks_a, "B": chunks_b}
    chunk_sets["C"] = chunk_sets["B"]
    chunking_seconds = {"A": chunking_seconds_a, "B": chunking_seconds_b, "C": 0.0}

    strategy_stats: dict[str, dict[str, Any]] = {}
    vectors_by_strategy: dict[str, list[list[float]]] = {}
    token_counts_by_strategy: dict[str, list[int]] = {}
    vectors_by_strategy["A"] = []
    vectors_by_strategy["B"] = []
    for strategy in ("A", "B"):
        chunks = chunk_sets[strategy]
        tokenization_start = time.perf_counter()
        token_counts = embedder.count_tokens([chunk.text for chunk in chunks])
        tokenization_seconds = time.perf_counter() - tokenization_start
        token_counts_by_strategy[strategy] = token_counts

        embedding_start = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        embedding_seconds = time.perf_counter() - embedding_start
        vectors_by_strategy[strategy] = vectors

        store_start = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        store_seconds = time.perf_counter() - store_start
        strategy_stats[strategy] = {
            **_chunk_stats(chunks),
            **_token_stats(token_counts),
            "chunking_seconds": chunking_seconds[strategy],
            "tokenization_seconds": tokenization_seconds,
            "embedding_seconds": embedding_seconds,
            "store_seconds": store_seconds,
            "ingestion_seconds": (
                chunking_seconds[strategy]
                + tokenization_seconds
                + embedding_seconds
                + store_seconds
            ),
            "embedding_reused": False,
        }

    strategy = "C"
    chunks = chunk_sets[strategy]
    token_counts_by_strategy[strategy] = token_counts_by_strategy["B"]
    store_start = time.perf_counter()
    store.upsert(strategy, chunks, vectors_by_strategy["B"])
    store_seconds = time.perf_counter() - store_start
    strategy_stats[strategy] = {
        **_chunk_stats(chunks),
        **_token_stats(token_counts_by_strategy[strategy]),
        "chunking_seconds": chunking_seconds[strategy],
        "tokenization_seconds": 0.0,
        "embedding_seconds": 0.0,
        "store_seconds": store_seconds,
        "ingestion_seconds": store_seconds,
        "embedding_reused": True,
        "embedding_source_strategy": "B",
    }

    logger.info(
        "Indexed %d pages and %d questions with %s (%d dimensions)",
        len(pages),
        len(questions),
        embedder.model_name,
        embedder.dimensions,
    )

    query_vectors: dict[str, list[float]] = {}
    query_embedding_start = time.perf_counter()
    for question in questions:
        query_vectors[question["id"]] = embedder.embed_query(question["question"])
    query_embedding_seconds = time.perf_counter() - query_embedding_start

    traces: list[dict[str, Any]] = []
    for strategy in ("A", "B", "C"):
        for question in questions:
            if question["split"] != "dev":
                continue
            filter_data = question.get("filter") if strategy == "C" else None
            filters = (
                {filter_data["field"]: filter_data["value"]}
                if filter_data
                else None
            )
            retrieval_start = time.perf_counter()
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=3,
                filters=filters,
            )
            retrieval_seconds = time.perf_counter() - retrieval_start
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            trace["filter_applied"] = filters
            trace["retrieval_seconds"] = retrieval_seconds
            traces.append(trace)

    dev_summary = summarize(traces)
    selected_strategy = max(
        ("A", "B", "C"),
        key=lambda name: (
            dev_summary[name]["dev"]["evidence_hit_at_3_rate"],
            dev_summary[name]["dev"]["hit_at_1_rate"],
            -("A", "B", "C").index(name),
        ),
    )
    logger.info(
        "Selected %s on dev using Hit@3, then Hit@1; now evaluating held-out test",
        selected_strategy,
    )

    for strategy in ("A", "B", "C"):
        for question in questions:
            if question["split"] != "test":
                continue
            filter_data = question.get("filter") if strategy == "C" else None
            filters = (
                {filter_data["field"]: filter_data["value"]}
                if filter_data
                else None
            )
            retrieval_start = time.perf_counter()
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=3,
                filters=filters,
            )
            retrieval_seconds = time.perf_counter() - retrieval_start
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            trace["filter_applied"] = filters
            trace["retrieval_seconds"] = retrieval_seconds
            traces.append(trace)

    summary = summarize(traces)
    result_dir = starter_dir / "results"
    _write_json(result_dir / "traces.json", traces)
    _write_json(
        result_dir / "manual_audit_template.json",
        _build_manual_audit_template(traces, questions_by_id),
    )
    _write_json(
        result_dir / "coherence_audit_template.json",
        _build_coherence_audit_template(pages, chunk_sets),
    )

    run_summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "source_file": str(pdf_path.relative_to(starter_dir)),
        "selected_physical_pages": sorted(page_numbers),
        "selected_page_count": len(pages),
        "useful_content_page_count": len(useful_pages),
        "normalized_character_count": sum(len(page.text) for page in pages),
        "embedding": {
            "provider": "local_sentence_transformers",
            "model": embedder.model_name,
            "model_revision": embedder.revision,
            "device": device,
            "dimensions": embedder.dimensions,
            "library_version": embedder.library_version,
            "query_embedding_seconds_total": query_embedding_seconds,
            "api_cost_usd": 0,
            "api_key_required": False,
        },
        "vector_store": {
            "name": "Chroma PersistentClient",
            "path": str(vector_path),
            "library_version": version("chromadb"),
            "metric": "cosine",
            "search_type": "approximate HNSW",
            "persistent": True,
        },
        "pdf_extraction_seconds": pdf_extraction_seconds,
        "strategies": {
            "A": {
                "description": "500-character fixed chunks, 50-character overlap, no query filter",
                **strategy_stats["A"],
            },
            "B": {
                "description": "Paragraph-aware chunks up to 800 characters, no query filter",
                **strategy_stats["B"],
            },
            "C": {
                "description": "Same chunks as B, section equality filter when question supplies one",
                **strategy_stats["C"],
            },
        },
        "selection_policy": (
            "Choose on dev by highest Evidence Hit@3 rate, then highest Hit@1 rate; "
            "ties prefer A, then B, then C. Test was run only after this choice was fixed."
        ),
        "selected_strategy_on_dev": selected_strategy,
        "retrieval_metrics": summary,
        "manual_assessment_note": (
            "Chunk coherence, metadata correctness/usefulness, and grounding require "
            "manual labels; no manual score is fabricated by this pipeline."
        ),
    }
    _write_json(result_dir / "run_summary.json", run_summary)

    print(json.dumps(
        {
            "selected_strategy_on_dev": selected_strategy,
            "metrics": summary,
            "outputs": [
                "results/run_summary.json",
                "results/traces.json",
                "results/manual_audit_template.json",
                "results/coherence_audit_template.json",
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

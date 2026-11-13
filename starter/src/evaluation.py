"""Score traceable evidence retrieval and aggregate retrieval metrics."""
import re
import unicodedata
from typing import Any


def _normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _gold_items(question: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives = question.get("gold_evidence")
    if alternatives:
        return alternatives
    return [
        {
            "source_id": question["gold_source_id"],
            "page": question["gold_page"],
            "evidence_quote": question["evidence_quote"],
        }
    ]


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that a result is from a gold page and contains quoted evidence.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    gold_items = _gold_items(question)
    scored_hits = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        text = _normalize_for_match(hit.get("text") or "")
        matched_gold = None
        for gold in gold_items:
            source_match = metadata.get("document_id") == gold["source_id"]
            try:
                page_match = int(metadata.get("page")) == int(gold["page"])
            except (TypeError, ValueError):
                page_match = False
            quote = _normalize_for_match(gold["evidence_quote"])
            if source_match and page_match and quote and quote in text:
                matched_gold = gold
                break

        scored_hits.append(
            {
                **hit,
                "is_evidence_hit": matched_gold is not None,
                "matched_gold_page": (
                    matched_gold["page"] if matched_gold is not None else None
                ),
            }
        )

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "strategy": "",
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "filter": question.get("filter"),
        "hit_at_1": bool(scored_hits and scored_hits[0]["is_evidence_hit"]),
        "evidence_hit_at_3": any(
            hit["is_evidence_hit"] for hit in scored_hits[:3]
        ),
        "hits": scored_hits[:3],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval counts and rates by strategy and split."""
    summary: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trace in traces:
        key = (trace["strategy"], trace["split"])
        groups.setdefault(key, []).append(trace)

    for (strategy, split), rows in sorted(groups.items()):
        total = len(rows)
        hit_at_1 = sum(bool(row["hit_at_1"]) for row in rows)
        hit_at_3 = sum(bool(row["evidence_hit_at_3"]) for row in rows)
        summary.setdefault(strategy, {})[split] = {
            "questions": total,
            "hit_at_1_count": hit_at_1,
            "hit_at_1_rate": hit_at_1 / total if total else 0.0,
            "evidence_hit_at_3_count": hit_at_3,
            "evidence_hit_at_3_rate": hit_at_3 / total if total else 0.0,
            "mean_retrieval_seconds": (
                sum(row.get("retrieval_seconds", 0.0) for row in rows) / total
                if total
                else 0.0
            ),
        }
    return summary

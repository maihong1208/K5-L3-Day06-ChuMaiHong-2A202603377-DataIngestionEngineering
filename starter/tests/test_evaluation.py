import unittest

from src.evaluation import evaluate_query, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.question = {
            "id": "q1",
            "split": "dev",
            "question": "What is the policy?",
            "expected_answer": "Be clear.",
            "gold_source_id": "doc-1",
            "gold_page": 2,
            "evidence_quote": "Use a clear policy.",
        }

    def test_evaluate_requires_correct_source_page_and_evidence(self):
        hits = [
            {
                "chunk_id": "wrong-page",
                "text": "Use a clear policy.",
                "metadata": {"document_id": "doc-1", "page": 3},
            },
            {
                "chunk_id": "gold",
                "text": "Use a clear\npolicy.",
                "metadata": {"document_id": "doc-1", "page": 2},
            },
            {
                "chunk_id": "wrong-source",
                "text": "Use a clear policy.",
                "metadata": {"document_id": "other", "page": 2},
            },
        ]

        result = evaluate_query(self.question, hits)

        self.assertFalse(result["hit_at_1"])
        self.assertTrue(result["evidence_hit_at_3"])
        self.assertTrue(result["hits"][1]["is_evidence_hit"])

    def test_summarize_reports_counts_and_rates_by_split(self):
        traces = [
            {
                "strategy": "A",
                "split": "dev",
                "hit_at_1": True,
                "evidence_hit_at_3": True,
            },
            {
                "strategy": "A",
                "split": "dev",
                "hit_at_1": False,
                "evidence_hit_at_3": True,
            },
        ]

        result = summarize(traces)

        self.assertEqual(result["A"]["dev"]["questions"], 2)
        self.assertEqual(result["A"]["dev"]["hit_at_1_count"], 1)
        self.assertEqual(result["A"]["dev"]["evidence_hit_at_3_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

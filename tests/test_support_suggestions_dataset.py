"""
tests/test_support_suggestions_dataset.py

Unit tests for the Support Suggestion Dataset Evaluation Runner.
Verifies the deterministic evaluation logic, pass/fail rules, cross-document matching,
safety disclaimers, metric aggregation, and report generation using mock inputs.
NO LIVE GEMINI OR NETWORK CALLS ARE MADE BY THIS TEST SUITE.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path

from scripts.test_support_suggestions_dataset import (
    evaluate_single_case,
    compute_summary_metrics,
    save_reports
)


class TestSupportSuggestionsDatasetEvaluator(unittest.TestCase):

    def setUp(self):
        # Sample test case definitions matching support_suggestion_test_cases.json structure
        self.grounded_single_doc_case = {
            "id": "SS-01",
            "document": "Vantor",
            "machine": "WP-400",
            "customer_query": "Our WP-400 panel saw just threw an E-206. What should we check?",
            "expected_category": "Technical Troubleshooting",
            "expected_grounded": True,
            "expected_source_refs": ["TS-WP400-3.1", "RP-WP400-4.1"]
        }

        self.cross_doc_case = {
            "id": "SS-09",
            "document": "Both",
            "machine": "GC-120 / GL-300",
            "customer_query": "Both GC-120 and GL-300 are throwing vacuum alarms.",
            "expected_category": "Technical Troubleshooting",
            "expected_grounded": True,
            "expected_source_refs": ["TS-GC120-3.4", "TS-GL300-3.4"]
        }

        self.no_context_case = {
            "id": "SS-11",
            "document": "Kestrel",
            "machine": "WR-700",
            "customer_query": "What's the part number for the spindle bearing on our WR-700?",
            "expected_category": "Maintenance & Parts",
            "expected_grounded": False,
            "expected_source_refs": []
        }

        self.sample_latencies = {
            "retrieval_ms": 120.0,
            "generation_ms": 1400.0,
            "total_ms": 1520.0
        }

    # --------------------------------------------------------------------------
    # 1. Grounded Cases: PASS / FAIL Scenarios
    # --------------------------------------------------------------------------
    def test_grounded_pass(self):
        """A valid grounded response with matching source reference must PASS."""
        retrieved_chunks = [
            {"metadata": {"document_reference": "TS-WP400-3.1", "page": 6}},
            {"metadata": {"document_reference": "RP-WP400-4.1", "page": 8}}
        ]
        llm_result = {
            "suggestion": "Stop at once. Inspect the saw blade and replace per section 4.1. Retighten arbor nut to 60 Nm.",
            "confidence": 0.95,
            "sources": [{"document_reference": "TS-WP400-3.1", "page": 6}],
            "is_grounded": True
        }

        res = evaluate_single_case(
            self.grounded_single_doc_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertTrue(res["passed"])
        self.assertIsNone(res["failure_reason"])
        self.assertTrue(res["actual_grounded"])
        self.assertIn("TS-WP400-3.1", res["retrieved_source_refs"])

    def test_grounded_fail_when_is_grounded_is_false(self):
        """Grounded case must FAIL if actual is_grounded is False."""
        retrieved_chunks = [
            {"metadata": {"document_reference": "TS-WP400-3.1", "page": 6}}
        ]
        llm_result = {
            "suggestion": "Some response.",
            "confidence": 0.5,
            "sources": [],
            "is_grounded": False
        }

        res = evaluate_single_case(
            self.grounded_single_doc_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertFalse(res["passed"])
        self.assertIn("Expected is_grounded=True", res["failure_reason"])

    def test_grounded_fail_when_expected_source_is_missing(self):
        """Grounded case must FAIL if none of the expected source refs are present."""
        retrieved_chunks = [
            {"metadata": {"document_reference": "UNRELATED-REF-9.9", "page": 1}}
        ]
        llm_result = {
            "suggestion": "Check the machine blade immediately.",
            "confidence": 0.9,
            "sources": [{"document_reference": "UNRELATED-REF-9.9"}],
            "is_grounded": True
        }

        res = evaluate_single_case(
            self.grounded_single_doc_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertFalse(res["passed"])
        self.assertIn("None of the expected source references were retrieved/returned", res["failure_reason"])

    def test_grounded_fail_on_ungrounded_disclaimer(self):
        """Grounded case must FAIL if model returns an ungrounded disclaimer."""
        retrieved_chunks = [
            {"metadata": {"document_reference": "TS-WP400-3.1", "page": 6}}
        ]
        llm_result = {
            "suggestion": "The available documentation does not provide sufficient information for this error.",
            "confidence": 0.8,
            "sources": [{"document_reference": "TS-WP400-3.1"}],
            "is_grounded": True
        }

        res = evaluate_single_case(
            self.grounded_single_doc_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertFalse(res["passed"])
        self.assertIn("Suggestion returned an ungrounded disclaimer", res["failure_reason"])

    # --------------------------------------------------------------------------
    # 2. No-Context Cases: PASS / FAIL Scenarios
    # --------------------------------------------------------------------------
    def test_no_context_pass(self):
        """No-context case must PASS when is_grounded is False and disclaimer is given."""
        retrieved_chunks = [
            {"metadata": {"document_reference": "WR7-OTHER-1.1", "page": 2}}
        ]
        llm_result = {
            "suggestion": "The available documentation does not provide sufficient information regarding the spindle bearing part number.",
            "confidence": 1.0,
            "sources": [],
            "is_grounded": False
        }

        res = evaluate_single_case(
            self.no_context_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertTrue(res["passed"])
        self.assertIsNone(res["failure_reason"])
        self.assertFalse(res["actual_grounded"])

    def test_no_context_fail_when_model_fabricates_grounded_answer(self):
        """No-context case must FAIL if model fabricates an answer with is_grounded=True."""
        retrieved_chunks = []
        llm_result = {
            "suggestion": "The spindle bearing part number is SC9-BR-6207 and should be changed annually.",
            "confidence": 0.95,
            "sources": [],
            "is_grounded": True
        }

        res = evaluate_single_case(
            self.no_context_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertFalse(res["passed"])
        self.assertIn("Expected is_grounded=False, but system returned is_grounded=True", res["failure_reason"])

    def test_no_context_fail_without_insufficient_documentation_phrase(self):
        """No-context case must FAIL if response does not state insufficient documentation."""
        retrieved_chunks = []
        llm_result = {
            "suggestion": "Everything looks okay with your machine setup.",
            "confidence": 0.4,
            "sources": [],
            "is_grounded": False
        }

        res = evaluate_single_case(
            self.no_context_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertFalse(res["passed"])
        self.assertIn("No-context response failed to clearly state insufficient documentation", res["failure_reason"])

    # --------------------------------------------------------------------------
    # 3. Cross-Document Source Matching Scenarios
    # --------------------------------------------------------------------------
    def test_cross_document_source_matching_pass(self):
        """Cross-document case must PASS when sources from both documents are retrieved."""
        retrieved_chunks = [
            {"metadata": {"document_reference": "TS-GC120-3.4", "page": 7, "source": "Vantor.pdf"}},
            {"metadata": {"document_reference": "TS-GL300-3.4", "page": 9, "source": "Kestrel.docx"}}
        ]
        llm_result = {
            "suggestion": "For GC-120 check seal GC1-VB-02. For GL-300 check sheet seating and filter.",
            "confidence": 0.95,
            "sources": [
                {"document_reference": "TS-GC120-3.4"},
                {"document_reference": "TS-GL300-3.4"}
            ],
            "is_grounded": True
        }

        res = evaluate_single_case(
            self.cross_doc_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertTrue(res["passed"])
        self.assertIsNone(res["failure_reason"])

    def test_cross_document_source_matching_fail_when_incomplete(self):
        """Cross-document case must FAIL if one required document's source is missing."""
        retrieved_chunks = [
            {"metadata": {"document_reference": "TS-GC120-3.4", "page": 7, "source": "Vantor.pdf"}}
        ]
        llm_result = {
            "suggestion": "Check the vacuum seal for GC-120 only.",
            "confidence": 0.9,
            "sources": [{"document_reference": "TS-GC120-3.4"}],
            "is_grounded": True
        }

        res = evaluate_single_case(
            self.cross_doc_case,
            retrieved_chunks,
            llm_result,
            self.sample_latencies
        )

        self.assertFalse(res["passed"])
        self.assertIn("Cross-document source missing", res["failure_reason"])

    # --------------------------------------------------------------------------
    # 4. Metric Aggregation & Report Generation
    # --------------------------------------------------------------------------
    def test_compute_summary_metrics(self):
        """Summary calculation must compute rates, averages, and 429 counts accurately."""
        sample_eval_cases = [
            # 1. Grounded pass
            {
                "passed": True, "expected_grounded": True, "document": "Vantor",
                "expected_source_refs": ["REF-1"], "retrieved_source_refs": ["REF-1"],
                "returned_source_refs": ["REF-1"], "generation_ms": 1000.0, "total_ms": 1200.0, "is_429": False
            },
            # 2. Grounded fail
            {
                "passed": False, "expected_grounded": True, "document": "Vantor",
                "expected_source_refs": ["REF-2"], "retrieved_source_refs": ["WRONG"],
                "returned_source_refs": [], "generation_ms": 1200.0, "total_ms": 1400.0, "is_429": False
            },
            # 3. Cross-doc pass
            {
                "passed": True, "expected_grounded": True, "document": "Both",
                "expected_source_refs": ["REF-A", "REF-B"], "retrieved_source_refs": ["REF-A", "REF-B"],
                "returned_source_refs": ["REF-A", "REF-B"], "generation_ms": 1500.0, "total_ms": 1800.0, "is_429": False
            },
            # 4. No-context pass
            {
                "passed": True, "expected_grounded": False, "document": "Kestrel",
                "expected_source_refs": [], "retrieved_source_refs": [],
                "returned_source_refs": [], "generation_ms": 800.0, "total_ms": 900.0, "is_429": False
            },
            # 5. Failed with 429
            {
                "passed": False, "expected_grounded": True, "document": "Kestrel",
                "expected_source_refs": ["REF-3"], "retrieved_source_refs": [],
                "returned_source_refs": [], "generation_ms": 0.0, "total_ms": 100.0, "is_429": True
            }
        ]

        summary = compute_summary_metrics(sample_eval_cases)

        self.assertEqual(summary["total_cases"], 5)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["grounded_cases_passed"], "1/3")
        self.assertEqual(summary["cross_doc_cases_passed"], "1/1")
        self.assertEqual(summary["no_context_cases_passed"], "1/1")
        self.assertEqual(summary["no_context_safety_rate"], 100.0)
        self.assertEqual(summary["num_429_errors"], 1)
        self.assertEqual(summary["avg_generation_latency_ms"], 900.0)
        self.assertEqual(summary["avg_total_latency_ms"], 1080.0)

    def test_save_reports(self):
        """Report generator must write both JSON and Markdown files correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            md_path = Path(tmpdir) / "results.md"

            cases = [
                {
                    "id": "SS-01", "machine": "WP-400", "document": "Vantor",
                    "customer_query": "Test query", "expected_grounded": True,
                    "actual_grounded": True, "expected_source_refs": ["REF-1"],
                    "retrieved_source_refs": ["REF-1"], "returned_source_refs": ["REF-1"],
                    "suggestion": "Step 1, step 2.", "confidence": 0.95,
                    "generation_ms": 1100.0, "total_ms": 1300.0, "passed": True,
                    "failure_reason": None, "is_429": False
                }
            ]
            summary = compute_summary_metrics(cases)

            save_reports(cases, summary, json_path, md_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
            self.assertIn("summary", saved_json)
            self.assertIn("cases", saved_json)
            self.assertEqual(saved_json["summary"]["total_cases"], 1)

            md_content = md_path.read_text(encoding="utf-8")
            self.assertIn("# Support Suggestion Dataset Evaluation Results", md_content)
            self.assertIn("SS-01", md_content)
            self.assertIn("✅ PASS", md_content)


if __name__ == "__main__":
    unittest.main()

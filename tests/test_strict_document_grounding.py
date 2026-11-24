"""
tests/test_strict_document_grounding.py

Comprehensive Verification Suite for Strict Document-Grounded Response Generation.
Verifies:
1. Fully supported questions (grounded in context, 4 distinct suggestions, valid citations).
2. Partially supported questions (answers supported portions, identifies unverified details, does not force 4).
3. Questions completely outside the knowledge base (suppresses suggestions, returns standard disclaimer).
4. Irrelevant document retrieval (low similarity score suppresses technical suggestions).
5. Unsupported technical parameters (rejects hallucinated pressures, voltages, velocities).
6. Hallucinated citations (rejects fictitious document IDs not in ChromaDB metadata).
7. Multiple-document evidence (multiple validated citations on one suggestion).
8. Positive conversations with no technical issue (has_issue=False, empty suggestions).
9. Never falling back to pretrained knowledge or generic substitutes when documentation is insufficient.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.llm_suggestions import (
    generate_support_suggestion,
    _evaluate_context_relevance,
    _validate_and_sanitize_suggestions,
    INSUFFICIENT_INFO_RESPONSE,
    EVIDENCE_STATUS_SUFFICIENT,
    EVIDENCE_STATUS_PARTIAL,
    EVIDENCE_STATUS_INSUFFICIENT
)


class TestStrictDocumentGrounding(unittest.TestCase):

    def setUp(self):
        import src.llm_suggestions
        src.llm_suggestions._shared_client = None
        src.llm_suggestions._shared_client_key = None
        self.doc_wp400_ts = {
            "text": (
                "Section 3.1 WP-400 error codes\n"
                "Code: E-102 | Meaning: Dust extraction fault | "
                "Likely cause: Filter cartridge blocked (pressure drop above 1200 Pa) or fan stopped.\n"
                "Action: Clean or replace filter cartridge WP4-FL-DE1. Check extraction fan breaker Q4 in electrical cabinet."
            ),
            "metadata": {
                "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                "page": 6,
                "document_reference": "TS-WP400-3.1",
                "title": "Section 3.1 WP-400 error codes",
                "section": "Section 3.1 WP-400 error codes",
                "machine": "WP-400"
            }
        }

        self.doc_wp400_um = {
            "text": (
                "Section 1.1.3 Dust extraction\n"
                "Minimum air velocity at the saw hood must be maintained at 20 m/s during all cutting operations. "
                "Verify ductwork is free of foreign material and ensure interlock switch S3 is closed."
            ),
            "metadata": {
                "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                "page": 2,
                "document_reference": "UM-WP400-1.1.3",
                "title": "Section 1.1.3 Dust extraction",
                "section": "Section 1.1.3 Dust extraction",
                "machine": "WP-400"
            }
        }

    def tearDown(self):
        import src.llm_suggestions
        src.llm_suggestions._shared_client = None
        src.llm_suggestions._shared_client_key = None

    def _create_mock_client(self, payload: dict):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(payload)
        mock_client.models.generate_content.return_value = mock_resp
        return mock_client

    # --------------------------------------------------------------------------
    # 1. Fully Supported Questions
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_fully_supported_question(self, mock_client_cls):
        """Verifies that fully supported questions produce 4 grounded suggestions with citations."""
        mock_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "WP-400 Dust Extraction Fault (E-102)",
            "suggestions": [
                "Clean or replace filter cartridge WP4-FL-DE1 per TS-WP400-3.1.",
                "Check extraction fan breaker Q4 in the main electrical cabinet.",
                "Verify saw hood minimum air velocity is at least 20 m/s.",
                "Ensure interlock switch S3 is fully closed and clear of saw debris."
            ],
            "suggestion_sources": [
                ["TS-WP400-3.1"],
                ["TS-WP400-3.1"],
                ["UM-WP400-1.1.3"],
                ["UM-WP400-1.1.3"]
            ],
            "unverified_details": "",
            "safety": "Isolate main electrical supply before opening the fan housing.",
            "confidence": 0.96,
            "is_grounded": True
        }
        mock_client = self._create_mock_client(mock_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="The WP-400 saw triggered alarm E-102 and the fan stopped.",
            retrieved_context=[self.doc_wp400_ts, self.doc_wp400_um],
            api_key="test-key"
        )

        self.assertTrue(res["has_issue"])
        self.assertEqual(res["evidence_status"], EVIDENCE_STATUS_SUFFICIENT)
        self.assertEqual(len(res["suggestions"]), 4)
        self.assertTrue(res["is_grounded"])
        self.assertGreaterEqual(res["confidence"], 0.9)
        self.assertIn("WP4-FL-DE1", res["suggestions"][0])
        self.assertIn("20 m/s", res["suggestions"][2])
        self.assertEqual(res["suggestion_sources"][0][0]["document_reference"], "TS-WP400-3.1")
        self.assertEqual(res["suggestion_sources"][2][0]["document_reference"], "UM-WP400-1.1.3")

    # --------------------------------------------------------------------------
    # 2. Partially Supported Questions
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_partially_supported_question_does_not_force_four_and_identifies_gaps(self, mock_client_cls):
        """Verifies partial documentation answers only supported portion, does not force 4 suggestions."""
        mock_payload = {
            "has_issue": True,
            "evidence_status": "partial",
            "likely_issue": "WP-400 Dust Extraction Fault E-102 (Hydraulic fluid type unverified)",
            "suggestions": [
                "Clean or replace filter cartridge WP4-FL-DE1 per Section 3.1.",
                "Check extraction fan breaker Q4 in electrical cabinet."
            ],
            "suggestion_sources": [
                ["TS-WP400-3.1"],
                ["TS-WP400-3.1"]
            ],
            "unverified_details": "Hydraulic oil specification and tilt reservoir volume are not documented in the available manual.",
            "safety": "Isolate power before inspecting electrical enclosure.",
            "confidence": 0.88,
            "is_grounded": True
        }
        mock_client = self._create_mock_client(mock_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="WP-400 shows E-102. What is the filter code and what type of hydraulic oil does the saw tilt use?",
            retrieved_context=[self.doc_wp400_ts],
            api_key="test-key"
        )

        self.assertTrue(res["has_issue"])
        self.assertEqual(res["evidence_status"], EVIDENCE_STATUS_PARTIAL)
        # MUST NOT force 4 suggestions when evidence is partial!
        self.assertEqual(len(res["suggestions"]), 2)
        self.assertIn("Hydraulic oil specification", res["unverified_details"])
        self.assertIn("Unverified Details", res["suggestion"])
        self.assertTrue(res["is_grounded"])

    # --------------------------------------------------------------------------
    # 3. Questions Completely Outside the Knowledge Base
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_question_outside_knowledge_base_suppresses_suggestions(self, mock_client_cls):
        """Verifies questions completely outside the knowledge base return no suggestions and clear disclaimer."""
        mock_payload = {
            "has_issue": True,
            "evidence_status": "insufficient",
            "likely_issue": "Query outside scope of documentation",
            "suggestions": [],
            "suggestion_sources": [],
            "unverified_details": "No documentation available for consumer vehicle repair.",
            "safety": "",
            "confidence": 0.0,
            "is_grounded": False
        }
        mock_client = self._create_mock_client(mock_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="How do I change the transmission fluid on a Honda Civic?",
            retrieved_context=[self.doc_wp400_ts],
            api_key="test-key"
        )

        self.assertTrue(res["has_issue"])
        self.assertEqual(res["evidence_status"], EVIDENCE_STATUS_INSUFFICIENT)
        self.assertEqual(len(res["suggestions"]), 0)
        self.assertEqual(res["suggestion_sources"], [])
        self.assertFalse(res["is_grounded"])
        self.assertEqual(res["confidence"], 0.0)
        self.assertIn("I couldn't find sufficient information about this issue", res["suggestion"])

    # --------------------------------------------------------------------------
    # 4. Irrelevant Document Retrieval (Low Similarity)
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_irrelevant_retrieval_similarity_threshold(self, mock_client_cls):
        """Verifies that retrieval with low similarity score suppresses technical suggestions without calling LLM."""
        irrelevant_chunks = [
            {
                "text": "General maintenance of unrelated equipment.",
                "metadata": {"document_reference": "DOC-UNRELATED"},
                "similarity_score": 0.22,  # below 0.35 threshold
                "distance": 1.56
            }
        ]

        res = generate_support_suggestion(
            conversation_text="How do I configure optical focal length on laser model LX-90?",
            retrieved_context=irrelevant_chunks,
            api_key="test-key"
        )

        # LLM must NOT be called
        mock_client_cls.assert_not_called()
        self.assertEqual(res["evidence_status"], EVIDENCE_STATUS_INSUFFICIENT)
        self.assertEqual(len(res["suggestions"]), 0)
        self.assertFalse(res["is_grounded"])
        self.assertIn(INSUFFICIENT_INFO_RESPONSE, res["suggestion"])

    # --------------------------------------------------------------------------
    # 5. Unsupported Technical Parameters Rejected by Guardrail
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_unsupported_parameters_rejected_by_guardrail(self, mock_client_cls):
        """Verifies that hallucinated technical specifications (e.g. 50 bar, 85 m/s) are rejected."""
        mock_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "WP-400 E-102 Fault",
            "suggestions": [
                "Clean or replace filter cartridge WP4-FL-DE1.",
                "Set pneumatic line pressure to 55.0 bar.",  # Hallucinated: context only supports 1200 Pa, not 55 bar
                "Ensure minimum air velocity is at least 85 m/s.",  # Hallucinated: context says 20 m/s
                "Check extraction fan breaker Q4 in electrical cabinet."
            ],
            "suggestion_sources": [
                ["TS-WP400-3.1"],
                ["TS-WP400-3.1"],
                ["UM-WP400-1.1.3"],
                ["TS-WP400-3.1"]
            ],
            "safety": "",
            "confidence": 0.9,
            "is_grounded": True
        }
        mock_client = self._create_mock_client(mock_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="The saw stopped with E-102 alarm.",
            retrieved_context=[self.doc_wp400_ts, self.doc_wp400_um],
            api_key="test-key"
        )

        # The 2 suggestions with ungrounded parameters (55.0 bar, 85 m/s) must be rejected
        all_suggs_text = " ".join(res["suggestions"])
        self.assertNotIn("55.0 bar", all_suggs_text)
        self.assertNotIn("85 m/s", all_suggs_text)
        # Verified grounded suggestions must remain
        self.assertTrue(any("WP4-FL-DE1" in s for s in res["suggestions"]))
        self.assertTrue(any("breaker Q4" in s for s in res["suggestions"]))

    # --------------------------------------------------------------------------
    # 6. Hallucinated Citations Stripped
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_hallucinated_citations_stripped(self, mock_client_cls):
        """Verifies that document references not present in ChromaDB metadata are stripped."""
        mock_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "WP-400 E-102 Fault",
            "suggestions": [
                "Clean or replace filter cartridge WP4-FL-DE1.",
                "Check extraction fan breaker Q4.",
                "Ensure saw hood minimum air velocity is 20 m/s.",
                "Check physical safety condition."
            ],
            "suggestion_sources": [
                ["TS-WP400-3.1", "DOC-GHOST-999"],  # DOC-GHOST-999 is fabricated!
                ["FABRICATED-MANUAL-X"],             # FABRICATED-MANUAL-X is fabricated!
                ["UM-WP400-1.1.3"],
                []
            ],
            "safety": "",
            "confidence": 0.95,
            "is_grounded": True
        }
        mock_client = self._create_mock_client(mock_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="Alarm E-102 extraction fan stopped.",
            retrieved_context=[self.doc_wp400_ts, self.doc_wp400_um],
            api_key="test-key"
        )

        sugg_sources = res["suggestion_sources"]
        # Suggestion 1 should keep TS-WP400-3.1 and strip DOC-GHOST-999
        self.assertEqual(len(sugg_sources[0]), 1)
        self.assertEqual(sugg_sources[0][0]["document_reference"], "TS-WP400-3.1")

        # Suggestion 2 had only FABRICATED-MANUAL-X, but mentions 'fan breaker Q4' from TS-WP400-3.1
        for c in sugg_sources[1]:
            self.assertNotEqual(c["document_reference"], "FABRICATED-MANUAL-X")

        # Suggestion 3 should keep UM-WP400-1.1.3
        self.assertEqual(sugg_sources[2][0]["document_reference"], "UM-WP400-1.1.3")

    # --------------------------------------------------------------------------
    # 7. Multiple-Document Evidence
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_multiple_document_evidence_supported(self, mock_client_cls):
        """Verifies suggestions referencing multiple distinct retrieved documents."""
        mock_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "WP-400 Extraction System Failure",
            "suggestions": [
                "Service filter cartridge WP4-FL-DE1 and verify saw hood air velocity is 20 m/s.",
                "Check extraction fan breaker Q4.",
                "Inspect ductwork for blockages.",
                "Escalate to technical support."
            ],
            "suggestion_sources": [
                ["TS-WP400-3.1", "UM-WP400-1.1.3"],
                ["TS-WP400-3.1"],
                ["UM-WP400-1.1.3"],
                []
            ],
            "safety": "",
            "confidence": 0.95,
            "is_grounded": True
        }
        mock_client = self._create_mock_client(mock_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="Extraction fan stopped and velocity dropped below minimum on WP-400.",
            retrieved_context=[self.doc_wp400_ts, self.doc_wp400_um],
            api_key="test-key"
        )

        card1_citations = res["suggestion_sources"][0]
        self.assertEqual(len(card1_citations), 2)
        refs = [c["document_reference"] for c in card1_citations]
        self.assertIn("TS-WP400-3.1", refs)
        self.assertIn("UM-WP400-1.1.3", refs)

    # --------------------------------------------------------------------------
    # 8. Positive Conversations With No Technical Issue
    # --------------------------------------------------------------------------
    def test_positive_conversation_returns_no_suggestions(self):
        """Verifies positive customer conversations return has_issue=False and 0 suggestions."""
        res = generate_support_suggestion(
            conversation_text="Thank you so much! Everything is running smoothly and works great now.",
            retrieved_context=[self.doc_wp400_ts],
            api_key="test-key"
        )

        self.assertFalse(res["has_issue"])
        self.assertEqual(len(res["suggestions"]), 0)
        self.assertEqual(res["suggestion_sources"], [])
        self.assertEqual(res["suggestion"], "")

    # --------------------------------------------------------------------------
    # 9. Ensuring Model Never Falls Back to Pretrained Knowledge on Missing Docs
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_never_falls_back_to_pretrained_knowledge(self, mock_client_cls):
        """Verifies model reports insufficient evidence rather than guessing from pretrained weights."""
        mock_payload = {
            "has_issue": True,
            "evidence_status": "insufficient",
            "likely_issue": "Undocumented equipment error",
            "suggestions": [],
            "suggestion_sources": [],
            "unverified_details": "Error E-999 is not in the documentation.",
            "safety": "",
            "confidence": 0.0,
            "is_grounded": False
        }
        mock_client = self._create_mock_client(mock_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="WP-400 displays E-999. What does it mean?",
            retrieved_context=[self.doc_wp400_ts],  # doc only has E-102, NOT E-999!
            api_key="test-key"
        )

        self.assertEqual(res["evidence_status"], EVIDENCE_STATUS_INSUFFICIENT)
        self.assertEqual(len(res["suggestions"]), 0)
        self.assertEqual(res["suggestion_sources"], [])
        self.assertFalse(res["is_grounded"])
        self.assertIn("I couldn't find sufficient information about this issue", res["suggestion"])


if __name__ == "__main__":
    unittest.main()

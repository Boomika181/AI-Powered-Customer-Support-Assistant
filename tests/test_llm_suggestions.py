"""
tests/test_llm_suggestions.py

Unit tests for AI-powered support suggestion generation using Google Gemini 3.8 Flash.
All tests mock Gemini API interactions and do NOT make external network requests.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock

from src.llm_suggestions import (
    generate_support_suggestion,
    generate_suggestions,
    get_shared_gemini_client,
    SuggestionGenerationError,
)


class TestLLMSuggestions(unittest.TestCase):

    def setUp(self):
        import src.llm_suggestions
        src.llm_suggestions._shared_client = None
        src.llm_suggestions._shared_client_key = None
        self.sample_context = [
            {
                "text": (
                    "Section 3.1 WP-400 error codes\n"
                    "Code: E-102 | Meaning: Dust extraction fault | "
                    "Likely cause: Filter blocked (pressure drop above 1200 Pa) or fan stopped\n"
                    "Action: Clean or replace cartridge WP4-FL-DE1. Check fan breaker."
                ),
                "metadata": {
                    "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                    "page": 6,
                    "document_reference": "TS-WP400-3.1",
                    "section": "Section 3.1 WP-400 error codes",
                    "category": "Technical Troubleshooting",
                    "machine": "WP-400"
                }
            }
        ]

    def _create_mock_client(self, response_text: str):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    # --------------------------------------------------------------------------
    # 1. Valid Suggestion Generation & Model Verification
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    @patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.5-flash-lite"})
    def test_valid_suggestion_generation(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "suggestion": (
                "The WP-400 E-102 alarm indicates a dust extraction fault caused by a blocked filter or stopped fan. "
                "Inspect and clean or replace filter cartridge WP4-FL-DE1 and check the fan breaker. "
                "Remember to follow site lock-out/tag-out safety procedures before servicing."
            ),
            "confidence": 0.95,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="The WP-400 is flashing E-102. What should I check?",
            retrieved_context=self.sample_context,
            category="Technical Troubleshooting",
            api_key="test-gemini-key"
        )

        self.assertIn("suggestion", res)
        self.assertIn("E-102", res["suggestion"])
        self.assertIn("WP4-FL-DE1", res["suggestion"])
        self.assertEqual(res["confidence"], 0.95)
        self.assertTrue(res["is_grounded"])
        self.assertEqual(res["category"], "Technical Troubleshooting")

        # Verify model name used was gemini-3.5-flash-lite
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.5-flash-lite")
        self.assertNotIn("config", call_kwargs)  # no structured output config

    # --------------------------------------------------------------------------
    # 2. Prompt Construction & Context / Category Inclusion
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_prompt_construction_includes_context_and_category(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "suggestion": "Follow startup steps.",
            "confidence": 0.9,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        generate_support_suggestion(
            conversation_text="How to start?",
            retrieved_context=self.sample_context,
            category="Machine Operation Issues",
            api_key="test-gemini-key"
        )

        call_args = mock_client.models.generate_content.call_args
        prompt_text = call_args.kwargs["contents"]
        self.assertIn("The query is classified under: Machine Operation Issues.", prompt_text)
        self.assertIn("TS-WP400-3.1", prompt_text)
        self.assertIn("Dust extraction fault", prompt_text)
        self.assertIn("How to start?", prompt_text)

    # --------------------------------------------------------------------------
    # 3. Source Metadata Preserved
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_source_metadata_preserved(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "suggestion": "Valid response.",
            "confidence": 0.9,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="Query",
            retrieved_context=self.sample_context,
            api_key="test-gemini-key"
        )

        self.assertIn("sources", res)
        self.assertEqual(len(res["sources"]), 1)
        source_meta = res["sources"][0]
        self.assertEqual(source_meta["source_file"], "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf")
        self.assertEqual(source_meta["document_reference"], "TS-WP400-3.1")
        self.assertEqual(source_meta["page"], 6)

    # --------------------------------------------------------------------------
    # 4. Empty Conversation Rejected
    # --------------------------------------------------------------------------
    def test_empty_conversation_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            generate_support_suggestion(
                conversation_text="",
                retrieved_context=self.sample_context,
                api_key="test-gemini-key"
            )
        self.assertIn("conversation_text cannot be empty", str(ctx.exception))

        with self.assertRaises(ValueError):
            generate_support_suggestion(
                conversation_text="   \n\t ",
                retrieved_context=self.sample_context,
                api_key="test-gemini-key"
            )

    # --------------------------------------------------------------------------
    # 5. Empty Retrieved Context Handled Safely (No Gemini Call)
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_empty_retrieved_context_handled_safely_without_calling_gemini(self, mock_client_cls):
        res = generate_support_suggestion(
            conversation_text="What is the hydraulic pressure for unknown machine?",
            retrieved_context=[],
            api_key="test-gemini-key"
        )

        # Gemini Client must NOT have been instantiated or called
        mock_client_cls.assert_not_called()
        self.assertIn("No relevant information was found in the knowledge base", res["suggestion"])
        self.assertEqual(res["confidence"], 0.0)
        self.assertFalse(res["is_grounded"])
        self.assertEqual(res["sources"], [])

    # --------------------------------------------------------------------------
    # 6. Malformed JSON & Markdown Fences Handled
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_markdown_fenced_response_handled(self, mock_client_cls):
        fenced_payload = (
            "```json\n"
            "{\n"
            '  "suggestion": "Grounded procedure.",\n'
            '  "confidence": 0.92,\n'
            '  "is_grounded": true\n'
            "}\n"
            "```"
        )
        mock_client = self._create_mock_client(fenced_payload)
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="Query",
            retrieved_context=self.sample_context,
            api_key="test-gemini-key"
        )
        self.assertEqual(res["suggestion"], "Grounded procedure.")
        self.assertEqual(res["confidence"], 0.92)

    @patch("src.llm_suggestions.genai.Client")
    def test_malformed_json_raises_error(self, mock_client_cls):
        mock_client = self._create_mock_client("Plain text without JSON")
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SuggestionGenerationError) as ctx:
            generate_support_suggestion(
                conversation_text="Query",
                retrieved_context=self.sample_context,
                api_key="test-gemini-key"
            )
        self.assertIn("invalid json", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 7. Gemini API Exceptions & Missing Key Handled
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_gemini_api_exception_raises_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Upstream API error")
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SuggestionGenerationError) as ctx:
            generate_support_suggestion(
                conversation_text="Query",
                retrieved_context=self.sample_context,
                api_key="test-gemini-key"
            )
        self.assertIn("Gemini API request failed", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            generate_support_suggestion(
                conversation_text="Query",
                retrieved_context=self.sample_context,
                api_key=None
            )
        self.assertIn("GEMINI_API_KEY is required", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 8. Multiple & Long Retrieved Context Chunks Handled Safely
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_multiple_chunks_handled(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "suggestion": "Multi-chunk resolution.",
            "confidence": 0.88,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        two_chunks = self.sample_context + [
            {
                "text": "Section 4.1 Replace saw blade: switch off isolator and lock out.",
                "metadata": {
                    "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                    "page": 8,
                    "document_reference": "RP-WP400-4.1",
                    "section": "Section 4.1",
                    "category": "Maintenance & Parts",
                    "machine": "WP-400"
                }
            }
        ]

        res = generate_support_suggestion(
            conversation_text="Procedure",
            retrieved_context=two_chunks,
            api_key="test-gemini-key"
        )
        self.assertEqual(len(res["sources"]), 2)

    # --------------------------------------------------------------------------
    # 9. Backward-Compatible Wrapper
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_generate_suggestions_wrapper(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "suggestion": "Wrapper test.",
            "confidence": 0.9,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        res = generate_suggestions(
            query_text="Query",
            retrieved_chunks=self.sample_context,
            api_key="test-gemini-key"
        )
        self.assertEqual(res["suggestion"], "Wrapper test.")

    def tearDown(self):
        import src.llm_suggestions
        src.llm_suggestions._shared_client = None
        src.llm_suggestions._shared_client_key = None

    # --------------------------------------------------------------------------
    # 10. Persistent Gemini Client Reuse & Injection
    # --------------------------------------------------------------------------
    def test_get_shared_gemini_client_caching(self):
        """Verifies get_shared_gemini_client creates once and returns the same client on repeated calls."""
        with patch("src.llm_suggestions.genai.Client") as mock_genai_ctor:
            fake_client = MagicMock()
            mock_genai_ctor.return_value = fake_client

            client_a = get_shared_gemini_client(api_key="test_shared_key_12345")
            client_b = get_shared_gemini_client(api_key="test_shared_key_12345")

            self.assertIs(client_a, client_b)
            self.assertEqual(mock_genai_ctor.call_count, 1)

    def test_two_suggestion_calls_reuse_shared_client(self):
        """Verifies two consecutive suggestion calls without an injected client reuse the shared client."""
        with patch("src.llm_suggestions.genai.Client") as mock_genai_ctor:
            fake_client = self._create_mock_client(json.dumps({
                "suggestion": "Test procedure.",
                "confidence": 0.95,
                "is_grounded": True
            }))
            mock_genai_ctor.return_value = fake_client

            res1 = generate_support_suggestion(
                conversation_text="First query",
                retrieved_context=self.sample_context,
                api_key="test-shared-key-reuse"
            )
            res2 = generate_support_suggestion(
                conversation_text="Second query",
                retrieved_context=self.sample_context,
                api_key="test-shared-key-reuse"
            )

            # genai.Client() constructor should only have been called ONCE across both turns
            self.assertEqual(mock_genai_ctor.call_count, 1)
            # generate_content should have been called twice on the same shared client
            self.assertEqual(fake_client.models.generate_content.call_count, 2)
            self.assertEqual(res1["suggestion"], "Test procedure.")
            self.assertEqual(res2["suggestion"], "Test procedure.")

    def test_injected_client_is_used(self):
        """Verifies that injecting a client uses it directly without instantiating genai.Client."""
        with patch("src.llm_suggestions.genai.Client") as mock_genai_ctor:
            injected_client = self._create_mock_client(json.dumps({
                "suggestion": "Injected client suggestion.",
                "confidence": 0.98,
                "is_grounded": True
            }))

            res = generate_support_suggestion(
                conversation_text="Query with injected client",
                retrieved_context=self.sample_context,
                api_key="dummy-key",
                client=injected_client
            )

            # genai.Client should not be constructed
            self.assertEqual(mock_genai_ctor.call_count, 0)
            injected_client.models.generate_content.assert_called_once()
            self.assertEqual(res["suggestion"], "Injected client suggestion.")

    # --------------------------------------------------------------------------
    # 11. Exactly 4 Distinct Suggestions & Issue Detection
    # --------------------------------------------------------------------------
    @patch("src.llm_suggestions.genai.Client")
    def test_exactly_four_suggestions_returned_for_issue(self, mock_client_cls):
        """Verifies exactly 4 distinct suggestions are returned when Gemini returns 4 items."""
        mock_client = self._create_mock_client(json.dumps({
            "has_issue": True,
            "likely_issue": "WP-400 Dust Extraction Interlock Fault (E-102)",
            "suggestions": [
                "Isolate Power: Switch off the main isolator switch and apply LOTO padlock.",
                "Inspect Cartridge: Check dust extraction cartridge WP4-FL-DE1 for blockages.",
                "Verify Differential Pressure: Measure pressure drop across the filter to ensure <1200 Pa.",
                "Check Circuit Breaker: Inspect and reset the dedicated 16A extraction fan circuit breaker."
            ],
            "safety": "Never open extraction ducting while the main blade is spinning.",
            "confidence": 0.98,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="The saw stopped with E-102 error.",
            retrieved_context=self.sample_context,
            api_key="test-key"
        )

        self.assertTrue(res["has_issue"])
        self.assertEqual(len(res["suggestions"]), 4)
        self.assertEqual(len(set(res["suggestions"])), 4)  # 4 distinct items
        self.assertIn("WP4-FL-DE1", res["suggestions"][1])
        self.assertTrue(res["is_grounded"])
        self.assertIn("1.", res["suggestion"])
        self.assertIn("4.", res["suggestion"])

    @patch("src.llm_suggestions.genai.Client")
    def test_fewer_than_four_suggestions_padded_to_four(self, mock_client_cls):
        """Verifies that if Gemini returns only 2 suggestions, it is padded with safe steps to exactly 4 distinct items."""
        mock_client = self._create_mock_client(json.dumps({
            "has_issue": True,
            "likely_issue": "Filter obstruction",
            "suggestions": [
                "Clean or replace filter cartridge WP4-FL-DE1.",
                "Verify fan breaker in cabinet."
            ],
            "safety": "Disconnect power before servicing.",
            "confidence": 0.92,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="Filter blocked on WP-400.",
            retrieved_context=self.sample_context,
            api_key="test-key"
        )

        self.assertTrue(res["has_issue"])
        self.assertEqual(len(res["suggestions"]), 4)
        self.assertEqual(len(set(res["suggestions"])), 4)
        self.assertIn("WP4-FL-DE1", res["suggestions"][0])

    def test_positive_conversation_without_issue_returns_no_suggestions(self):
        """Verifies that a purely positive conversation with no reported issue generates 0 suggestions and has_issue=False."""
        res = generate_support_suggestion(
            conversation_text="Thank you so much! Everything is running smoothly now, we appreciate the help.",
            retrieved_context=self.sample_context,
            api_key="test-key"
        )

        self.assertFalse(res["has_issue"])
        self.assertEqual(len(res["suggestions"]), 0)
        self.assertEqual(res["suggestion"], "")

    def test_incomplete_information_with_empty_context_provides_safe_next_steps(self):
        """Verifies incomplete info / empty context returns 4 safe distinct next steps without calling LLM."""
        res = generate_support_suggestion(
            conversation_text="Our machine stopped but no error code was displayed.",
            retrieved_context=[],
            api_key="test-key"
        )

        self.assertTrue(res["has_issue"])
        self.assertEqual(len(res["suggestions"]), 4)
        self.assertEqual(len(set(res["suggestions"])), 4)
        self.assertFalse(res["is_grounded"])
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(res["suggestion_sources"], [[], [], [], []])

    @patch("src.llm_suggestions.genai.Client")
    def test_single_and_multiple_citations_per_suggestion(self, mock_client_cls):
        """Tests 1 citation, 2 citations, 3 citations, shared docs, and standard protocols."""
        multi_context = [
            {
                "text": "Code: E-102 | Dust extraction fault | Action: Clean or replace cartridge WP4-FL-DE1. Check fan breaker.",
                "metadata": {
                    "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                    "page": 6,
                    "document_reference": "TS-WP400-3.1",
                    "section": "Section 3.1 WP-400 error codes"
                }
            },
            {
                "text": "Minimum air velocity at the saw hood is 20 m/s. Replace filter cartridge WP4-FL-DE1.",
                "metadata": {
                    "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                    "page": 2,
                    "document_reference": "UM-WP400-1.1.3",
                    "section": "Section 1.1.3 Dust extraction"
                }
            },
            {
                "text": "WP4-FL-DE1 Dust extraction filter cartridge Fits WP-400 Replace when Alarm E-102 or every 3 months.",
                "metadata": {
                    "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                    "page": 4,
                    "document_reference": "MP-ALL-2.3",
                    "section": "Section 2.3 Parts catalog"
                }
            }
        ]

        mock_client = self._create_mock_client(json.dumps({
            "has_issue": True,
            "likely_issue": "Extraction fan thermal overload and filter blockage",
            "suggestions": [
                "Inspect cartridge WP4-FL-DE1 and check the fan breaker.",
                "Verify minimum air velocity is at least 20 m/s and replace cartridge WP4-FL-DE1 every 3 months per catalog.",
                "Cross-reference TS-WP400-3.1, UM-WP400-1.1.3, and MP-ALL-2.3 for complete dust extraction specifications.",
                "Escalate to technical engineering if the fault persists."
            ],
            "suggestion_sources": [
                ["TS-WP400-3.1"],
                ["UM-WP400-1.1.3", "MP-ALL-2.3"],
                ["TS-WP400-3.1", "UM-WP400-1.1.3", "MP-ALL-2.3"],
                []
            ],
            "safety": "Isolate machine power before servicing.",
            "confidence": 0.95,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="WP-400 extraction fault alarm E-102.",
            retrieved_context=multi_context,
            api_key="test-key"
        )

        sugg_sources = res["suggestion_sources"]
        self.assertEqual(len(sugg_sources), 4)

        # 1. One suggestion with one citation
        self.assertEqual(len(sugg_sources[0]), 1)
        self.assertEqual(sugg_sources[0][0]["document_reference"], "TS-WP400-3.1")
        self.assertEqual(sugg_sources[0][0]["page"], 6)

        # 2. One suggestion with two citations
        self.assertEqual(len(sugg_sources[1]), 2)
        refs_1 = [c["document_reference"] for c in sugg_sources[1]]
        self.assertIn("UM-WP400-1.1.3", refs_1)
        self.assertIn("MP-ALL-2.3", refs_1)

        # 3. One suggestion with three citations
        self.assertEqual(len(sugg_sources[2]), 3)
        refs_2 = [c["document_reference"] for c in sugg_sources[2]]
        self.assertIn("TS-WP400-3.1", refs_2)
        self.assertIn("UM-WP400-1.1.3", refs_2)
        self.assertIn("MP-ALL-2.3", refs_2)

        # 4. Standard support protocol with no document citation
        self.assertEqual(len(sugg_sources[3]), 0)

        # 5. Multiple suggestions sharing the same document
        # TS-WP400-3.1 is in suggestion #1 and suggestion #3
        ts_source = next(s for s in res["sources"] if s["document_reference"] == "TS-WP400-3.1")
        self.assertIn(1, ts_source["supported_suggestions"])
        self.assertIn(3, ts_source["supported_suggestions"])

        # 6. No fabricated references
        valid_refs = {"TS-WP400-3.1", "UM-WP400-1.1.3", "MP-ALL-2.3"}
        for group in sugg_sources:
            for c in group:
                self.assertIn(c["document_reference"], valid_refs)

    @patch("src.llm_suggestions.genai.Client")
    def test_missing_citation_metadata_does_not_fabricate(self, mock_client_cls):
        """Verifies that chunks with missing page numbers (-1) do not fabricate pages."""
        sparse_context = [
            {
                "text": "Check sensor alignment.",
                "metadata": {
                    "document_reference": "DOC-SPARSE-01",
                    "page": -1,
                    "section": "Sensor Maintenance"
                }
            }
        ]

        mock_client = self._create_mock_client(json.dumps({
            "has_issue": True,
            "likely_issue": "Sensor issue",
            "suggestions": [
                "Check sensor alignment per DOC-SPARSE-01.",
                "Verify wiring harness.",
                "Clean optical sensor surface.",
                "Escalate to technical engineering."
            ],
            "suggestion_sources": [
                ["DOC-SPARSE-01"],
                [],
                [],
                []
            ],
            "safety": "",
            "confidence": 0.9,
            "is_grounded": True
        }))
        mock_client_cls.return_value = mock_client

        res = generate_support_suggestion(
            conversation_text="Sensor is not detecting material.",
            retrieved_context=sparse_context,
            api_key="test-key"
        )

        citation = res["suggestion_sources"][0][0]
        self.assertEqual(citation["document_reference"], "DOC-SPARSE-01")
        self.assertEqual(citation["page"], -1)  # preserved as -1, not fabricated


if __name__ == "__main__":
    unittest.main()

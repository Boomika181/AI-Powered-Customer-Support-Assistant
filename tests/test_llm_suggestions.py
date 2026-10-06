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
    SuggestionGenerationError,
)


class TestLLMSuggestions(unittest.TestCase):

    def setUp(self):
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
    @patch.dict(os.environ, {"LLM_MODEL": "gemini-3.8-flash"})
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

        # Verify model name used was gemini-3.8-flash
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.8-flash")
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


if __name__ == "__main__":
    unittest.main()

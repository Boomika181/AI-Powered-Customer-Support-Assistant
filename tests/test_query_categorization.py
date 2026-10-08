"""
tests/test_query_categorization.py

Unit tests for Gemini 3.8 Flash Customer Query Categorization.
All tests use mocked responses and do NOT make live external API calls.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock

from src.query_categorization import (
    classify_query,
    QueryCategorizationError,
)


class TestQueryCategorization(unittest.TestCase):

    def _create_mock_client(self, response_text: str):
        """Helper to create a mocked google.genai.Client instance."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    # --------------------------------------------------------------------------
    # 1. Machine Operation Issues
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_machine_operation_issues(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Machine Operation Issues",
            "confidence": 0.96,
            "rationale": "Customer is asking about the startup sequence and operating mode."
        }))
        mock_client_cls.return_value = mock_client

        text = "How do I start the WP-400 saw and what working pressure should I set?"
        result = classify_query(text, api_key="test-api-key")

        self.assertEqual(result["category"], "Machine Operation Issues")
        self.assertAlmostEqual(result["confidence"], 0.96)
        self.assertTrue(len(result["rationale"]) > 0)
        mock_client.models.generate_content.assert_called_once()

    # --------------------------------------------------------------------------
    # 2. Maintenance & Parts
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_maintenance_and_parts(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Maintenance & Parts",
            "confidence": 0.98,
            "rationale": "Customer is inquiring about replacing the spindle drive belt and lubrication."
        }))
        mock_client_cls.return_value = mock_client

        text = "When do I need to replace the spindle belt and what grease should I use for linear rails?"
        result = classify_query(text, api_key="test-api-key")

        self.assertEqual(result["category"], "Maintenance & Parts")
        self.assertAlmostEqual(result["confidence"], 0.98)
        self.assertIn("spindle drive belt", result["rationale"])

    # --------------------------------------------------------------------------
    # 3. Technical Troubleshooting
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_technical_troubleshooting(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Technical Troubleshooting",
            "confidence": 0.99,
            "rationale": "Customer reports error code E-102 and dust extraction fault alarm."
        }))
        mock_client_cls.return_value = mock_client

        text = "The machine stopped and is flashing error E-102 on the screen."
        result = classify_query(text, api_key="test-api-key")

        self.assertEqual(result["category"], "Technical Troubleshooting")
        self.assertAlmostEqual(result["confidence"], 0.99)
        self.assertIn("E-102", result["rationale"])

    # --------------------------------------------------------------------------
    # 4. Markdown-Fenced JSON Response
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_markdown_fenced_json_response(self, mock_client_cls):
        fenced_payload = (
            "```json\n"
            "{\n"
            '  "category": "Maintenance & Parts",\n'
            '  "confidence": 0.94,\n'
            '  "rationale": "Customer requests replacement nozzle part number."\n'
            "}\n"
            "```"
        )
        mock_client = self._create_mock_client(fenced_payload)
        mock_client_cls.return_value = mock_client

        result = classify_query("I need part number for 65A nozzle on MC-250.", api_key="test-api-key")
        self.assertEqual(result["category"], "Maintenance & Parts")
        self.assertEqual(result["confidence"], 0.94)
        self.assertEqual(result["rationale"], "Customer requests replacement nozzle part number.")

    # --------------------------------------------------------------------------
    # 5. Malformed JSON
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_malformed_json_raises_query_categorization_error(self, mock_client_cls):
        mock_client = self._create_mock_client("This is not valid JSON")
        mock_client_cls.return_value = mock_client

        with self.assertRaises(QueryCategorizationError) as ctx:
            classify_query("Machine not starting", api_key="test-api-key")
        self.assertIn("invalid json", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 6. Invalid Category
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_invalid_category_raises_error(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Billing and Invoicing",
            "confidence": 0.8,
            "rationale": "Customer asked about costs."
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(QueryCategorizationError) as ctx:
            classify_query("How much does this cost?", api_key="test-api-key")
        self.assertIn("Invalid category 'Billing and Invoicing'", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 7. Confidence Below 0
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_confidence_below_zero(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Technical Troubleshooting",
            "confidence": -0.1,
            "rationale": "Valid explanation."
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(QueryCategorizationError) as ctx:
            classify_query("Error E-101", api_key="test-api-key")
        self.assertIn("out of bounds", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 8. Confidence Above 1
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_confidence_above_one(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Technical Troubleshooting",
            "confidence": 1.25,
            "rationale": "Valid explanation."
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(QueryCategorizationError) as ctx:
            classify_query("Error E-101", api_key="test-api-key")
        self.assertIn("out of bounds", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 9. Confidence Non-Numeric
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_confidence_non_numeric(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Technical Troubleshooting",
            "confidence": "very high",
            "rationale": "Valid explanation."
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(QueryCategorizationError) as ctx:
            classify_query("Error E-101", api_key="test-api-key")
        self.assertIn("numeric float", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 10. Empty Rationale
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_empty_rationale_raises_error(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Machine Operation Issues",
            "confidence": 0.85,
            "rationale": "   "
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(QueryCategorizationError) as ctx:
            classify_query("How to turn on machine", api_key="test-api-key")
        self.assertIn("rationale", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 11. Empty Conversation
    # --------------------------------------------------------------------------
    def test_empty_input_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            classify_query("", api_key="test-api-key")
        self.assertIn("conversation_text cannot be empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx_ws:
            classify_query("   \n\t  ", api_key="test-api-key")
        self.assertIn("conversation_text cannot be empty", str(ctx_ws.exception))

    # --------------------------------------------------------------------------
    # 12. Missing API Key
    # --------------------------------------------------------------------------
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            classify_query("How do I operate this saw?", api_key=None)
        self.assertIn("GEMINI_API_KEY is required", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 13. Gemini API Failure
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    def test_gemini_api_failure_raises_query_categorization_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Upstream connection reset")
        mock_client_cls.return_value = mock_client

        with self.assertRaises(QueryCategorizationError) as ctx:
            classify_query("Machine trouble", api_key="test-api-key")
        self.assertIn("Gemini API request failed", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 14. Valid Response Structure & Model Name
    # --------------------------------------------------------------------------
    @patch("src.query_categorization.genai.Client")
    @patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.5-flash-lite"})
    def test_valid_successful_response_and_model(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "category": "Machine Operation Issues",
            "confidence": 0.92,
            "rationale": "Inquiring about startup sequence."
        }))
        mock_client_cls.return_value = mock_client

        res = classify_query("Start machine procedure", api_key="test-api-key")
        self.assertIsInstance(res, dict)
        self.assertIn("category", res)
        self.assertIn("confidence", res)
        self.assertIn("rationale", res)
        self.assertEqual(res["category"], "Machine Operation Issues")
        self.assertEqual(res["confidence"], 0.92)

        called_args = mock_client.models.generate_content.call_args
        self.assertEqual(called_args.kwargs["model"], "gemini-3.5-flash-lite")
        # Verify no response_schema or response_mime_type in call
        self.assertNotIn("config", called_args.kwargs)


if __name__ == "__main__":
    unittest.main()

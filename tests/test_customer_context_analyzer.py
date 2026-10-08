"""
tests/test_customer_context_analyzer.py

Unit tests for the consolidated Customer Context Analyzer:
Combines Sentiment Analysis and Query Categorization into a single Gemini generation call.
NO LIVE GEMINI API CALLS ARE MADE BY THIS TEST SUITE.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.customer_context_analyzer import (
    analyze_customer_context,
    CustomerContextAnalysisError,
    VALID_SENTIMENTS,
    VALID_CATEGORIES,
)


class TestCustomerContextAnalyzer(unittest.TestCase):

    def _mock_client_with_response(self, text_response: str):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = text_response
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    # --------------------------------------------------------------------------
    # 1. Valid Positive + Machine Operation Issues response
    # --------------------------------------------------------------------------
    def test_valid_positive_machine_operation(self):
        payload = json.dumps({
            "sentiment": "Positive",
            "sentiment_confidence": 0.95,
            "sentiment_rationale": "Customer expressing satisfaction with feed rate adjustment.",
            "category": "Machine Operation Issues",
            "category_confidence": 0.92,
            "category_rationale": "Inquiry regarding standard spindle speed and feed rate operation."
        })
        client = self._mock_client_with_response(payload)
        res = analyze_customer_context("How do I set the spindle speed for aluminum?", client=client)

        self.assertEqual(res["sentiment"], "Positive")
        self.assertEqual(res["sentiment_confidence"], 0.95)
        self.assertEqual(res["category"], "Machine Operation Issues")
        self.assertEqual(res["category_confidence"], 0.92)

    # --------------------------------------------------------------------------
    # 2. Valid Negative + Technical Troubleshooting response
    # --------------------------------------------------------------------------
    def test_valid_negative_technical_troubleshooting(self):
        payload = json.dumps({
            "sentiment": "Negative",
            "sentiment_confidence": 0.88,
            "sentiment_rationale": "Customer expressing frustration about machine breakdown.",
            "category": "Technical Troubleshooting",
            "category_confidence": 0.98,
            "category_rationale": "Reporting alarm code E-102 and dust extraction fault."
        })
        client = self._mock_client_with_response(payload)
        res = analyze_customer_context("My WP400 has error E102 and won't start.", client=client)

        self.assertEqual(res["sentiment"], "Negative")
        self.assertEqual(res["category"], "Technical Troubleshooting")
        self.assertEqual(res["category_confidence"], 0.98)

    # --------------------------------------------------------------------------
    # 3. Valid Neutral + Maintenance & Parts response
    # --------------------------------------------------------------------------
    def test_valid_neutral_maintenance_parts(self):
        payload = json.dumps({
            "sentiment": "Neutral",
            "sentiment_confidence": 0.90,
            "sentiment_rationale": "Factual request for replacement parts.",
            "category": "Maintenance & Parts",
            "category_confidence": 0.94,
            "category_rationale": "Asking for part number of the air filter cartridge."
        })
        client = self._mock_client_with_response(payload)
        res = analyze_customer_context("I need the part number for the WP-400 filter cartridge.", client=client)

        self.assertEqual(res["sentiment"], "Neutral")
        self.assertEqual(res["category"], "Maintenance & Parts")

    # --------------------------------------------------------------------------
    # 4. Valid JSON parsing with markdown code fences
    # --------------------------------------------------------------------------
    def test_valid_json_with_markdown_fences(self):
        raw_text = """```json
{
  "sentiment": "Agitated",
  "sentiment_confidence": 0.99,
  "sentiment_rationale": "Urgent and frustrated about ongoing stoppage.",
  "category": "Technical Troubleshooting",
  "category_confidence": 0.96,
  "category_rationale": "Spindle motor overheating alarm."
}
```"""
        client = self._mock_client_with_response(raw_text)
        res = analyze_customer_context("The machine is overheating, help immediately!", client=client)

        self.assertEqual(res["sentiment"], "Agitated")
        self.assertEqual(res["category"], "Technical Troubleshooting")

    # --------------------------------------------------------------------------
    # 5. Invalid JSON handling
    # --------------------------------------------------------------------------
    def test_invalid_json_handling(self):
        client = self._mock_client_with_response("This is not JSON at all!")
        with self.assertRaises(CustomerContextAnalysisError) as ctx:
            analyze_customer_context("Help me with my machine.", client=client)
        self.assertIn("invalid JSON", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 6. Missing sentiment handling
    # --------------------------------------------------------------------------
    def test_missing_sentiment_handling(self):
        payload = json.dumps({
            "category": "Machine Operation Issues",
            "category_confidence": 0.90,
            "category_rationale": "Operation query"
        })
        client = self._mock_client_with_response(payload)
        with self.assertRaises(CustomerContextAnalysisError) as ctx:
            analyze_customer_context("How do I start it?", client=client)
        self.assertIn("Invalid sentiment", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 7. Missing category handling
    # --------------------------------------------------------------------------
    def test_missing_category_handling(self):
        payload = json.dumps({
            "sentiment": "Neutral",
            "sentiment_confidence": 0.90,
            "sentiment_rationale": "Neutral tone"
        })
        client = self._mock_client_with_response(payload)
        with self.assertRaises(CustomerContextAnalysisError) as ctx:
            analyze_customer_context("How do I start it?", client=client)
        self.assertIn("Invalid category", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 8. Invalid sentiment label handling
    # --------------------------------------------------------------------------
    def test_invalid_sentiment_label_handling(self):
        payload = json.dumps({
            "sentiment": "SuperAngry",
            "sentiment_confidence": 0.90,
            "sentiment_rationale": "Invalid label",
            "category": "Technical Troubleshooting",
            "category_confidence": 0.90,
            "category_rationale": "Valid category"
        })
        client = self._mock_client_with_response(payload)
        with self.assertRaises(CustomerContextAnalysisError) as ctx:
            analyze_customer_context("Machine is broken.", client=client)
        self.assertIn("Invalid sentiment 'SuperAngry'", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 9. Invalid category label handling
    # --------------------------------------------------------------------------
    def test_invalid_category_label_handling(self):
        payload = json.dumps({
            "sentiment": "Neutral",
            "sentiment_confidence": 0.90,
            "sentiment_rationale": "Neutral tone",
            "category": "Billing & Payments",
            "category_confidence": 0.90,
            "category_rationale": "Invalid industrial category"
        })
        client = self._mock_client_with_response(payload)
        with self.assertRaises(CustomerContextAnalysisError) as ctx:
            analyze_customer_context("How much does it cost?", client=client)
        self.assertIn("Invalid category 'Billing & Payments'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

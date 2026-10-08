"""
tests/test_sentiment_analysis.py

Unit tests for Gemini 3.8 Flash Customer Sentiment Analysis.
All tests use mocked responses and do NOT make live external API calls.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock

from src.sentiment_analysis import (
    analyze_sentiment,
    SentimentAnalysisError,
)


class TestSentimentAnalysis(unittest.TestCase):

    def _create_mock_client(self, response_text: str):
        """Helper to create a mocked google.genai.Client instance."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    # --------------------------------------------------------------------------
    # 1. Positive Response
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_positive_sentiment(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Positive",
            "confidence": 0.98,
            "rationale": "Customer expressed appreciation and noted the machine is working perfectly."
        }))
        mock_client_cls.return_value = mock_client

        text = "Thank you, the machine is working perfectly now. I really appreciate your help."
        result = analyze_sentiment(text, api_key="test-api-key")

        self.assertEqual(result["sentiment"], "Positive")
        self.assertAlmostEqual(result["confidence"], 0.98)
        self.assertTrue(len(result["rationale"]) > 0)
        mock_client.models.generate_content.assert_called_once()

    # --------------------------------------------------------------------------
    # 2. Neutral Response
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_neutral_sentiment(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Neutral",
            "confidence": 0.95,
            "rationale": "Customer provided factual problem description regarding error code E101."
        }))
        mock_client_cls.return_value = mock_client

        text = "The machine is showing error E101. It stopped after approximately ten minutes."
        result = analyze_sentiment(text, api_key="test-api-key")

        self.assertEqual(result["sentiment"], "Neutral")
        self.assertAlmostEqual(result["confidence"], 0.95)
        self.assertIn("error code E101", result["rationale"])

    # --------------------------------------------------------------------------
    # 3. Negative Response
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_negative_sentiment(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Negative",
            "confidence": 0.91,
            "rationale": "Customer expressed clear disappointment that the machine failed again."
        }))
        mock_client_cls.return_value = mock_client

        text = "I am disappointed because the machine has stopped working again."
        result = analyze_sentiment(text, api_key="test-api-key")

        self.assertEqual(result["sentiment"], "Negative")
        self.assertAlmostEqual(result["confidence"], 0.91)
        self.assertTrue(result["confidence"] <= 1.0)

    # --------------------------------------------------------------------------
    # 4. Agitated Response
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_agitated_sentiment(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Agitated",
            "confidence": 0.99,
            "rationale": "Customer displays intense frustration, urgency, and repeated breakdowns."
        }))
        mock_client_cls.return_value = mock_client

        text = (
            "This is extremely frustrating! The machine has stopped three times today "
            "and nobody has fixed this. I need this resolved immediately."
        )
        result = analyze_sentiment(text, api_key="test-api-key")

        self.assertEqual(result["sentiment"], "Agitated")
        self.assertAlmostEqual(result["confidence"], 0.99)
        self.assertTrue(len(result["rationale"]) > 0)

    # --------------------------------------------------------------------------
    # 5. Markdown-Fenced JSON Response
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_markdown_fenced_json_response(self, mock_client_cls):
        fenced_payload = (
            "```json\n"
            "{\n"
            '  "sentiment": "Positive",\n'
            '  "confidence": 0.96,\n'
            '  "rationale": "Customer expressed relief."\n'
            "}\n"
            "```"
        )
        mock_client = self._create_mock_client(fenced_payload)
        mock_client_cls.return_value = mock_client

        result = analyze_sentiment("Problem solved, thank you!", api_key="test-api-key")
        self.assertEqual(result["sentiment"], "Positive")
        self.assertEqual(result["confidence"], 0.96)
        self.assertEqual(result["rationale"], "Customer expressed relief.")

    # --------------------------------------------------------------------------
    # 6. Malformed JSON
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_malformed_json_raises_sentiment_analysis_error(self, mock_client_cls):
        mock_client = self._create_mock_client("Not a valid JSON string")
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SentimentAnalysisError) as ctx:
            analyze_sentiment("Some text", api_key="test-api-key")
        self.assertIn("invalid json", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 7. Invalid Sentiment Label
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_invalid_sentiment_label_raises_error(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Happy",
            "confidence": 0.8,
            "rationale": "Customer seems cheerful."
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SentimentAnalysisError) as ctx:
            analyze_sentiment("Some text", api_key="test-api-key")
        self.assertIn("Invalid sentiment 'Happy'", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 8. Confidence Below 0
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_confidence_below_zero(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Neutral",
            "confidence": -0.25,
            "rationale": "Valid rationale."
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SentimentAnalysisError) as ctx:
            analyze_sentiment("Some text", api_key="test-api-key")
        self.assertIn("out of bounds", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 9. Confidence Above 1
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_confidence_above_one(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Neutral",
            "confidence": 1.45,
            "rationale": "Valid rationale."
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SentimentAnalysisError) as ctx:
            analyze_sentiment("Some text", api_key="test-api-key")
        self.assertIn("out of bounds", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 10. Empty Rationale
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_empty_rationale_raises_error(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Positive",
            "confidence": 0.8,
            "rationale": "   "
        }))
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SentimentAnalysisError) as ctx:
            analyze_sentiment("Some text", api_key="test-api-key")
        self.assertIn("rationale", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 11. Empty Conversation
    # --------------------------------------------------------------------------
    def test_empty_input_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            analyze_sentiment("", api_key="test-api-key")
        self.assertIn("conversation_text cannot be empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx_ws:
            analyze_sentiment("   \n\t  ", api_key="test-api-key")
        self.assertIn("conversation_text cannot be empty", str(ctx_ws.exception))

    # --------------------------------------------------------------------------
    # 12. Missing API Key
    # --------------------------------------------------------------------------
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            analyze_sentiment("Machine error occurred", api_key=None)
        self.assertIn("GEMINI_API_KEY is required", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 13. Gemini API Failure
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    def test_gemini_api_failure_raises_sentiment_analysis_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Upstream network timeout")
        mock_client_cls.return_value = mock_client

        with self.assertRaises(SentimentAnalysisError) as ctx:
            analyze_sentiment("Some issue", api_key="test-api-key")
        self.assertIn("Gemini API request failed", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 14. Valid Successful Response Structure
    # --------------------------------------------------------------------------
    @patch("src.sentiment_analysis.genai.Client")
    @patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.5-flash-lite"})
    def test_valid_successful_response_and_model(self, mock_client_cls):
        mock_client = self._create_mock_client(json.dumps({
            "sentiment": "Neutral",
            "confidence": 0.85,
            "rationale": "Customer gave operational details without emotion."
        }))
        mock_client_cls.return_value = mock_client

        res = analyze_sentiment("Operating at 4000 rpm", api_key="test-api-key")
        self.assertIsInstance(res, dict)
        self.assertIn("sentiment", res)
        self.assertIn("confidence", res)
        self.assertIn("rationale", res)
        self.assertEqual(res["sentiment"], "Neutral")
        self.assertEqual(res["confidence"], 0.85)

        called_args = mock_client.models.generate_content.call_args
        self.assertEqual(called_args.kwargs["model"], "gemini-3.5-flash-lite")
        # Verify no response_schema or response_mime_type in call
        self.assertNotIn("config", called_args.kwargs)


if __name__ == "__main__":
    unittest.main()

"""
tests/test_speaker_role_classifier.py

Unit tests for Semantic Speaker Role Classifier (Phase 2 - Semantic Role Identification).
Validates all 20 required offline scenarios using mocked Gemini clients (zero live API calls).
"""

import unittest
from unittest.mock import MagicMock, patch

from src.speaker_role_classifier import (
    SpeakerContinuityTracker,
    build_role_classifier_prompt,
    classify_speaker_role,
    get_confidence_threshold,
    get_shared_gemini_client,
)
from src.pipeline_coordinator import PipelineCoordinator
from src.transcription import ConversationState


class TestSpeakerRoleClassifier(unittest.TestCase):

    def _make_mock_client(self, role: str, confidence: float, rationale: str = "Test reason"):
        """Helper to create a mocked Gemini client returning structured JSON."""
        mock_response = MagicMock()
        mock_response.text = f'{{"role": "{role}", "confidence": {confidence}, "rationale": "{rationale}"}}'
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    # --------------------------------------------------------------------------
    # 1. Customer speaks first
    # --------------------------------------------------------------------------
    def test_customer_speaks_first(self):
        """Turn 1 opening complaint from customer classifies as CUSTOMER."""
        mock_client = self._make_mock_client("CUSTOMER", 0.95, "Opening machine failure report")
        turn = {"text": "My WP400 has stopped working.", "raw_speaker": "A"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "CUSTOMER")
        self.assertGreaterEqual(result["confidence"], 0.80)
        self.assertGreater(result["latency_ms"], 0.0)

    # --------------------------------------------------------------------------
    # 2. Agent speaks first
    # --------------------------------------------------------------------------
    def test_agent_speaks_first(self):
        """Turn 1 opening greeting from support agent classifies as AGENT."""
        mock_client = self._make_mock_client("AGENT", 0.96, "Opening support greeting")
        turn = {"text": "Thank you for contacting industrial technical support. How may I assist?", "raw_speaker": "B"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "AGENT")
        self.assertGreaterEqual(result["confidence"], 0.80)

    # --------------------------------------------------------------------------
    # 3. A = Customer / B = Agent
    # --------------------------------------------------------------------------
    def test_speaker_a_customer_b_agent(self):
        """Preserves continuity when Speaker A is Customer and Speaker B is Agent."""
        tracker = SpeakerContinuityTracker()
        tracker.record_turn("A", "CUSTOMER", 0.95)
        tracker.record_turn("B", "AGENT", 0.92)

        summary = tracker.get_continuity_summary()
        self.assertIn("A", summary)
        self.assertIn("B", summary)
        self.assertIn("CUSTOMER", summary["A"])
        self.assertIn("AGENT", summary["B"])

    # --------------------------------------------------------------------------
    # 4. A = Agent / B = Customer
    # --------------------------------------------------------------------------
    def test_speaker_a_agent_b_customer(self):
        """Preserves continuity when Speaker A is Agent and Speaker B is Customer."""
        tracker = SpeakerContinuityTracker()
        tracker.record_turn("A", "AGENT", 0.95)
        tracker.record_turn("B", "CUSTOMER", 0.92)

        summary = tracker.get_continuity_summary()
        self.assertIn("A", summary)
        self.assertIn("B", summary)
        self.assertIn("AGENT", summary["A"])
        self.assertIn("CUSTOMER", summary["B"])

    # --------------------------------------------------------------------------
    # 5. Short "Okay"
    # --------------------------------------------------------------------------
    def test_short_okay_ambiguity(self):
        """Short ambiguous 'Okay' with low confidence falls back to UNKNOWN."""
        mock_client = self._make_mock_client("UNKNOWN", 0.40, "Ambiguous single-word acknowledgement")
        turn = {"text": "Okay", "raw_speaker": "A"}
        recent = [{"text": "Please check the breaker.", "role": "AGENT", "raw_speaker": "B"}]

        result = classify_speaker_role(current_turn=turn, recent_turns=recent, client=mock_client)

        self.assertEqual(result["role"], "UNKNOWN")
        self.assertLess(result["confidence"], 0.80)

    # --------------------------------------------------------------------------
    # 6. Short "Yes"
    # --------------------------------------------------------------------------
    def test_short_yes_ambiguity(self):
        """Short ambiguous 'Yes' with low confidence falls back to UNKNOWN."""
        mock_client = self._make_mock_client("UNKNOWN", 0.50, "Unclear affirmation")
        turn = {"text": "Yes", "raw_speaker": "B"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "UNKNOWN")

    # --------------------------------------------------------------------------
    # 7. Customer describes machine failure
    # --------------------------------------------------------------------------
    def test_customer_describes_machine_failure(self):
        """Detailed symptom report correctly classified as CUSTOMER."""
        mock_client = self._make_mock_client("CUSTOMER", 0.98, "Reports error E102 and fan stoppage")
        turn = {"text": "It is showing error E102 and the extraction fan has stopped spinning.", "raw_speaker": "A"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "CUSTOMER")
        self.assertGreaterEqual(result["confidence"], 0.80)

    # --------------------------------------------------------------------------
    # 8. Agent gives troubleshooting instruction
    # --------------------------------------------------------------------------
    def test_agent_gives_troubleshooting_instruction(self):
        """Procedural troubleshooting instructions classified as AGENT."""
        mock_client = self._make_mock_client("AGENT", 0.97, "Instructs breaker reset procedure")
        turn = {"text": "Please inspect circuit breaker CB-2 and replace filter cartridge WP4-FL-DE1.", "raw_speaker": "B"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "AGENT")
        self.assertGreaterEqual(result["confidence"], 0.80)

    # --------------------------------------------------------------------------
    # 9. Customer asks a question
    # --------------------------------------------------------------------------
    def test_customer_asks_question(self):
        """Customer diagnostic question ('What should I check?') classified as CUSTOMER."""
        mock_client = self._make_mock_client("CUSTOMER", 0.91, "Customer asking for troubleshooting advice")
        turn = {"text": "What should I check next on the WP-400?", "raw_speaker": "A"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "CUSTOMER")

    # --------------------------------------------------------------------------
    # 10. Agent asks a diagnostic question
    # --------------------------------------------------------------------------
    def test_agent_asks_diagnostic_question(self):
        """Agent diagnostic inquiry ('Is the warning LED illuminated?') classified as AGENT."""
        mock_client = self._make_mock_client("AGENT", 0.93, "Diagnostic verification question")
        turn = {"text": "Is the status LED on the control panel flashing amber or red?", "raw_speaker": "B"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "AGENT")

    # --------------------------------------------------------------------------
    # 11. Unknown role
    # --------------------------------------------------------------------------
    def test_explicit_unknown_role(self):
        """Model explicitly outputting UNKNOWN is preserved as UNKNOWN."""
        mock_client = self._make_mock_client("UNKNOWN", 0.30, "Insufficient conversational clues")
        turn = {"text": "Uh, hm, let's see.", "raw_speaker": "A"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "UNKNOWN")

    # --------------------------------------------------------------------------
    # 12. Low confidence
    # --------------------------------------------------------------------------
    def test_low_confidence_gate(self):
        """Model returning CUSTOMER with confidence < 0.80 gated to UNKNOWN."""
        mock_client = self._make_mock_client("CUSTOMER", 0.65, "Weak signal")
        turn = {"text": "I think so.", "raw_speaker": "A"}

        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "UNKNOWN")
        self.assertEqual(result["raw_role"], "CUSTOMER")
        self.assertEqual(result["confidence"], 0.65)

    # --------------------------------------------------------------------------
    # 13. Malformed Gemini response
    # --------------------------------------------------------------------------
    def test_malformed_gemini_response(self):
        """Malformed non-JSON output safely falls back to UNKNOWN with zero confidence."""
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON at all!"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        turn = {"text": "Machine stopped.", "raw_speaker": "A"}
        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "UNKNOWN")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("Malformed", result["rationale"])

    # --------------------------------------------------------------------------
    # 14. Gemini API error
    # --------------------------------------------------------------------------
    def test_gemini_api_error_handling(self):
        """Upstream network/quota exception caught safely; returns UNKNOWN without raising."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

        turn = {"text": "WP400 broken.", "raw_speaker": "A"}
        result = classify_speaker_role(current_turn=turn, recent_turns=[], client=mock_client)

        self.assertEqual(result["role"], "UNKNOWN")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("Gemini API error", result["rationale"])

    # --------------------------------------------------------------------------
    # 15. Agent turn does not enter sentiment
    # --------------------------------------------------------------------------
    def test_agent_turn_does_not_enter_sentiment(self):
        """Verifies coordinator excludes AGENT turns from sentiment analysis."""
        mock_sentiment = MagicMock()
        mock_rag = MagicMock(return_value=[])
        mock_role_fn = MagicMock(return_value={"role": "AGENT", "confidence": 0.95, "latency_ms": 10.0})

        coord = PipelineCoordinator(
            sentiment_fn=mock_sentiment,
            rag_fn=mock_rag,
            role_classifier_fn=mock_role_fn,
            async_downstream=False
        )

        state = ConversationState()
        segment = {"text": "Welcome to support. Let me help you.", "raw_speaker": "B"}
        state.add_final_segment(segment["text"], raw_speaker="B")

        coord._handle_speech_break(segment, state)

        mock_sentiment.assert_not_called()

    # --------------------------------------------------------------------------
    # 16. Agent turn does not enter RAG
    # --------------------------------------------------------------------------
    def test_agent_turn_does_not_enter_rag(self):
        """Verifies coordinator excludes AGENT turns from RAG search."""
        mock_sentiment = MagicMock()
        mock_rag = MagicMock(return_value=[])
        mock_role_fn = MagicMock(return_value={"role": "AGENT", "confidence": 0.95, "latency_ms": 10.0})

        coord = PipelineCoordinator(
            sentiment_fn=mock_sentiment,
            rag_fn=mock_rag,
            role_classifier_fn=mock_role_fn,
            async_downstream=False
        )

        state = ConversationState()
        segment = {"text": "I will check the troubleshooting guide.", "raw_speaker": "B"}
        state.add_final_segment(segment["text"], raw_speaker="B")

        coord._handle_speech_break(segment, state)

        mock_rag.assert_not_called()

    # --------------------------------------------------------------------------
    # 17. Unknown turn does not enter sentiment
    # --------------------------------------------------------------------------
    def test_unknown_turn_does_not_enter_sentiment(self):
        """Verifies coordinator excludes UNKNOWN turns from sentiment analysis."""
        mock_sentiment = MagicMock()
        mock_rag = MagicMock(return_value=[])
        mock_role_fn = MagicMock(return_value={"role": "UNKNOWN", "confidence": 0.45, "latency_ms": 10.0})

        coord = PipelineCoordinator(
            sentiment_fn=mock_sentiment,
            rag_fn=mock_rag,
            role_classifier_fn=mock_role_fn,
            async_downstream=False
        )

        state = ConversationState()
        segment = {"text": "Uh, okay.", "raw_speaker": "A"}
        state.add_final_segment(segment["text"], raw_speaker="A")

        coord._handle_speech_break(segment, state)

        mock_sentiment.assert_not_called()

    # --------------------------------------------------------------------------
    # 18. Unknown turn does not enter RAG
    # --------------------------------------------------------------------------
    def test_unknown_turn_does_not_enter_rag(self):
        """Verifies coordinator excludes UNKNOWN turns from RAG knowledge base search."""
        mock_sentiment = MagicMock()
        mock_rag = MagicMock(return_value=[])
        mock_role_fn = MagicMock(return_value={"role": "UNKNOWN", "confidence": 0.45, "latency_ms": 10.0})

        coord = PipelineCoordinator(
            sentiment_fn=mock_sentiment,
            rag_fn=mock_rag,
            role_classifier_fn=mock_role_fn,
            async_downstream=False
        )

        state = ConversationState()
        segment = {"text": "Right, okay.", "raw_speaker": "A"}
        state.add_final_segment(segment["text"], raw_speaker="A")

        coord._handle_speech_break(segment, state)

        mock_rag.assert_not_called()

    # --------------------------------------------------------------------------
    # 19. Customer context accumulates correctly
    # --------------------------------------------------------------------------
    def test_customer_context_accumulates_correctly(self):
        """Verifies multi-turn Customer utterances accumulate while Agent utterances are excluded."""
        state = ConversationState()
        state.add_final_segment("My WP400 has stopped.", speaker="CUSTOMER", raw_speaker="A")
        state.add_final_segment("Okay, let me check that.", speaker="AGENT", raw_speaker="B")
        state.add_final_segment("It is showing error E102.", speaker="CUSTOMER", raw_speaker="A")
        state.add_final_segment("Check the breaker.", speaker="AGENT", raw_speaker="B")

        cust_transcript = state.get_customer_transcript(customer_role="CUSTOMER")

        self.assertIn("My WP400 has stopped.", cust_transcript)
        self.assertIn("It is showing error E102.", cust_transcript)
        self.assertNotIn("Okay, let me check that.", cust_transcript)
        self.assertNotIn("Check the breaker.", cust_transcript)
        self.assertEqual(cust_transcript, "My WP400 has stopped. It is showing error E102.")

    # --------------------------------------------------------------------------
    # 20. New session clears role state
    # --------------------------------------------------------------------------
    def test_new_session_clears_role_state(self):
        """Verifies reset_session clears both conversation state and speaker continuity tracker."""
        coord = PipelineCoordinator(async_downstream=False)
        coord._continuity_tracker.record_turn("A", "CUSTOMER", 0.95)
        coord._continuity_tracker.record_turn("B", "AGENT", 0.95)
        coord.conversation_state.add_final_segment("Old session text", speaker="CUSTOMER")

        # Verify state is populated before reset
        self.assertGreater(len(coord._continuity_tracker._history), 0)
        self.assertEqual(len(coord.conversation_state.segments), 1)

        # Reset session
        coord.reset_session()

        # Both conversation state and continuity tracker must be completely empty
        self.assertEqual(len(coord.conversation_state.segments), 0)
        self.assertEqual(len(coord._continuity_tracker._history), 0)
        self.assertEqual(coord._continuity_tracker.get_continuity_summary(), {})

    # --------------------------------------------------------------------------
    # 21. Injected client is reused across multiple calls
    # --------------------------------------------------------------------------
    def test_client_reuse_across_multiple_calls(self):
        """Verifies that an injected Gemini client is reused across multiple classification turns."""
        mock_client = self._make_mock_client("CUSTOMER", 0.95, "Reused client test")

        with patch("src.speaker_role_classifier.genai.Client") as mock_genai_ctor:
            # Call 1
            res1 = classify_speaker_role(
                current_turn={"text": "Turn 1", "raw_speaker": "A"},
                client=mock_client
            )
            # Call 2
            res2 = classify_speaker_role(
                current_turn={"text": "Turn 2", "raw_speaker": "B"},
                client=mock_client
            )
            # Call 3
            res3 = classify_speaker_role(
                current_turn={"text": "Turn 3", "raw_speaker": "A"},
                client=mock_client
            )

            # Assert genai.Client was NEVER instantiated because mock_client was reused
            mock_genai_ctor.assert_not_called()
            # Assert generate_content was called exactly 3 times on the single injected mock_client
            self.assertEqual(mock_client.models.generate_content.call_count, 3)
            self.assertEqual(res1["role"], "CUSTOMER")
            self.assertEqual(res2["role"], "CUSTOMER")
            self.assertEqual(res3["role"], "CUSTOMER")

    # --------------------------------------------------------------------------
    # 22. get_shared_gemini_client caching behavior
    # --------------------------------------------------------------------------
    def test_get_shared_gemini_client_caching(self):
        """Verifies get_shared_gemini_client creates once and returns the same client on repeated calls."""
        with patch("src.speaker_role_classifier.genai.Client") as mock_genai_ctor:
            fake_client = MagicMock()
            mock_genai_ctor.return_value = fake_client

            client_a = get_shared_gemini_client(api_key="test_shared_key_12345")
            client_b = get_shared_gemini_client(api_key="test_shared_key_12345")

            self.assertIs(client_a, client_b)
            self.assertEqual(mock_genai_ctor.call_count, 1)

    # --------------------------------------------------------------------------
    # 23. PipelineCoordinator passes persistent client into classifier
    # --------------------------------------------------------------------------
    def test_pipeline_coordinator_passes_persistent_client(self):
        """Verifies PipelineCoordinator injects its persistent client into role_classifier_fn."""
        mock_client = MagicMock()
        mock_role_fn = MagicMock(return_value={"role": "CUSTOMER", "confidence": 0.95, "latency_ms": 10.0})

        coord = PipelineCoordinator(
            gemini_client=mock_client,
            role_classifier_fn=mock_role_fn,
            sentiment_fn=MagicMock(return_value={"sentiment": "Neutral", "confidence": 0.9}),
            category_fn=MagicMock(return_value={"category": "Machine Operation Issues", "confidence": 0.9}),
            rag_fn=MagicMock(return_value=[]),
            suggestion_fn=MagicMock(return_value={"suggestion": "Test", "confidence": 0.9, "is_grounded": True, "sources": []}),
            async_downstream=False
        )

        state = ConversationState()
        segment = {"text": "Machine stopped.", "raw_speaker": "A"}
        state.add_final_segment(segment["text"], raw_speaker="A")

        coord._handle_speech_break(segment, state)

        # Verify mock_role_fn was called with client=mock_client
        mock_role_fn.assert_called_once()
        _, kwargs = mock_role_fn.call_args
        self.assertIn("client", kwargs)
        self.assertIs(kwargs["client"], mock_client)


if __name__ == "__main__":
    unittest.main()

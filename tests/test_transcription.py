"""
tests/test_transcription.py

Unit tests for Phase 1 - Slice 1:
Validates AudioCapture, ConversationState, and Speech-Break Trigger Logic
WITHOUT requiring an active AssemblyAI API key or network calls.
"""

import unittest
from unittest.mock import MagicMock
from src.transcription import ConversationState, RealtimeTranscriber
from src.audio_capture import AudioCapture


class TestConversationState(unittest.TestCase):

    def setUp(self):
        self.state = ConversationState(session_id="test_session_123")

    def test_initial_state(self):
        self.assertEqual(self.state.session_id, "test_session_123")
        self.assertEqual(len(self.state.segments), 0)
        self.assertIsNone(self.state.current_partial)
        self.assertIsNone(self.state.current_speaker)
        self.assertEqual(self.state.get_full_transcript(), "")

    def test_partial_transcript_handling(self):
        self.state.set_partial("Customer reporting issue", speaker="customer")
        self.assertEqual(self.state.current_partial, "Customer reporting issue")
        self.assertEqual(self.state.current_speaker, "customer")
        # Partials should not be counted in finalized segments
        self.assertEqual(len(self.state.segments), 0)
        self.assertEqual(self.state.get_full_transcript(), "")

    def test_final_segment_and_speech_break_resolution(self):
        self.state.set_partial("Customer reporting", speaker="customer")
        segment = self.state.add_final_segment(
            text="Customer reporting saw blade wobble on WP-400.",
            speaker="customer",
            confidence=0.97
        )

        # Segment verification
        self.assertEqual(segment["text"], "Customer reporting saw blade wobble on WP-400.")
        self.assertEqual(segment["speaker"], "customer")
        self.assertEqual(segment["confidence"], 0.97)
        self.assertTrue(segment["is_final"])

        # State should be updated and partial reset
        self.assertEqual(len(self.state.segments), 1)
        self.assertIsNone(self.state.current_partial)
        self.assertIsNone(self.state.current_speaker)
        self.assertEqual(self.state.get_full_transcript(), "Customer reporting saw blade wobble on WP-400.")

    def test_speaker_metadata_and_formatting(self):
        self.state.add_final_segment("I have an error E-101 on my machine.", speaker="customer")
        self.state.add_final_segment("Please check the compressed-air regulator.", speaker="agent")

        formatted = self.state.get_full_transcript(include_speaker=True)
        self.assertIn("[customer] I have an error E-101 on my machine.", formatted)
        self.assertIn("[agent] Please check the compressed-air regulator.", formatted)

    def test_recent_context_retrieval(self):
        for i in range(5):
            self.state.add_final_segment(f"Message {i}", speaker="user")

        recent = self.state.get_recent_context(max_segments=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["text"], "Message 3")
        self.assertEqual(recent[1]["text"], "Message 4")

    def test_clear_state(self):
        self.state.add_final_segment("Message", speaker="user")
        self.state.clear()
        self.assertEqual(len(self.state.segments), 0)
        self.assertIsNone(self.state.current_partial)


class TestSpeechBreakLogic(unittest.TestCase):

    def test_speech_break_hook_triggered_on_end_of_turn(self):
        hook_called = False
        captured_segment = None
        captured_state = None

        def custom_speech_break_hook(seg, state):
            nonlocal hook_called, captured_segment, captured_state
            hook_called = True
            captured_segment = seg
            captured_state = state

        transcriber = RealtimeTranscriber(
            api_key="mock_key",
            on_speech_break=custom_speech_break_hook
        )

        # 1. Simulate interim turn event (end_of_turn = False)
        mock_interim_event = MagicMock()
        mock_interim_event.transcript = "Machine stopped working"
        mock_interim_event.end_of_turn = False
        mock_interim_event.speaker_label = "customer"

        transcriber._on_turn(None, mock_interim_event)
        self.assertFalse(hook_called, "Speech break hook should NOT trigger on partial transcript.")
        self.assertEqual(transcriber.conversation_state.current_partial, "Machine stopped working")

        # 2. Simulate final turn event (end_of_turn = True) -> Speech Break!
        mock_final_event = MagicMock()
        mock_final_event.transcript = "Machine stopped working after alarm E-102 appeared."
        mock_final_event.end_of_turn = True
        mock_final_event.speaker_label = "customer"
        mock_final_event.end_of_turn_confidence = 0.99

        transcriber._on_turn(None, mock_final_event)
        self.assertTrue(hook_called, "Speech break hook MUST trigger on end_of_turn.")
        self.assertEqual(captured_segment["text"], "Machine stopped working after alarm E-102 appeared.")
        self.assertEqual(captured_segment["speaker"], "customer")
        self.assertIn("alarm E-102", captured_state.get_full_transcript())


class TestAudioCaptureComponent(unittest.TestCase):

    def test_audio_capture_start_read_stop_lifecycle(self):
        capture = AudioCapture(sample_rate=16000, channels=1, chunk_size=1024)
        try:
            capture.start()
            self.assertTrue(capture.is_running)

            # Read one PCM chunk
            chunk = capture.read_chunk(timeout=1.0)
            self.assertIsNotNone(chunk, "Audio chunk should be received from microphone.")
            self.assertIsInstance(chunk, bytes)
            # 1024 samples * 2 bytes/sample (16-bit) = 2048 bytes
            self.assertEqual(len(chunk), 1024 * 2)

        finally:
            capture.stop()
            self.assertFalse(capture.is_running)


class TestSpeakerRoleMapping(unittest.TestCase):

    def test_assemblyai_speaker_labels_preserved_and_mapped(self):
        """Verifies AssemblyAI A/B labels are preserved and mapped to Customer/Agent."""
        from src.transcription import map_speaker_to_role

        # Test mapping
        self.assertEqual(map_speaker_to_role("A"), "Customer")
        self.assertEqual(map_speaker_to_role("B"), "Agent")
        self.assertEqual(map_speaker_to_role("Speaker A"), "Customer")
        self.assertEqual(map_speaker_to_role("Speaker B"), "Agent")

        # Test state preserves raw_speaker and maps role
        state = ConversationState()
        seg_a = state.add_final_segment("I have a problem.", raw_speaker="A")
        self.assertEqual(seg_a["speaker"], "Customer")
        self.assertEqual(seg_a["raw_speaker"], "A")
        self.assertEqual(seg_a["speaker_label"], "A")

        seg_b = state.add_final_segment("How can I assist?", raw_speaker="B")
        self.assertEqual(seg_b["speaker"], "Agent")
        self.assertEqual(seg_b["raw_speaker"], "B")
        self.assertEqual(seg_b["speaker_label"], "B")

    def test_customer_only_transcript_filtering(self):
        """Verifies get_customer_transcript extracts ONLY Customer speech."""
        state = ConversationState()
        state.add_final_segment("Customer message 1", speaker="Customer")
        state.add_final_segment("Agent message 1", speaker="Agent")
        state.add_final_segment("Customer message 2", speaker="Customer")
        state.add_final_segment("Agent message 2", speaker="Agent")

        full = state.get_full_transcript()
        self.assertIn("Customer message 1", full)
        self.assertIn("Agent message 1", full)

        cust = state.get_customer_transcript()
        self.assertEqual(cust, "Customer message 1 Customer message 2")
        self.assertNotIn("Agent", cust)

    def test_turn_event_raw_speaker_preservation(self):
        """
        Verifies that RealtimeTranscriber._on_turn preserves raw speaker_label
        from incoming AssemblyAI TurnEvent before role mapping:
        A -> Customer (raw_speaker='A')
        B -> Agent (raw_speaker='B')
        """
        from unittest.mock import MagicMock
        transcriber = RealtimeTranscriber(api_key="test_dummy_key")
        captured_segments = []
        transcriber.on_speech_break = lambda seg, st: captured_segments.append(seg)

        # 1. TurnEvent with raw speaker_label 'A'
        event_a = MagicMock()
        event_a.transcript = "Customer issue with WP400."
        event_a.speaker_label = "A"
        event_a.end_of_turn = True
        event_a.end_of_turn_confidence = 0.95

        transcriber._on_turn(None, event_a)

        self.assertEqual(len(captured_segments), 1)
        self.assertEqual(captured_segments[0]["raw_speaker"], "A")
        self.assertEqual(captured_segments[0]["speaker"], "Customer")
        self.assertEqual(captured_segments[0]["speaker_label"], "A")

        # 2. TurnEvent with raw speaker_label 'B'
        event_b = MagicMock()
        event_b.transcript = "Agent troubleshooting response."
        event_b.speaker_label = "B"
        event_b.end_of_turn = True
        event_b.end_of_turn_confidence = 0.92

        transcriber._on_turn(None, event_b)

        self.assertEqual(len(captured_segments), 2)
        self.assertEqual(captured_segments[1]["raw_speaker"], "B")
        self.assertEqual(captured_segments[1]["speaker"], "Agent")
        self.assertEqual(captured_segments[1]["speaker_label"], "B")


class TestRecentCustomerTranscript(unittest.TestCase):
    """
    Unit tests for ConversationState.get_recent_customer_transcript().
    Validates bounded context retrieval, turn filtering, and history preservation.
    """

    def setUp(self):
        self.state = ConversationState()

    def test_a_empty_conversation(self):
        """A. Empty conversation returns empty string."""
        self.assertEqual(self.state.get_recent_customer_transcript(max_turns=4), "")

    def test_b_one_customer_turn(self):
        """B. Single customer turn returns that turn."""
        self.state.add_final_segment("My WP400 is broken.", speaker="CUSTOMER")
        result = self.state.get_recent_customer_transcript(max_turns=4)
        self.assertEqual(result, "My WP400 is broken.")

    def test_c_multiple_customer_turns_within_limit(self):
        """C. All customer turns returned when count <= max_turns."""
        self.state.add_final_segment("Turn 1", speaker="CUSTOMER")
        self.state.add_final_segment("Turn 2", speaker="CUSTOMER")
        self.state.add_final_segment("Turn 3", speaker="CUSTOMER")

        result = self.state.get_recent_customer_transcript(max_turns=4)
        self.assertEqual(result, "Turn 1 Turn 2 Turn 3")

    def test_d_more_than_max_turns(self):
        """D. Returns only latest max_turns when count > max_turns."""
        for i in range(1, 6):
            self.state.add_final_segment(f"Turn {i}", speaker="CUSTOMER")

        result = self.state.get_recent_customer_transcript(max_turns=3)
        self.assertEqual(result, "Turn 3 Turn 4 Turn 5")

    def test_e_mixed_customer_and_agent(self):
        """E. Mixed CUSTOMER + AGENT excludes all AGENT turns."""
        self.state.add_final_segment("Customer problem", speaker="CUSTOMER")
        self.state.add_final_segment("Agent greeting", speaker="AGENT")
        self.state.add_final_segment("Customer detail", speaker="CUSTOMER")
        self.state.add_final_segment("Agent instruction", speaker="AGENT")

        result = self.state.get_recent_customer_transcript(max_turns=4)
        self.assertEqual(result, "Customer problem Customer detail")
        self.assertNotIn("Agent greeting", result)
        self.assertNotIn("Agent instruction", result)

    def test_f_mixed_customer_and_unknown(self):
        """F. Mixed CUSTOMER + UNKNOWN excludes all UNKNOWN turns."""
        self.state.add_final_segment("Customer issue", speaker="CUSTOMER")
        self.state.add_final_segment("Um, okay", speaker="UNKNOWN")
        self.state.add_final_segment("Customer symptom", speaker="CUSTOMER")

        result = self.state.get_recent_customer_transcript(max_turns=4)
        self.assertEqual(result, "Customer issue Customer symptom")
        self.assertNotIn("Um, okay", result)

    def test_g_current_customer_turn_included(self):
        """G. Most recently added customer turn is always included."""
        self.state.add_final_segment("First statement", speaker="CUSTOMER")
        self.state.add_final_segment("Latest question", speaker="CUSTOMER")

        result = self.state.get_recent_customer_transcript(max_turns=1)
        self.assertEqual(result, "Latest question")

    def test_h_chronological_ordering(self):
        """H. Chronological order is strictly preserved (older before newer)."""
        self.state.add_final_segment("First", speaker="CUSTOMER")
        self.state.add_final_segment("Second", speaker="CUSTOMER")
        self.state.add_final_segment("Third", speaker="CUSTOMER")
        self.state.add_final_segment("Fourth", speaker="CUSTOMER")

        result = self.state.get_recent_customer_transcript(max_turns=3)
        self.assertEqual(result, "Second Third Fourth")

    def test_i_full_transcript_preservation(self):
        """I. self.segments remains completely intact after calling get_recent_customer_transcript."""
        self.state.add_final_segment("Cust 1", speaker="CUSTOMER")
        self.state.add_final_segment("Agent 1", speaker="AGENT")
        self.state.add_final_segment("Cust 2", speaker="CUSTOMER")

        initial_len = len(self.state.segments)
        bounded = self.state.get_recent_customer_transcript(max_turns=1)

        self.assertEqual(bounded, "Cust 2")
        self.assertEqual(len(self.state.segments), initial_len)
        self.assertEqual(self.state.segments[0]["text"], "Cust 1")
        self.assertEqual(self.state.segments[1]["text"], "Agent 1")
        self.assertEqual(self.state.segments[2]["text"], "Cust 2")

    def test_j_regression_case_prompt_example(self):
        """
        J. Regression test case from specification:
        4 customer turns, 3 agent turns, max_turns=3.
        Must return turns 2, 3, 4 only; turn 1 and agent turns excluded.
        """
        self.state.add_final_segment("My WP400 has stopped working.", speaker="CUSTOMER")
        self.state.add_final_segment("Okay, let me help you.", speaker="AGENT")
        self.state.add_final_segment("It is showing error E102.", speaker="CUSTOMER")
        self.state.add_final_segment("Please check the extraction breaker.", speaker="AGENT")
        self.state.add_final_segment("The extraction fan isn't running.", speaker="CUSTOMER")
        self.state.add_final_segment("Check the filter.", speaker="AGENT")
        self.state.add_final_segment("The filter is blocked.", speaker="CUSTOMER")

        result = self.state.get_recent_customer_transcript(max_turns=3)

        expected = "It is showing error E102. The extraction fan isn't running. The filter is blocked."
        self.assertEqual(result, expected)
        self.assertNotIn("Okay, let me help you.", result)
        self.assertNotIn("Please check the extraction breaker.", result)
        self.assertNotIn("Check the filter.", result)
        self.assertNotIn("My WP400 has stopped working.", result)

        # Full transcript still contains all 7 turns
        self.assertEqual(len(self.state.segments), 7)


if __name__ == "__main__":
    unittest.main()

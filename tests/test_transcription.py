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


if __name__ == "__main__":
    unittest.main()

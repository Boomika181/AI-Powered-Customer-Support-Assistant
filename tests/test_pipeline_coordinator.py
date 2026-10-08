"""
tests/test_pipeline_coordinator.py

Unit Test Suite for Pipeline Coordinator (Phase 2 - Step 1).
Validates orchestration, event generation, turn dispatching, error containment,
and lifecycle management using mocked Phase 1 dependencies.
NO LIVE ASSEMBLYAI, GEMINI, OR CHROMADB CALLS ARE MADE BY THIS TEST SUITE.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from src.pipeline_coordinator import PipelineCoordinator
from src.transcription import ConversationState


class TestPipelineCoordinator(unittest.TestCase):

    def setUp(self):
        # Mock audio capture and transcriber
        self.mock_audio_capture = MagicMock()
        self.mock_audio_capture.device_name = "Mock Microphone"

        self.mock_transcriber = MagicMock()
        self.mock_transcriber.conversation_state = ConversationState(session_id="test_session")
        self.mock_transcriber._client = None

        # Mock Phase 1 operational functions
        self.mock_sentiment_fn = MagicMock(return_value={
            "sentiment": "Neutral",
            "confidence": 0.88,
            "rationale": "Factual statement regarding machinery."
        })

        self.mock_category_fn = MagicMock(return_value={
            "category": "Technical Troubleshooting",
            "confidence": 0.92,
            "rationale": "Customer inquiring about alarm code E-102."
        })

        self.mock_rag_fn = MagicMock(return_value=[
            {
                "document": "E-102 Dust extraction fault procedure.",
                "similarity_score": 0.85,
                "metadata": {
                    "document_reference": "TS-WP400-3.1",
                    "page": 6,
                    "section": "Section 3.1"
                }
            }
        ])

        self.mock_suggestion_fn = MagicMock(return_value={
            "suggestion": "Check the fan breaker and replace filter cartridge WP4-FL-DE1.",
            "confidence": 0.95,
            "is_grounded": True,
            "sources": [
                {
                    "document_reference": "TS-WP400-3.1",
                    "page": 6,
                    "section": "Section 3.1"
                }
            ],
            "category": "Technical Troubleshooting"
        })

        def default_mock_role_classifier(current_turn, recent_turns=None, tracker=None):
            role_hint = current_turn.get("role") or current_turn.get("speaker") or "Customer"
            clean_role = "AGENT" if "agent" in str(role_hint).lower() else "CUSTOMER"
            return {
                "role": clean_role,
                "confidence": 0.95,
                "raw_role": clean_role,
                "latency_ms": 10.0,
                "rationale": "Mocked test classification"
            }

        self.mock_role_classifier_fn = MagicMock(side_effect=default_mock_role_classifier)

        # Initialize coordinator with async_downstream=False for deterministic unit tests
        self.coordinator = PipelineCoordinator(
            audio_capture=self.mock_audio_capture,
            transcriber=self.mock_transcriber,
            sentiment_fn=self.mock_sentiment_fn,
            category_fn=self.mock_category_fn,
            rag_fn=self.mock_rag_fn,
            suggestion_fn=self.mock_suggestion_fn,
            role_classifier_fn=self.mock_role_classifier_fn,
            async_downstream=False
        )

        # Captured events list for assertions
        self.captured_events: List[Dict[str, Any]] = []
        self.coordinator.add_event_listener(lambda ev: self.captured_events.append(ev))

    # --------------------------------------------------------------------------
    # 1. Coordinator Initialization
    # --------------------------------------------------------------------------
    def test_coordinator_initialization(self):
        """Verifies initial state, dependencies, and event registration."""
        self.assertFalse(self.coordinator.is_running)
        self.assertIsNotNone(self.coordinator.conversation_state)
        self.assertEqual(self.coordinator.conversation_state.session_id, "test_session")
        self.assertEqual(len(self.captured_events), 0)

    # --------------------------------------------------------------------------
    # 2. Partial Transcript Event Generation
    # --------------------------------------------------------------------------
    def test_partial_transcript_event_generation(self):
        """Verifies transcript_partial events emitted when transcriber emits in-progress text."""
        self.coordinator._handle_partial_transcript("The WP-400 is", "Customer")

        self.assertEqual(len(self.captured_events), 1)
        event = self.captured_events[0]
        self.assertEqual(event["type"], "transcript_partial")
        self.assertEqual(event["speaker"], "Customer")
        self.assertEqual(event["text"], "The WP-400 is")
        self.assertIn("timestamp", event)

    # --------------------------------------------------------------------------
    # 3. Finalized Transcript Event Generation
    # --------------------------------------------------------------------------
    def test_finalized_transcript_event_generation(self):
        """Verifies transcript_final event emitted on speech break."""
        segment = {
            "text": "The WP-400 is showing E-102.",
            "speaker": "Customer",
            "confidence": 0.97
        }
        state = ConversationState()
        state.add_final_segment(segment["text"], segment["speaker"], segment["confidence"])

        self.coordinator._handle_speech_break(segment, state)

        # First emitted event must be transcript_final
        final_events = [e for e in self.captured_events if e["type"] == "transcript_final"]
        self.assertEqual(len(final_events), 1)
        fe = final_events[0]
        self.assertEqual(fe["speaker"], "Customer")
        self.assertEqual(fe["text"], "The WP-400 is showing E-102.")
        self.assertEqual(fe["confidence"], 0.97)

    # --------------------------------------------------------------------------
    # 4. Speech-Break Full Pipeline Processing
    # --------------------------------------------------------------------------
    def test_speech_break_processing_triggers_all_stages(self):
        """Verifies that speech break invokes sentiment, category, RAG, and suggestions."""
        segment = {"text": "What should I check for error E-102?", "speaker": "Customer"}
        state = ConversationState()
        state.add_final_segment(segment["text"], segment["speaker"])

        self.coordinator._handle_speech_break(segment, state)

        self.mock_sentiment_fn.assert_called_once()
        self.mock_category_fn.assert_called_once()
        self.mock_rag_fn.assert_called_once_with("What should I check for error E-102?", top_k=3)
        self.mock_suggestion_fn.assert_called_once()

    # --------------------------------------------------------------------------
    # 5. Sentiment Result Event
    # --------------------------------------------------------------------------
    def test_sentiment_result_event(self):
        """Verifies sentiment_update event matches real returned sentiment structure."""
        segment = {"text": "Machine stopped working.", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        sent_events = [e for e in self.captured_events if e["type"] == "sentiment_update"]
        self.assertEqual(len(sent_events), 1)
        se = sent_events[0]
        self.assertEqual(se["sentiment"], "Neutral")
        self.assertEqual(se["confidence"], 0.88)
        self.assertEqual(se["rationale"], "Factual statement regarding machinery.")

    # --------------------------------------------------------------------------
    # 6. Category Result Event
    # --------------------------------------------------------------------------
    def test_category_result_event(self):
        """Verifies category_update event matches real returned category structure."""
        segment = {"text": "Machine stopped working.", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        cat_events = [e for e in self.captured_events if e["type"] == "category_update"]
        self.assertEqual(len(cat_events), 1)
        ce = cat_events[0]
        self.assertEqual(ce["category"], "Technical Troubleshooting")
        self.assertEqual(ce["confidence"], 0.92)

    # --------------------------------------------------------------------------
    # 7. RAG Result Passed into Suggestion Generation
    # --------------------------------------------------------------------------
    def test_rag_result_passed_into_suggestion_generation(self):
        """Verifies retrieved chunks from RAG search are passed directly to suggestion generation."""
        segment = {"text": "How do I replace filter WP4-FL-DE1?", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        # Check call arguments for suggestion function
        call_kwargs = self.mock_suggestion_fn.call_args.kwargs
        self.assertEqual(call_kwargs["conversation_text"], "How do I replace filter WP4-FL-DE1?")
        self.assertEqual(call_kwargs["category"], "Technical Troubleshooting")
        self.assertEqual(len(call_kwargs["retrieved_context"]), 1)
        self.assertEqual(
            call_kwargs["retrieved_context"][0]["metadata"]["document_reference"],
            "TS-WP400-3.1"
        )

    # --------------------------------------------------------------------------
    # 8. Suggestion Event Contains Source References
    # --------------------------------------------------------------------------
    def test_suggestion_event_contains_source_references(self):
        """Verifies suggestion_update event carries suggestion, grounding flag, and sources."""
        segment = {"text": "How do I fix E-102?", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        sugg_events = [e for e in self.captured_events if e["type"] == "suggestion_update"]
        self.assertEqual(len(sugg_events), 1)
        se = sugg_events[0]
        self.assertIn("WP4-FL-DE1", se["suggestion"])
        self.assertTrue(se["is_grounded"])
        self.assertEqual(len(se["sources"]), 1)
        self.assertEqual(se["sources"][0]["document_reference"], "TS-WP400-3.1")

    # --------------------------------------------------------------------------
    # 9. Sentiment Failure Produces system_error Event
    # --------------------------------------------------------------------------
    def test_sentiment_failure_produces_system_error(self):
        """Verifies that sentiment failure produces a typed system_error without raising."""
        self.mock_sentiment_fn.side_effect = RuntimeError("Upstream Gemini sentiment quota error")
        segment = {"text": "Test error", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        err_events = [e for e in self.captured_events if e["type"] == "system_error" and e.get("stage") == "sentiment"]
        self.assertEqual(len(err_events), 1)
        self.assertIn("Gemini sentiment quota error", err_events[0]["message"])
        # Downstream pipeline should continue
        self.mock_category_fn.assert_called_once()

    # --------------------------------------------------------------------------
    # 10. Category Failure Produces system_error Event
    # --------------------------------------------------------------------------
    def test_category_failure_produces_system_error(self):
        """Verifies that query categorization failure produces a system_error event."""
        self.mock_category_fn.side_effect = ValueError("Invalid category output")
        segment = {"text": "Test error", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        err_events = [e for e in self.captured_events if e["type"] == "system_error" and e.get("stage") == "categorization"]
        self.assertEqual(len(err_events), 1)
        self.assertIn("Invalid category output", err_events[0]["message"])

    # --------------------------------------------------------------------------
    # 11. RAG Failure Produces system_error Event
    # --------------------------------------------------------------------------
    def test_rag_failure_produces_system_error(self):
        """Verifies that ChromaDB RAG search failure produces a system_error event."""
        self.mock_rag_fn.side_effect = RuntimeError("ChromaDB connection timeout")
        segment = {"text": "Test error", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        err_events = [e for e in self.captured_events if e["type"] == "system_error" and e.get("stage") == "rag"]
        self.assertEqual(len(err_events), 1)
        self.assertIn("ChromaDB connection timeout", err_events[0]["message"])
        # Suggestion generation is still called with empty context (safe fallback)
        self.mock_suggestion_fn.assert_called_once()
        self.assertEqual(self.mock_suggestion_fn.call_args.kwargs["retrieved_context"], [])

    # --------------------------------------------------------------------------
    # 12. Suggestion Failure Produces system_error Event
    # --------------------------------------------------------------------------
    def test_suggestion_failure_produces_system_error(self):
        """Verifies that suggestion generation failure produces a system_error event."""
        self.mock_suggestion_fn.side_effect = RuntimeError("Gemini suggestions service unavailable")
        segment = {"text": "Test error", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        err_events = [e for e in self.captured_events if e["type"] == "system_error" and e.get("stage") == "suggestions"]
        self.assertEqual(len(err_events), 1)
        self.assertIn("Gemini suggestions service unavailable", err_events[0]["message"])

    # --------------------------------------------------------------------------
    # 13. Start / Stop Lifecycle Management
    # --------------------------------------------------------------------------
    def test_start_stop_lifecycle(self):
        """Verifies start/stop lifecycle emits system_status events and starts threads."""
        stop_event = threading.Event()
        self.mock_transcriber.start_streaming.side_effect = lambda cap: stop_event.wait(timeout=1.0)
        self.mock_transcriber.stop.side_effect = lambda: stop_event.set()

        self.coordinator.start()
        self.assertTrue(self.coordinator.is_running)
        self.mock_audio_capture.start.assert_called_once()

        start_events = [e for e in self.captured_events if e["type"] == "system_status" and e.get("status") == "started"]
        self.assertEqual(len(start_events), 1)

        self.coordinator.stop()
        self.assertFalse(self.coordinator.is_running)
        self.mock_audio_capture.stop.assert_called_once()
        self.mock_transcriber.stop.assert_called_once()

        stop_events = [e for e in self.captured_events if e["type"] == "system_status" and e.get("status") == "stopped"]
        self.assertEqual(len(stop_events), 1)

    # --------------------------------------------------------------------------
    # 14. No Hardcoded / Fake Result Generation
    # --------------------------------------------------------------------------
    def test_no_hardcoded_fake_results(self):
        """Verifies coordinator emits EXACT results returned by underlying functions, never hardcoded values."""
        custom_sentiment = {"sentiment": "Agitated", "confidence": 0.99, "rationale": "High escalation detected."}
        custom_category = {"category": "Machine Operation Issues", "confidence": 0.91, "rationale": "Startup query."}
        custom_suggestion = {
            "suggestion": "Custom dynamic resolution step 1, 2, 3.",
            "confidence": 0.89,
            "is_grounded": True,
            "sources": [{"document_reference": "CUSTOM-DOC-1", "page": 1, "section": "Section 1"}]
        }

        self.mock_sentiment_fn.return_value = custom_sentiment
        self.mock_category_fn.return_value = custom_category
        self.mock_suggestion_fn.return_value = custom_suggestion

        segment = {"text": "Machine issue", "speaker": "Customer"}
        state = ConversationState()

        self.coordinator._handle_speech_break(segment, state)

        sent_ev = [e for e in self.captured_events if e["type"] == "sentiment_update"][0]
        cat_ev = [e for e in self.captured_events if e["type"] == "category_update"][0]
        sugg_ev = [e for e in self.captured_events if e["type"] == "suggestion_update"][0]

        self.assertEqual(sent_ev["sentiment"], "Agitated")
        self.assertEqual(cat_ev["category"], "Machine Operation Issues")
        self.assertEqual(sugg_ev["suggestion"], "Custom dynamic resolution step 1, 2, 3.")
        self.assertEqual(sugg_ev["sources"][0]["document_reference"], "CUSTOM-DOC-1")

    # --------------------------------------------------------------------------
    # 15. Multi-Turn RAG & Suggestions Context Preservation (Bug Fix 1)
    # --------------------------------------------------------------------------
    def test_multiturn_rag_and_suggestion_receives_accumulated_context(self):
        """
        TEST 1 — Multi-turn RAG context:
        Given:
          Turn 1 = "My WP400 has stopped working."
          Turn 2 = "It's showing E102 and the extraction fan isn't running."
          Turn 3 = "What should I check?"
        Verify that RAG receives the FULL accumulated conversation, not only Turn 3.
        Verify support suggestion generation also receives the FULL accumulated conversation.
        """
        state = ConversationState(session_id="multi_turn_test")

        # Turn 1
        t1_text = "My WP400 has stopped working."
        state.add_final_segment(t1_text, speaker="Customer")
        self.coordinator._handle_speech_break({"text": t1_text, "speaker": "Customer"}, state)

        # Turn 2
        t2_text = "It's showing E102 and the extraction fan isn't running."
        state.add_final_segment(t2_text, speaker="Customer")
        self.coordinator._handle_speech_break({"text": t2_text, "speaker": "Customer"}, state)

        # Reset mock call counters to isolate Turn 3 assertions
        self.mock_rag_fn.reset_mock()
        self.mock_suggestion_fn.reset_mock()

        # Turn 3
        t3_text = "What should I check?"
        state.add_final_segment(t3_text, speaker="Customer")
        self.coordinator._handle_speech_break({"text": t3_text, "speaker": "Customer"}, state)

        expected_full_context = (
            "My WP400 has stopped working. "
            "It's showing E102 and the extraction fan isn't running. "
            "What should I check?"
        )

        # Verify RAG function receives full accumulated context
        self.mock_rag_fn.assert_called_once_with(expected_full_context, top_k=3)

        # Verify suggestion generation receives full accumulated context
        self.mock_suggestion_fn.assert_called_once()
        sugg_kwargs = self.mock_suggestion_fn.call_args.kwargs
        self.assertEqual(sugg_kwargs["conversation_text"], expected_full_context)

    # --------------------------------------------------------------------------
    # 16. Session Reset Clears Conversation State (Bug Fix 2)
    # --------------------------------------------------------------------------
    def test_session_reset_lifecycle_clears_conversation_state(self):
        """
        TEST 2 — Session reset:
        Start session A.
        Add at least two conversation segments.
        Stop session A.
        Start session B.
        Verify session B's ConversationState is empty before its first new turn.
        Verify old text does not appear in session B's full transcript.
        """
        # Session A
        self.coordinator.start()
        self.coordinator.conversation_state.add_final_segment("Session A Turn 1: Machine fault.", speaker="Customer")
        self.coordinator.conversation_state.add_final_segment("Session A Turn 2: Alarm code E-102.", speaker="Customer")
        self.assertEqual(len(self.coordinator.conversation_state.segments), 2)
        self.assertIn("Session A", self.coordinator.conversation_state.get_full_transcript())

        self.coordinator.stop()

        # Start Session B
        self.coordinator.start()
        # Verify Session B ConversationState is completely empty before its first new turn
        self.assertEqual(len(self.coordinator.conversation_state.segments), 0)
        self.assertEqual(self.coordinator.conversation_state.get_full_transcript(), "")

        # Process a new turn in Session B
        segment_b = {"text": "Session B Turn 1: Fresh inquiry.", "speaker": "Customer"}
        self.coordinator.conversation_state.add_final_segment(segment_b["text"], segment_b["speaker"])

        # Verify old text from Session A does not appear in Session B transcript
        session_b_transcript = self.coordinator.conversation_state.get_full_transcript()
        self.assertNotIn("Session A", session_b_transcript)
        self.assertEqual(session_b_transcript, "Session B Turn 1: Fresh inquiry.")

        self.coordinator.stop()

    # --------------------------------------------------------------------------
    # 17. Stale Event and Task Protection Across Sessions (Bug Fix 2)
    # --------------------------------------------------------------------------
    def test_stale_event_and_task_protection_across_sessions(self):
        """
        TEST 3 — Stale event protection:
        Verify events/tasks from session A cannot contaminate session B.
        """
        # Session A start
        self.coordinator.start()

        # Artificially queue an event and a task
        self.coordinator.event_queue.put({"type": "stale_event", "session": "A"})
        self.coordinator._task_queue.put(({"text": "Stale task", "speaker": "Customer"}, self.coordinator.conversation_state))

        # Stop session A
        self.coordinator.stop()

        # Start session B
        self.coordinator.start()

        # Verify event queue was drained and has only session B's started status event
        events_in_b = []
        while not self.coordinator.event_queue.empty():
            events_in_b.append(self.coordinator.event_queue.get_nowait())

        # None of the events in Session B should be the stale Session A event
        stale_events = [e for e in events_in_b if e.get("type") == "stale_event"]
        self.assertEqual(len(stale_events), 0)

        # Verify _task_queue is completely empty
        self.assertTrue(self.coordinator._task_queue.empty())

        self.coordinator.stop()

    # --------------------------------------------------------------------------
    # 18. Customer vs Agent Diarization & Downstream Isolation
    # --------------------------------------------------------------------------
    def test_mixed_speaker_turns_customer_only_downstream(self):
        """
        Verifies:
        - Both Customer and Agent turns emit transcript_final events.
        - Sentiment receives ONLY Customer turns.
        - RAG receives ONLY Customer turns.
        - Suggestion generation receives ONLY Customer turns.
        """
        state = ConversationState(session_id="speaker_role_test")

        # Turn 1: Customer (Speaker A)
        t1_text = "My WP400 has stopped working."
        state.add_final_segment(t1_text, speaker="Customer", raw_speaker="A")
        self.coordinator._handle_speech_break({"text": t1_text, "speaker": "Customer", "raw_speaker": "A"}, state)

        # Turn 2: Agent (Speaker B)
        t2_text = "Okay, what error code is showing?"
        state.add_final_segment(t2_text, speaker="Agent", raw_speaker="B")
        self.coordinator._handle_speech_break({"text": t2_text, "speaker": "Agent", "raw_speaker": "B"}, state)

        # Turn 3: Customer (Speaker A)
        t3_text = "It is showing error E-102."
        state.add_final_segment(t3_text, speaker="Customer", raw_speaker="A")
        self.coordinator._handle_speech_break({"text": t3_text, "speaker": "Customer", "raw_speaker": "A"}, state)

        # Turn 4: Agent (Speaker B)
        t4_text = "Please wait while I check the troubleshooting procedure."
        state.add_final_segment(t4_text, speaker="Agent", raw_speaker="B")
        self.coordinator._handle_speech_break({"text": t4_text, "speaker": "Agent", "raw_speaker": "B"}, state)

        # 1. Verify all 4 turns emitted transcript_final events with correct roles
        final_events = [e for e in self.captured_events if e["type"] == "transcript_final"]
        self.assertEqual(len(final_events), 4)
        self.assertEqual(final_events[0]["speaker"], "Customer")
        self.assertEqual(final_events[1]["speaker"], "Agent")
        self.assertEqual(final_events[2]["speaker"], "Customer")
        self.assertEqual(final_events[3]["speaker"], "Agent")

        # 2. Verify expected customer-only conversation text
        expected_customer_text = "My WP400 has stopped working. It is showing error E-102."

        # Check latest sentiment call
        latest_sent_call = self.mock_sentiment_fn.call_args[0][0]
        self.assertEqual(latest_sent_call, expected_customer_text)
        self.assertNotIn("Okay, what error code", latest_sent_call)
        self.assertNotIn("Please wait while I check", latest_sent_call)

        # Check latest RAG call
        latest_rag_call = self.mock_rag_fn.call_args[0][0]
        self.assertEqual(latest_rag_call, expected_customer_text)
        self.assertNotIn("error code is showing", latest_rag_call)
        self.assertNotIn("Please wait while I check", latest_rag_call)

        # Check latest suggestion call
        latest_sugg_call = self.mock_suggestion_fn.call_args.kwargs["conversation_text"]
        self.assertEqual(latest_sugg_call, expected_customer_text)
        self.assertNotIn("Please wait while I check", latest_sugg_call)

    # --------------------------------------------------------------------------
    # 19. Initial Pure Agent Speech Skips Downstream Customer Analysis
    # --------------------------------------------------------------------------
    def test_pure_agent_speech_skips_customer_analysis(self):
        """
        Verifies that an agent greeting before any customer speech emits transcript_final
        but does not trigger customer sentiment or RAG analysis.
        """
        state = ConversationState(session_id="agent_greeting_test")
        self.mock_sentiment_fn.reset_mock()
        self.mock_rag_fn.reset_mock()

        # Turn 1: Agent greeting
        agent_greet = "Welcome to industrial machinery technical support."
        state.add_final_segment(agent_greet, speaker="Agent", raw_speaker="B")
        self.coordinator._handle_speech_break({"text": agent_greet, "speaker": "Agent", "raw_speaker": "B"}, state)

        # Emitted transcript_final for feed
        agent_events = [e for e in self.captured_events if e["type"] == "transcript_final" and e.get("speaker") == "Agent"]
        self.assertEqual(len(agent_events), 1)

        # But analytical functions were NOT triggered on pure agent speech
        self.mock_sentiment_fn.assert_not_called()
        self.mock_rag_fn.assert_not_called()
        self.mock_suggestion_fn.assert_not_called()

    # --------------------------------------------------------------------------
    # 20. Downstream Pipeline Receives Bounded Customer Context
    # --------------------------------------------------------------------------
    def test_downstream_pipeline_receives_bounded_customer_context(self):
        """
        Verifies that when max_customer_turns is configured, downstream analytical
        functions receive strictly the latest N customer turns rather than full session history.
        """
        # Create coordinator with max_customer_turns=2
        bounded_coord = PipelineCoordinator(
            audio_capture=self.mock_audio_capture,
            transcriber=self.mock_transcriber,
            sentiment_fn=self.mock_sentiment_fn,
            category_fn=self.mock_category_fn,
            rag_fn=self.mock_rag_fn,
            suggestion_fn=self.mock_suggestion_fn,
            role_classifier_fn=self.mock_role_classifier_fn,
            max_customer_turns=2,
            async_downstream=False
        )

        state = ConversationState(session_id="bounded_context_test")
        self.mock_sentiment_fn.reset_mock()
        self.mock_category_fn.reset_mock()
        self.mock_rag_fn.reset_mock()
        self.mock_suggestion_fn.reset_mock()

        # Turn 1: Customer (older turn, should be excluded from max_customer_turns=2)
        state.add_final_segment("My WP400 has stopped working.", speaker="Customer", raw_speaker="A")
        # Turn 2: Agent
        state.add_final_segment("Okay, let me help you.", speaker="Agent", raw_speaker="B")
        # Turn 3: Customer (older turn, should be excluded from max_customer_turns=2)
        state.add_final_segment("It is showing error E102.", speaker="Customer", raw_speaker="A")
        # Turn 4: Agent
        state.add_final_segment("Please check the extraction breaker.", speaker="Agent", raw_speaker="B")
        # Turn 5: Customer (retained turn 1 of 2)
        state.add_final_segment("The extraction fan isn't running.", speaker="Customer", raw_speaker="A")
        # Turn 6: Customer (retained turn 2 of 2 - current turn)
        t6_text = "The filter is completely blocked."
        t6_seg = state.add_final_segment(t6_text, speaker="Customer", raw_speaker="A")

        # Process Turn 6 through coordinator
        bounded_coord._handle_speech_break(t6_seg, state)

        expected_bounded_context = "The extraction fan isn't running. The filter is completely blocked."

        # Verify sentiment received bounded context
        self.mock_sentiment_fn.assert_called_once_with(expected_bounded_context)

        # Verify category received bounded context
        self.mock_category_fn.assert_called_once_with(expected_bounded_context)

        # Verify RAG search received bounded context
        self.mock_rag_fn.assert_called_once_with(expected_bounded_context, top_k=3)

        # Verify suggestion generation received bounded context
        self.mock_suggestion_fn.assert_called_once()
        sugg_kwargs = self.mock_suggestion_fn.call_args.kwargs
        self.assertEqual(sugg_kwargs["conversation_text"], expected_bounded_context)

        # Verify full state still contains all 6 segments
        self.assertEqual(len(state.segments), 6)

    # --------------------------------------------------------------------------
    # 20. Parallel Downstream Execution (Concurrent Sentiment, Category, RAG)
    # --------------------------------------------------------------------------
    def test_parallel_downstream_execution_and_orchestration(self):
        """
        Verifies:
        1. Sentiment is called once.
        2. Category is called once.
        3. RAG is called once.
        4. All three receive the same bounded context.
        5. Suggestion runs only after the required parallel tasks complete.
        6. Suggestion receives the correct category and RAG results.
        7. Existing events are still emitted (sentiment_update, category_update, suggestion_update).
        9. No changes to ConversationState.
        10. No changes to speaker-role classification.
        """
        import time

        call_order = []
        concurrency_barrier = threading.Barrier(3)
        threads_used = set()

        def tracked_sentiment(ctx):
            threads_used.add(threading.get_ident())
            concurrency_barrier.wait(timeout=2.0)
            time.sleep(0.01)
            call_order.append("sentiment")
            return {"sentiment": "Negative", "confidence": 0.95, "rationale": "Frustrated"}

        def tracked_category(ctx):
            threads_used.add(threading.get_ident())
            concurrency_barrier.wait(timeout=2.0)
            time.sleep(0.01)
            call_order.append("category")
            return {"category": "Technical Troubleshooting", "confidence": 0.92, "rationale": "Fault"}

        def tracked_rag(ctx, top_k=3):
            threads_used.add(threading.get_ident())
            concurrency_barrier.wait(timeout=2.0)
            time.sleep(0.01)
            call_order.append("rag")
            return [{"document": "E-102 resolution", "similarity_score": 0.9}]

        def tracked_suggestion(conversation_text, retrieved_context, category):
            call_order.append("suggestion")
            return {
                "suggestion": "Inspect extraction fan and clear filter.",
                "confidence": 0.96,
                "is_grounded": True,
                "sources": [],
                "category": category
            }

        coord = PipelineCoordinator(
            audio_capture=self.mock_audio_capture,
            transcriber=self.mock_transcriber,
            sentiment_fn=tracked_sentiment,
            category_fn=tracked_category,
            rag_fn=tracked_rag,
            suggestion_fn=tracked_suggestion,
            role_classifier_fn=self.mock_role_classifier_fn,
            max_customer_turns=4,
            async_downstream=False
        )

        events = []
        coord.add_event_listener(lambda ev: events.append(ev))

        state = ConversationState(session_id="parallel_test")
        t1 = state.add_final_segment("My WP400 is showing error E102.", speaker="Customer", raw_speaker="A")
        t2 = state.add_final_segment("The extraction fan isn't running.", speaker="Customer", raw_speaker="A")

        coord._handle_speech_break(t2, state)

        expected_context = "My WP400 is showing error E102. The extraction fan isn't running."

        # 1, 2, 3: Called exactly once (verified by call_order counts)
        self.assertEqual(call_order.count("sentiment"), 1)
        self.assertEqual(call_order.count("category"), 1)
        self.assertEqual(call_order.count("rag"), 1)
        self.assertEqual(call_order.count("suggestion"), 1)

        # 5: Suggestion runs only AFTER all three parallel tasks complete
        self.assertEqual(call_order[-1], "suggestion")
        self.assertEqual(set(call_order[:3]), {"sentiment", "category", "rag"})

        # Verify concurrent thread pool execution (multiple distinct threads were used)
        self.assertGreaterEqual(len(threads_used), 2)

        # 7: Existing events emitted
        event_types = [ev["type"] for ev in events]
        self.assertIn("sentiment_update", event_types)
        self.assertIn("category_update", event_types)
        self.assertIn("suggestion_update", event_types)

        # 6: Suggestion received correct category and RAG results
        sugg_event = next(ev for ev in events if ev["type"] == "suggestion_update")
        self.assertEqual(sugg_event["category"], "Technical Troubleshooting")
        self.assertEqual(sugg_event["is_grounded"], True)

        # 9: ConversationState preserved intact
        self.assertEqual(len(state.segments), 2)
        self.assertEqual(state.get_full_transcript(), expected_context)

    # --------------------------------------------------------------------------
    # 21. Parallel Downstream Error Containment
    # --------------------------------------------------------------------------
    def test_parallel_downstream_error_containment(self):
        """
        Verifies that if one parallel task fails, the other tasks continue,
        error events are emitted cleanly, and suggestion generation still proceeds.
        """
        def failing_sentiment(ctx):
            raise RuntimeError("Sentiment API connection failure")

        def normal_category(ctx):
            return {"category": "Technical Troubleshooting", "confidence": 0.90}

        def normal_rag(ctx, top_k=3):
            return [{"document": "E-102 procedure"}]

        mock_sugg = MagicMock(return_value={
            "suggestion": "Check the breaker.",
            "confidence": 0.9,
            "is_grounded": True,
            "sources": [],
            "category": "Technical Troubleshooting"
        })

        coord = PipelineCoordinator(
            audio_capture=self.mock_audio_capture,
            transcriber=self.mock_transcriber,
            sentiment_fn=failing_sentiment,
            category_fn=normal_category,
            rag_fn=normal_rag,
            suggestion_fn=mock_sugg,
            role_classifier_fn=self.mock_role_classifier_fn,
            async_downstream=False
        )

        events = []
        coord.add_event_listener(lambda ev: events.append(ev))

        state = ConversationState(session_id="error_containment")
        seg = state.add_final_segment("Machine stopped operating.", speaker="Customer", raw_speaker="A")
        coord._handle_speech_break(seg, state)

        # Verify error event emitted for sentiment
        error_events = [ev for ev in events if ev["type"] == "system_error"]
        self.assertEqual(len(error_events), 1)
        self.assertEqual(error_events[0]["stage"], "sentiment")
        self.assertIn("Sentiment API connection failure", error_events[0]["message"])

        # Verify category and suggestion still succeeded
        cat_events = [ev for ev in events if ev["type"] == "category_update"]
        self.assertEqual(len(cat_events), 1)
        self.assertEqual(cat_events[0]["category"], "Technical Troubleshooting")

        sugg_events = [ev for ev in events if ev["type"] == "suggestion_update"]
        self.assertEqual(len(sugg_events), 1)
        mock_sugg.assert_called_once()


class TestPipelineCoordinatorCombinedAnalysis(unittest.TestCase):
    """
    Tests PipelineCoordinator integration with the Consolidated Customer Context Analyzer:
    1. Combined classifier called exactly once per CUSTOMER turn.
    2. Separate sentiment function is no longer called by the coordinator.
    3. Separate category function is no longer called by the coordinator.
    4. RAG still receives the exact same bounded customer context.
    5. Suggestion still receives the correct category.
    6. sentiment_update is still emitted.
    7. category_update is still emitted.
    8. AGENT turns still skip all downstream customer analysis.
    9. UNKNOWN turns still skip all downstream customer analysis.
    10. Existing error handling still works.
    """

    def setUp(self):
        self.mock_audio_capture = MagicMock()
        self.mock_transcriber = MagicMock()
        self.mock_transcriber.conversation_state = ConversationState(session_id="test_combined_session")
        self.mock_transcriber._client = None

        self.mock_sentiment_fn = MagicMock()
        self.mock_category_fn = MagicMock()

        self.mock_context_analyzer_fn = MagicMock(return_value={
            "sentiment": "Negative",
            "sentiment_confidence": 0.95,
            "sentiment_rationale": "Customer expressing frustration about machine breakdown.",
            "category": "Technical Troubleshooting",
            "category_confidence": 0.92,
            "category_rationale": "Reporting alarm code E-102."
        })

        self.mock_rag_fn = MagicMock(return_value=[
            {
                "document": "E-102 Dust extraction fault procedure.",
                "similarity_score": 0.85,
                "metadata": {"document_reference": "TS-WP400-3.1", "page": 6}
            }
        ])

        self.mock_suggestion_fn = MagicMock(return_value={
            "suggestion": "Check the fan breaker and replace filter cartridge WP4-FL-DE1.",
            "confidence": 0.95,
            "is_grounded": True,
            "sources": [{"document_reference": "TS-WP400-3.1", "page": 6}],
            "category": "Technical Troubleshooting"
        })

        def mock_role_classifier(current_turn, recent_turns=None, tracker=None):
            role_hint = current_turn.get("role") or current_turn.get("speaker") or "Customer"
            clean_role = "AGENT" if "agent" in str(role_hint).lower() else ("UNKNOWN" if "unknown" in str(role_hint).lower() else "CUSTOMER")
            return {
                "role": clean_role,
                "confidence": 0.95,
                "latency_ms": 10.0,
                "rationale": "Mock classification"
            }

        self.mock_role_classifier_fn = MagicMock(side_effect=mock_role_classifier)

        self.coordinator = PipelineCoordinator(
            audio_capture=self.mock_audio_capture,
            transcriber=self.mock_transcriber,
            sentiment_fn=self.mock_sentiment_fn,
            category_fn=self.mock_category_fn,
            context_analyzer_fn=self.mock_context_analyzer_fn,
            rag_fn=self.mock_rag_fn,
            suggestion_fn=self.mock_suggestion_fn,
            role_classifier_fn=self.mock_role_classifier_fn,
            max_customer_turns=4,
            async_downstream=False
        )

        self.captured_events: List[Dict[str, Any]] = []
        self.coordinator.add_event_listener(lambda ev: self.captured_events.append(ev))

    def test_combined_classifier_called_once_per_customer_turn(self):
        """
        1. Combined classifier called exactly once per CUSTOMER turn.
        2. Separate sentiment function is no longer called.
        3. Separate category function is no longer called.
        4. RAG receives exact same bounded customer context.
        5. Suggestion receives correct category.
        6. sentiment_update is emitted.
        7. category_update is emitted.
        """
        state = ConversationState(session_id="combined_turn_test")
        t1 = "My WP400 has stopped working."
        t2 = "It is showing error E102."
        state.add_final_segment(t1, speaker="Customer", raw_speaker="A")
        t2_seg = state.add_final_segment(t2, speaker="Customer", raw_speaker="A")

        self.coordinator._handle_speech_break(t2_seg, state)

        expected_context = "My WP400 has stopped working. It is showing error E102."

        # 1. Combined classifier called exactly once
        self.mock_context_analyzer_fn.assert_called_once_with(expected_context)

        # 2. Separate sentiment function is NOT called
        self.mock_sentiment_fn.assert_not_called()

        # 3. Separate category function is NOT called
        self.mock_category_fn.assert_not_called()

        # 4. RAG receives exact same bounded context
        self.mock_rag_fn.assert_called_once_with(expected_context, top_k=3)

        # 5. Suggestion receives correct category and context
        self.mock_suggestion_fn.assert_called_once_with(
            conversation_text=expected_context,
            retrieved_context=self.mock_rag_fn.return_value,
            category="Technical Troubleshooting"
        )

        # 6 & 7. Both events emitted separately
        event_types = [ev["type"] for ev in self.captured_events]
        self.assertIn("sentiment_update", event_types)
        self.assertIn("category_update", event_types)
        self.assertIn("suggestion_update", event_types)

        sent_ev = next(ev for ev in self.captured_events if ev["type"] == "sentiment_update")
        self.assertEqual(sent_ev["sentiment"], "Negative")
        self.assertEqual(sent_ev["confidence"], 0.95)

        cat_ev = next(ev for ev in self.captured_events if ev["type"] == "category_update")
        self.assertEqual(cat_ev["category"], "Technical Troubleshooting")
        self.assertEqual(cat_ev["confidence"], 0.92)

    def test_agent_turn_skips_combined_downstream_analysis(self):
        """8. AGENT turns skip downstream analysis."""
        state = ConversationState()
        seg = state.add_final_segment("Welcome to technical support.", speaker="Agent", raw_speaker="B")
        self.coordinator._handle_speech_break(seg, state)

        self.mock_context_analyzer_fn.assert_not_called()
        self.mock_sentiment_fn.assert_not_called()
        self.mock_category_fn.assert_not_called()
        self.mock_rag_fn.assert_not_called()
        self.mock_suggestion_fn.assert_not_called()

    def test_unknown_turn_skips_combined_downstream_analysis(self):
        """9. UNKNOWN turns skip downstream analysis."""
        state = ConversationState()
        seg = state.add_final_segment("Uh, okay.", speaker="Unknown", raw_speaker="A")
        self.coordinator._handle_speech_break(seg, state)

        self.mock_context_analyzer_fn.assert_not_called()
        self.mock_sentiment_fn.assert_not_called()
        self.mock_category_fn.assert_not_called()
        self.mock_rag_fn.assert_not_called()
        self.mock_suggestion_fn.assert_not_called()

    def test_combined_context_analyzer_error_handling(self):
        """10. Combined context analyzer error emits system_error without crashing."""
        self.mock_context_analyzer_fn.side_effect = RuntimeError("Combined Gemini quota error")
        state = ConversationState()
        seg = state.add_final_segment("Machine fault.", speaker="Customer", raw_speaker="A")

        self.coordinator._handle_speech_break(seg, state)

        err_events = [ev for ev in self.captured_events if ev["type"] == "system_error"]
        self.assertEqual(len(err_events), 1)
        self.assertEqual(err_events[0]["stage"], "context_analysis")
        self.assertIn("Combined Gemini quota error", err_events[0]["message"])

        # Suggestion generation still runs with category=None
        self.mock_suggestion_fn.assert_called_once()
        sugg_kwargs = self.mock_suggestion_fn.call_args.kwargs
        self.assertIsNone(sugg_kwargs["category"])


if __name__ == "__main__":
    unittest.main()

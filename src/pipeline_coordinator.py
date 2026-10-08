"""
src/pipeline_coordinator.py

Pipeline Coordinator for AI-Powered Customer Support Assistant (Phase 2 - Step 1).
Provides an orchestration layer that connects the existing Phase 1 components:
    AudioCapture (sounddevice)
        ↓
    RealtimeTranscriber (AssemblyAI v3)
        ↓
    on_partial / on_speech_break hooks
        ↓
    Sentiment Analysis (analyze_sentiment)
        ↓
    Query Categorization (classify_query)
        ↓
    Semantic Retrieval (search_knowledge_base)
        ↓
    Grounded Suggestions (generate_support_suggestion)
        ↓
    Structured Event Queue & Listeners

Designed to be independent of web/HTTP/WebSocket frameworks.
A web layer (e.g. FastAPI WebSocket) can subscribe to structured pipeline events.
"""

import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.audio_capture import AudioCapture
from src.transcription import RealtimeTranscriber, ConversationState
from src.sentiment_analysis import analyze_sentiment
from src.query_categorization import classify_query
from src.customer_context_analyzer import analyze_customer_context
from src.rag_search import search_knowledge_base
from src.llm_suggestions import generate_support_suggestion
from src.speaker_role_classifier import (
    classify_speaker_role,
    SpeakerContinuityTracker,
    get_shared_gemini_client
)

logger = logging.getLogger("pipeline_coordinator")

EventListener = Callable[[Dict[str, Any]], None]


# Default maximum number of recent customer turns passed to downstream analysis
DEFAULT_MAX_CUSTOMER_TURNS: int = 4


class PipelineCoordinator:
    """
    Coordinates real-time audio capture, transcription, conversation state,
    sentiment evaluation, categorization, and RAG suggestions into a unified event stream.
    """

    def __init__(
        self,
        audio_capture: Optional[AudioCapture] = None,
        transcriber: Optional[RealtimeTranscriber] = None,
        sentiment_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        category_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        context_analyzer_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        rag_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        suggestion_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        role_classifier_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        gemini_client: Optional[Any] = None,
        max_customer_turns: Optional[int] = None,
        async_downstream: bool = True
    ) -> None:
        """
        Initializes the coordinator with Phase 1 components or injected mock functions.

        Args:
            audio_capture: Optional AudioCapture instance (defaults to new instance).
            transcriber: Optional RealtimeTranscriber instance.
            sentiment_fn: Optional legacy sentiment function.
            category_fn: Optional legacy query categorization function.
            context_analyzer_fn: Combined sentiment and categorization function (defaults to analyze_customer_context).
            rag_fn: Function for knowledge base search (defaults to search_knowledge_base).
            suggestion_fn: Function for suggestion generation (defaults to generate_support_suggestion).
            role_classifier_fn: Function for speaker role classification (defaults to classify_speaker_role).
            gemini_client: Optional persistent Gemini client for role classification.
            max_customer_turns: Bounded number of recent customer turns for downstream analysis (default: 4).
            async_downstream: Whether to process speech-break downstream LLM/RAG calls in a background worker.
        """
        self._audio_capture = audio_capture or AudioCapture()
        self._sentiment_fn = sentiment_fn
        self._category_fn = category_fn
        self._has_custom_context_analyzer = context_analyzer_fn is not None
        self._context_analyzer_fn = context_analyzer_fn or analyze_customer_context
        self._rag_fn = rag_fn or search_knowledge_base
        self._suggestion_fn = suggestion_fn or generate_support_suggestion
        self._role_classifier_fn = role_classifier_fn or classify_speaker_role
        self._gemini_client = gemini_client if gemini_client is not None else get_shared_gemini_client()
        self._max_customer_turns = (
            max_customer_turns
            if max_customer_turns is not None
            else int(os.getenv("MAX_CUSTOMER_TURNS", str(DEFAULT_MAX_CUSTOMER_TURNS)))
        )
        self._continuity_tracker = SpeakerContinuityTracker()
        self._async_downstream = async_downstream

        # Determine once if role_classifier_fn accepts 'client' argument
        import inspect
        target_fn = getattr(self._role_classifier_fn, "side_effect", None) or self._role_classifier_fn
        if not callable(target_fn):
            target_fn = self._role_classifier_fn
        try:
            sig = inspect.signature(target_fn)
            self._role_classifier_accepts_client = (
                "client" in sig.parameters
                or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            )
        except (ValueError, TypeError):
            self._role_classifier_accepts_client = True

        # Event listeners and thread-safe event queue
        self._listeners: List[EventListener] = []
        self._listeners_lock = threading.Lock()
        self.event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

        # State and lifecycle
        self._is_running = False
        self._transcription_thread: Optional[threading.Thread] = None

        # Background worker for speech-break processing to prevent blocking the transcription stream
        self._task_queue: queue.Queue[Optional[tuple]] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None

        # Real-time transcriber initialization
        if transcriber is not None:
            self._transcriber = transcriber
            # Rebind hooks to coordinator handlers
            self._transcriber.on_partial = self._handle_partial_transcript
            self._transcriber.on_speech_break = self._handle_speech_break
        else:
            self._transcriber = RealtimeTranscriber(
                on_partial=self._handle_partial_transcript,
                on_speech_break=self._handle_speech_break
            )

        self._start_worker()

    @property
    def is_running(self) -> bool:
        """Returns True if the streaming pipeline is currently active."""
        return self._is_running

    @property
    def conversation_state(self) -> ConversationState:
        """Returns the in-memory conversation state tracker."""
        return self._transcriber.conversation_state

    # ==========================================================================
    # Event Emission & Subscription
    # ==========================================================================

    def add_event_listener(self, listener: EventListener) -> None:
        """Registers a callback listener for pipeline events."""
        with self._listeners_lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_event_listener(self, listener: EventListener) -> None:
        """Removes a previously registered callback listener."""
        with self._listeners_lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit_event(self, event: Dict[str, Any]) -> None:
        """
        Publishes a structured event to the in-memory queue and all registered listeners.
        """
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()

        self.event_queue.put(event)

        with self._listeners_lock:
            listeners_copy = list(self._listeners)

        for listener in listeners_copy:
            try:
                listener(event)
            except Exception as exc:
                logger.error("Error in event listener callback: %s", exc)

    # ==========================================================================
    # Pipeline Callbacks
    # ==========================================================================

    def _handle_partial_transcript(self, text: str, speaker: Optional[str]) -> None:
        """Invoked when AssemblyAI emits an in-progress partial transcript."""
        role_a = os.getenv("SPEAKER_A_ROLE", "Customer")
        self.emit_event({
            "type": "transcript_partial",
            "speaker": speaker or role_a,
            "text": text
        })

    def _handle_speech_break(self, segment: Dict[str, Any], state: ConversationState) -> None:
        """
        Invoked on end-of-turn / speech-break trigger.
        Classifies semantic speaker role (CUSTOMER / AGENT / UNKNOWN) using dialogue context.
        Immediately emits transcript_final and queues downstream processing if role is CUSTOMER.
        """
        turn_text = segment.get("text", "").strip()
        raw_spk = segment.get("raw_speaker") or segment.get("speaker_label") or "A"

        # Semantic Role Classification via Gemini 3.5 Flash Lite
        recent_turns = [s for s in state.segments if s != segment][-4:]
        classifier_kwargs: Dict[str, Any] = {
            "current_turn": segment,
            "recent_turns": recent_turns,
            "tracker": self._continuity_tracker
        }
        if self._gemini_client is None:
            self._gemini_client = get_shared_gemini_client()

        if self._gemini_client is not None and self._role_classifier_accepts_client:
            classifier_kwargs["client"] = self._gemini_client
            try:
                role_res = self._role_classifier_fn(**classifier_kwargs)
            except TypeError as te:
                if "client" in str(te):
                    self._role_classifier_accepts_client = False
                    classifier_kwargs.pop("client", None)
                    role_res = self._role_classifier_fn(**classifier_kwargs)
                else:
                    raise
        else:
            role_res = self._role_classifier_fn(**classifier_kwargs)

        assigned_role = str(role_res.get("role", "UNKNOWN")).upper()
        display_speaker = "Customer" if assigned_role == "CUSTOMER" else ("Agent" if assigned_role == "AGENT" else "Unknown")
        role_conf = float(role_res.get("confidence", 0.0))
        latency_ms = float(role_res.get("latency_ms", 0.0))

        # Update segment dictionary in-place so state.segments reflects the validated role
        segment["speaker"] = display_speaker
        segment["role"] = assigned_role
        segment["role_confidence"] = role_conf
        segment["role_latency_ms"] = latency_ms

        turn_idx = len(state.segments)
        ts = segment.get("timestamp")

        self.emit_event({
            "type": "transcript_final",
            "speaker": display_speaker,
            "raw_speaker": raw_spk,
            "text": turn_text,
            "confidence": segment.get("confidence"),
            "role": assigned_role,
            "role_confidence": role_conf,
            "role_latency_ms": latency_ms
        })

        # Diagnostic output for diarization tracing
        ts_line = f"\ntimestamp={ts}" if ts else ""
        print(
            f"\n[DIARIZATION]\n"
            f"turn={turn_idx}\n"
            f"text=\"{turn_text}\"\n"
            f"raw_speaker={raw_spk}\n"
            f"mapped_role={assigned_role}\n"
            f"confidence={role_conf:.2f}\n"
            f"latency_ms={latency_ms:.1f}"
            f"{ts_line}\n",
            flush=True
        )

        if self._async_downstream:
            self._task_queue.put((segment, state))
        else:
            self._process_downstream_turn(segment, state)

    # ==========================================================================
    # Downstream Turn Processing (Sentiment, Category, RAG, Suggestions)
    # ==========================================================================

    def _start_worker(self) -> None:
        """Starts the dedicated background worker for downstream processing."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="PipelineCoordinatorWorker",
                daemon=True
            )
            self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker thread loop consuming speech-break tasks sequentially."""
        while True:
            item = self._task_queue.get()
            if item is None:
                self._task_queue.task_done()
                break
            try:
                segment, state = item
                self._process_downstream_turn(segment, state)
            except Exception as exc:
                logger.error("Unhandled error in downstream worker loop: %s", exc)
            finally:
                self._task_queue.task_done()

    def _process_downstream_turn(self, segment: Dict[str, Any], state: ConversationState) -> None:
        """
        Executes analytical stages on confirmed customer speech only.
        Agent and Unknown responses are displayed in the transcript feed but excluded from
        customer sentiment, query categorization, and knowledge retrieval.
        """
        turn_text = segment.get("text", "").strip()
        if not turn_text:
            return

        assigned_role = str(segment.get("role") or segment.get("speaker") or "UNKNOWN").upper()

        # Decision Gate: Only confirmed CUSTOMER speech triggers downstream analysis
        if assigned_role != "CUSTOMER":
            logger.info("Turn role is %s (not CUSTOMER); skipping customer downstream analysis.", assigned_role)
            return

        # 1. Extract bounded customer-only conversation history (latest N customer turns)
        customer_text = state.get_recent_customer_transcript(
            max_turns=self._max_customer_turns,
            customer_role="CUSTOMER"
        )

        # Fallback for unit tests where state was not pre-populated
        if not customer_text:
            customer_text = turn_text

        context_text = customer_text

        # Check whether to use Combined Context Analysis (production & when context_analyzer_fn is injected)
        # or legacy separate sentiment/category mocks passed by older unit tests.
        if self._has_custom_context_analyzer or (self._sentiment_fn is None and self._category_fn is None):
            # 1. Combined Context Analysis (Sentiment + Category in ONE Gemini call)
            # 2. ChromaDB Semantic Retrieval (Gemini Embedding 2 + Vector Search)
            # Uses standard library ThreadPoolExecutor with exactly 2 workers.
            # Both tasks receive the exact same bounded customer context (context_text).
            analysis_res = None
            detected_category = None
            retrieved_chunks = []

            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_analysis = executor.submit(self._context_analyzer_fn, context_text)
                fut_rag = executor.submit(self._rag_fn, context_text, top_k=3)

                # 1. Combined Sentiment & Category handling
                try:
                    analysis_res = fut_analysis.result()
                    if analysis_res:
                        sentiment = analysis_res.get("sentiment")
                        sent_conf = analysis_res.get("sentiment_confidence")
                        if sent_conf is None:
                            sent_conf = analysis_res.get("confidence")
                        sent_rat = analysis_res.get("sentiment_rationale")
                        if sent_rat is None:
                            sent_rat = analysis_res.get("rationale")

                        self.emit_event({
                            "type": "sentiment_update",
                            "sentiment": sentiment,
                            "confidence": sent_conf,
                            "rationale": sent_rat
                        })

                        detected_category = analysis_res.get("category")
                        cat_conf = analysis_res.get("category_confidence")
                        if cat_conf is None:
                            cat_conf = analysis_res.get("confidence")
                        cat_rat = analysis_res.get("category_rationale")
                        if cat_rat is None:
                            cat_rat = analysis_res.get("rationale")

                        self.emit_event({
                            "type": "category_update",
                            "category": detected_category,
                            "confidence": cat_conf,
                            "rationale": cat_rat
                        })
                except Exception as exc:
                    logger.error("Combined customer context analysis stage failed: %s", exc)
                    self.emit_event({
                        "type": "system_error",
                        "stage": "context_analysis",
                        "message": str(exc)
                    })

                # 2. ChromaDB Semantic Retrieval handling
                try:
                    retrieved_chunks = fut_rag.result() or []
                except Exception as exc:
                    logger.error("RAG search stage failed: %s", exc)
                    self.emit_event({
                        "type": "system_error",
                        "stage": "rag",
                        "message": str(exc)
                    })
        else:
            # Legacy fallback: Separate Sentiment and Category functions for older test cases
            sentiment_res = None
            detected_category = None
            retrieved_chunks = []

            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_sentiment = executor.submit(self._sentiment_fn, context_text)
                fut_category = executor.submit(self._category_fn, context_text)
                fut_rag = executor.submit(self._rag_fn, context_text, top_k=3)

                try:
                    sentiment_res = fut_sentiment.result()
                    if sentiment_res:
                        self.emit_event({
                            "type": "sentiment_update",
                            "sentiment": sentiment_res.get("sentiment"),
                            "confidence": sentiment_res.get("confidence"),
                            "rationale": sentiment_res.get("rationale")
                        })
                except Exception as exc:
                    logger.error("Sentiment analysis stage failed: %s", exc)
                    self.emit_event({
                        "type": "system_error",
                        "stage": "sentiment",
                        "message": str(exc)
                    })

                try:
                    cat_res = fut_category.result()
                    if cat_res:
                        detected_category = cat_res.get("category")
                        self.emit_event({
                            "type": "category_update",
                            "category": detected_category,
                            "confidence": cat_res.get("confidence"),
                            "rationale": cat_res.get("rationale")
                        })
                except Exception as exc:
                    logger.error("Query categorization stage failed: %s", exc)
                    self.emit_event({
                        "type": "system_error",
                        "stage": "categorization",
                        "message": str(exc)
                    })

                try:
                    retrieved_chunks = fut_rag.result() or []
                except Exception as exc:
                    logger.error("RAG search stage failed: %s", exc)
                    self.emit_event({
                        "type": "system_error",
                        "stage": "rag",
                        "message": str(exc)
                    })

        # 4. Support Suggestion Generation (Uses accumulated context_text)
        try:
            sugg_res = self._suggestion_fn(
                conversation_text=context_text,
                retrieved_context=retrieved_chunks,
                category=detected_category
            )
            self.emit_event({
                "type": "suggestion_update",
                "suggestion": sugg_res.get("suggestion"),
                "confidence": sugg_res.get("confidence"),
                "is_grounded": sugg_res.get("is_grounded"),
                "sources": sugg_res.get("sources", []),
                "category": sugg_res.get("category")
            })
        except Exception as exc:
            logger.error("Support suggestion generation stage failed: %s", exc)
            self.emit_event({
                "type": "system_error",
                "stage": "suggestions",
                "message": str(exc)
            })

    def wait_for_downstream(self, timeout: Optional[float] = 5.0) -> None:
        """Blocks until all queued downstream tasks are completed (useful for testing)."""
        self._task_queue.join()

    # ==========================================================================
    # Lifecycle Management
    # ==========================================================================

    def reset_session(self) -> None:
        """
        Resets ephemeral session state:
        - Clears conversation state segments and partial text.
        - Drains pending downstream tasks from previous session.
        - Drains unconsumed events from internal event queue.
        """
        if hasattr(self, "_transcriber") and self._transcriber is not None:
            if hasattr(self._transcriber, "conversation_state") and self._transcriber.conversation_state is not None:
                self._transcriber.conversation_state.clear()

        if hasattr(self, "_continuity_tracker") and self._continuity_tracker is not None:
            self._continuity_tracker.reset()

        # Drain downstream task queue
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except (queue.Empty, ValueError):
                break

        # Drain internal event queue
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def start(self) -> None:
        """
        Starts the live microphone capture and AssemblyAI streaming loop in a background thread.
        Resets conversation state to ensure a completely fresh session.
        """
        if self._is_running:
            logger.warning("Pipeline is already running.")
            return

        logger.info("Starting Pipeline Coordinator...")
        self.reset_session()
        self._audio_capture.start()
        self._is_running = True

        self._transcription_thread = threading.Thread(
            target=self._run_transcription_loop,
            name="PipelineCoordinatorTranscription",
            daemon=True
        )
        self._transcription_thread.start()

        self.emit_event({
            "type": "system_status",
            "status": "started",
            "device": getattr(self._audio_capture, "device_name", "Default Microphone")
        })

    def _run_transcription_loop(self) -> None:
        """Background thread executing the blocking transcriber streaming method."""
        try:
            self._transcriber.start_streaming(self._audio_capture)
        except Exception as exc:
            logger.error("Transcription streaming loop failed: %s", exc)
            self.emit_event({
                "type": "system_error",
                "stage": "transcription",
                "message": str(exc)
            })
        finally:
            self._is_running = False

    def stop(self) -> None:
        """
        Gracefully terminates audio capture, transcriber, and downstream worker tasks.
        Drains pending tasks to prevent cross-session contamination.
        """
        if not self._is_running and self._transcriber._client is None:
            return

        logger.info("Stopping Pipeline Coordinator...")
        try:
            self._transcriber.stop()
        except Exception as exc:
            logger.warning("Error stopping transcriber: %s", exc)

        try:
            self._audio_capture.stop()
        except Exception as exc:
            logger.warning("Error stopping audio capture: %s", exc)

        self._is_running = False

        if hasattr(self, "_continuity_tracker") and self._continuity_tracker is not None:
            self._continuity_tracker.reset()

        # Drain pending downstream tasks so they cannot leak into subsequent sessions
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except (queue.Empty, ValueError):
                break

        self.emit_event({
            "type": "system_status",
            "status": "stopped"
        })

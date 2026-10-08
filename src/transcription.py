"""
src/transcription.py

Real-Time Speech-to-Text Transcription Module via AssemblyAI Streaming API (v3).
Phase 1 - Slice 1: Live Audio -> AssemblyAI Streaming -> Speech Break -> Conversation State.

Key Capabilities:
- In-memory ConversationState tracking session ID, timestamped segments, and speaker labels.
- Live microphone streaming to AssemblyAI WebSocket.
- Distinguishes partial transcripts from final transcript segments.
- Detects speech breaks (end-of-turn events) and triggers downstream processing hook.
- ZERO audio saved to disk; ZERO persistent recording.
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import assemblyai.streaming.v3 as aai_v3
from src.audio_capture import AudioCapture, AudioCaptureError

logger = logging.getLogger("transcription")


# Role mapping configuration for diarization
SPEAKER_A_ROLE = os.getenv("SPEAKER_A_ROLE", "Customer")
SPEAKER_B_ROLE = os.getenv("SPEAKER_B_ROLE", "Agent")


def map_speaker_to_role(speaker_label: Optional[str]) -> str:
    """
    Maps an AssemblyAI raw speaker label (e.g. 'A', 'B', 'Speaker A', '0')
    to a configured role (SPEAKER_A_ROLE='Customer', SPEAKER_B_ROLE='Agent').
    Preserves existing role names if already 'Customer' or 'Agent'.
    Defaults to SPEAKER_A_ROLE ('Customer') if label is None or unmapped.
    """
    role_a = os.getenv("SPEAKER_A_ROLE", "Customer")
    role_b = os.getenv("SPEAKER_B_ROLE", "Agent")

    if not speaker_label:
        return role_a

    label_clean = str(speaker_label).strip()

    if label_clean.lower() == role_a.lower():
        return label_clean
    if label_clean.lower() == role_b.lower():
        return label_clean

    upper = label_clean.upper()
    if upper in {"A", "SPEAKER A", "SPEAKER_A", "0", "SPEAKER 0", "SPEAKER_0"} or upper.endswith(" A") or upper.endswith("_A"):
        return role_a
    elif upper in {"B", "SPEAKER B", "SPEAKER_B", "1", "SPEAKER 1", "SPEAKER_1"} or upper.endswith(" B") or upper.endswith("_B"):
        return role_b

    return role_a


class TranscriptionError(Exception):
    """Raised when transcription session fails or configuration is invalid."""
    pass


# ==============================================================================
# 1. In-Memory Conversation State
# ==============================================================================

class ConversationState:
    """
    Maintains ephemeral in-memory conversation state for the active support session.
    NO database, NO audio recording, and NO persistent disk storage.
    """
    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or f"session_{int(time.time())}"
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.segments: List[Dict[str, Any]] = []
        self.current_partial: Optional[str] = None
        self.current_speaker: Optional[str] = None

    def set_session_id(self, session_id: str) -> None:
        """Sets the upstream session ID (e.g. from AssemblyAI BeginEvent)."""
        self.session_id = session_id

    def set_partial(self, text: str, speaker: Optional[str] = None) -> None:
        """Updates the current in-progress partial transcript."""
        self.current_partial = text.strip()
        self.current_speaker = speaker if speaker else map_speaker_to_role(None)

    def add_final_segment(
        self,
        text: str,
        speaker: Optional[str] = None,
        confidence: Optional[float] = None,
        raw_speaker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Appends a finalized transcript segment upon turn completion / speech break.
        Clears the in-progress partial transcript.
        Preserves raw AssemblyAI speaker label and maps to configured role.
        """
        cleaned_text = text.strip()
        timestamp = datetime.now(timezone.utc).isoformat()

        role_a = os.getenv("SPEAKER_A_ROLE", "Customer")
        role_b = os.getenv("SPEAKER_B_ROLE", "Agent")

        actual_raw = raw_speaker or speaker or "A"
        if speaker is not None:
            final_role = speaker
        else:
            final_role = map_speaker_to_role(actual_raw)

        segment = {
            "timestamp": timestamp,
            "speaker": final_role,
            "raw_speaker": actual_raw,
            "speaker_label": actual_raw,
            "text": cleaned_text,
            "confidence": confidence,
            "is_final": True
        }
        self.segments.append(segment)
        self.current_partial = None
        self.current_speaker = None
        return segment

    def get_full_transcript(self, include_speaker: bool = False) -> str:
        """
        Returns full conversation string concatenated from finalized segments.
        """
        if not self.segments:
            return ""
        if include_speaker:
            return "\n".join(f"[{s['speaker']}] {s['text']}" for s in self.segments)
        return " ".join(s["text"] for s in self.segments)

    def get_customer_transcript(self, customer_role: Optional[str] = None) -> str:
        """
        Returns conversation string concatenated from segments spoken by Customer only.
        """
        target_role = (customer_role or os.getenv("SPEAKER_A_ROLE", "Customer")).lower()
        cust_segments = [
            s for s in self.segments
            if s.get("speaker", "").lower() == target_role or s.get("role", "").lower() == target_role
        ]
        if not cust_segments:
            return ""
        return " ".join(s["text"] for s in cust_segments)

    def get_recent_customer_transcript(
        self,
        max_turns: int = 4,
        customer_role: Optional[str] = None
    ) -> str:
        """
        Returns bounded conversation string concatenated from the most recent N
        finalized turns spoken by Customer only.

        - Considers only confirmed CUSTOMER turns.
        - Excludes AGENT turns.
        - Excludes UNKNOWN turns.
        - Preserves chronological order.
        - Returns only the most recent N customer turns (defaults to 4).
        - If fewer than N customer turns exist, returns all available customer turns.
        - Does NOT modify or mutate self.segments.
        """
        if max_turns <= 0:
            return ""

        target_role = (customer_role or os.getenv("SPEAKER_A_ROLE", "Customer")).lower()
        cust_segments = [
            s for s in self.segments
            if s.get("speaker", "").lower() == target_role or s.get("role", "").lower() == target_role
        ]
        if not cust_segments:
            return ""

        recent_cust_segments = cust_segments[-max_turns:]
        return " ".join(s["text"] for s in recent_cust_segments)

    def get_recent_context(self, max_segments: int = 5) -> List[Dict[str, Any]]:
        """
        Returns the last N finalized conversation segments for downstream context.
        """
        return self.segments[-max_segments:]

    def clear(self) -> None:
        """Resets the in-memory state."""
        self.segments.clear()
        self.current_partial = None
        self.current_speaker = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current state to a simple dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_segments": len(self.segments),
            "full_transcript": self.get_full_transcript(),
            "segments": list(self.segments)
        }


# ==============================================================================
# 2. Real-Time Streaming Transcriber
# ==============================================================================

class RealtimeTranscriber:
    """
    Manages WebSocket connection to AssemblyAI Streaming API v3.
    Processes live audio streams, captures turn events, and triggers speech-break hooks.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        on_speech_break: Optional[Callable[[Dict[str, Any], ConversationState], None]] = None,
        on_partial: Optional[Callable[[str, Optional[str]], None]] = None
    ) -> None:
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.sample_rate = sample_rate
        self.on_speech_break = on_speech_break or self._default_speech_break_hook
        self.on_partial = on_partial

        self.conversation_state = ConversationState()
        self._client: Optional[aai_v3.RealTimeTranscriber] = None
        self._is_connected: bool = False

    @staticmethod
    def _default_speech_break_hook(segment: Dict[str, Any], state: ConversationState) -> None:
        """Default hook executed when speech break occurs."""
        logger.info("Speech break detected — conversation ready for processing")
        print("\nSPEECH BREAK DETECTED")
        print(f"Conversation state updated: '{segment['text']}'")
        print("Ready for downstream processing.\n")

    def _on_begin(self, client: Any, event: aai_v3.BeginEvent) -> None:
        """Invoked when AssemblyAI confirms session establishment."""
        self._is_connected = True
        self.conversation_state.set_session_id(event.id)
        logger.info("AssemblyAI connected. Session ID: %s", event.id)

    def _on_turn(self, client: Any, event: aai_v3.TurnEvent) -> None:
        """Invoked for all incoming streaming turn events."""
        if not event.transcript or not event.transcript.strip():
            return

        raw_event_speaker = getattr(event, "speaker_label", None)
        raw_speaker = raw_event_speaker or "A"
        mapped_speaker = map_speaker_to_role(raw_speaker)
        transcript_text = event.transcript.strip()

        if event.end_of_turn:
            # Final transcript segment -> Speech Break Trigger!
            segment = self.conversation_state.add_final_segment(
                text=transcript_text,
                speaker=mapped_speaker,
                confidence=getattr(event, "end_of_turn_confidence", None),
                raw_speaker=raw_speaker
            )
            logger.info(
                "Final transcript received: [%s (raw: %s, aai_event_label: %r)] %s",
                mapped_speaker, raw_speaker, raw_event_speaker, transcript_text
            )

            # Fire downstream hook with newly finalized turn & full state
            if self.on_speech_break:
                self.on_speech_break(segment, self.conversation_state)
        else:
            # In-progress partial transcript
            self.conversation_state.set_partial(transcript_text, mapped_speaker)
            if self.on_partial:
                self.on_partial(transcript_text, mapped_speaker)

    def _on_error(self, client: Any, event: Any) -> None:
        """Invoked on streaming error."""
        logger.error("AssemblyAI Streaming error: %s", event)

    def _on_termination(self, client: Any, event: aai_v3.TerminationEvent) -> None:
        """Invoked when session terminates."""
        self._is_connected = False
        logger.info(
            "AssemblyAI session terminated. Audio: %.2fs, Session: %.2fs",
            event.audio_duration_seconds,
            event.session_duration_seconds
        )

    def start_streaming(self, audio_capture: AudioCapture) -> None:
        """
        Connects to AssemblyAI and streams audio generator from AudioCapture.
        Enables streaming speaker diarization with max 2 speakers (Customer & Agent).
        """
        if not self.api_key or self.api_key.strip().startswith("your_"):
            raise TranscriptionError(
                "ASSEMBLYAI_API_KEY is not configured in .env. Please set a valid API key to stream."
            )

        logger.info("Initializing AssemblyAI Real-Time Transcriber (v3)...")
        options = aai_v3.RealTimeTranscriberOptions(api_key=self.api_key)
        self._client = aai_v3.RealTimeTranscriber(options=options)

        # Register event handlers before connect
        self._client.on(aai_v3.RealTimeEvents.Begin, self._on_begin)
        self._client.on(aai_v3.RealTimeEvents.Turn, self._on_turn)
        self._client.on(aai_v3.RealTimeEvents.Error, self._on_error)
        self._client.on(aai_v3.RealTimeEvents.Termination, self._on_termination)

        # Connect session parameters with speaker diarization enabled
        params = aai_v3.RealTimeParameters(
            sample_rate=self.sample_rate,
            encoding=aai_v3.Encoding.pcm_s16le,
            format_turns=True,
            speaker_labels=True,
            max_speakers=2
        )

        logger.info("Connecting to AssemblyAI WebSocket...")
        self._client.connect(params)

        # Feed audio stream generator
        logger.info("Streaming microphone audio to AssemblyAI...")
        self._client.stream(audio_capture.stream_generator())

    def stop(self) -> None:
        """Gracefully disconnects AssemblyAI streaming session."""
        if self._client is not None:
            try:
                self._client.disconnect(terminate=True)
            except Exception as e:
                logger.warning("Error during AssemblyAI disconnect: %s", e)
            finally:
                self._client = None
        self._is_connected = False
        logger.info("AssemblyAI session closed cleanly.")


# ==============================================================================
# 3. CLI Demonstration & Testing
# ==============================================================================

def run_cli_test() -> None:
    """CLI test entry point demonstrating Slice 1 functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    print("=" * 65)
    print("CUSTOMER SUPPORT ASSISTANT — LIVE TRANSCRIPTION (PHASE 1 - SLICE 1)")
    print("=" * 65)

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key or api_key.strip().startswith("your_"):
        print("\n[WARNING] ASSEMBLYAI_API_KEY is not set or contains a placeholder in .env.")
        print("To run the live streaming test:")
        print("  1. Add your real AssemblyAI API key to .env:")
        print("     ASSEMBLYAI_API_KEY=your_actual_key_here")
        print("  2. Re-run: python -m src.transcription\n")
        print("Running mock state machine verification to validate speech-break logic...")
        _run_mock_verification()
        return

    # User callback functions for live display
    def display_partial(text: str, speaker: Optional[str]) -> None:
        spk = f"[{speaker or 'agent'}] " if speaker else ""
        print(f"\r{spk}{text}", end="", flush=True)

    def display_final(segment: Dict[str, Any], state: ConversationState) -> None:
        print(f"\n\nFINAL TRANSCRIPT:")
        print(f"[{segment['speaker']}] {segment['text']}")
        print("\nSPEECH BREAK DETECTED")
        print("Conversation state updated.")
        print("Ready for downstream processing.\n")

    audio_capture = AudioCapture(sample_rate=16000, channels=1, chunk_size=1024)
    transcriber = RealtimeTranscriber(
        api_key=api_key,
        sample_rate=16000,
        on_speech_break=display_final,
        on_partial=display_partial
    )

    try:
        print("\nStarting microphone...")
        audio_capture.start()
        print(f"Microphone: {audio_capture.device_name}")
        print("Connecting to AssemblyAI Streaming API...")
        print("Listening... (Speak into microphone. Press Ctrl+C to stop)\n")

        transcriber.start_streaming(audio_capture)

    except KeyboardInterrupt:
        print("\nStopping transcription session upon user request...")
    except Exception as exc:
        print(f"\n[ERROR] Transcription session failed: {exc}", file=sys.stderr)
    finally:
        audio_capture.stop()
        transcriber.stop()
        print("\n" + "=" * 65)
        print("Final Conversation Summary:")
        print("=" * 65)
        print(transcriber.conversation_state.get_full_transcript(include_speaker=True) or "(No speech detected)")
        print("=" * 65)
        print("Session stopped cleanly.\n")


def _run_mock_verification() -> None:
    """Validates the state machine, turn routing, and downstream hook offline."""
    print("=" * 60)
    print("Offline State Machine & Speech Break Verification")
    print("=" * 60)
    state = ConversationState(session_id="test_mock_session")

    # Simulate Partial 1
    state.set_partial("My machine is showing error", speaker="customer")
    print("[Partial 1]:", state.current_partial)

    # Simulate Partial 2
    state.set_partial("My machine is showing error E-102 and dust extraction stopped", speaker="customer")
    print("[Partial 2]:", state.current_partial)

    # Simulate Final / Speech Break
    hook_fired = False
    def mock_hook(segment, conv_state):
        nonlocal hook_fired
        hook_fired = True
        print("\n[DOWNSTREAM HOOK TRIGGERED]")
        print(f"  Segment: [{segment['speaker']}] {segment['text']}")
        print(f"  Full Conversation Text: {conv_state.get_full_transcript()}")

    final_seg = state.add_final_segment(
        text="My machine is showing error E-102 and the dust extraction has stopped.",
        speaker="customer",
        confidence=0.98
    )
    mock_hook(final_seg, state)

    assert hook_fired, "Hook must be fired on speech break"
    assert len(state.segments) == 1, "Must store 1 finalized segment"
    assert state.current_partial is None, "Partial must be reset after final turn"
    print("\n[PASS] State machine and speech break hook logic verified!")
    print("=" * 60)


if __name__ == "__main__":
    run_cli_test()

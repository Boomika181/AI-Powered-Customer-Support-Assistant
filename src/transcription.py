"""
src/transcription.py

Real-Time Speech-to-Text & Diarization Module via AssemblyAI Streaming API.
Scheduled for Phase 1 implementation.

Phase 1 Responsibilities:
- Establish WebSocket connection with AssemblyAI Real-Time Transcriber.
- Pipe PCM audio frames from audio_capture.py.
- Receive live transcripts with speaker diarization labels.
- Detect speech breaks / turn transitions (`utterance_end` or `message_type == 'FinalTranscript'`).
- Trigger downstream conversation analysis.
"""

from typing import Callable, Optional


class RealtimeTranscriber:
    """
    Placeholder for AssemblyAI real-time streaming integration in Phase 1.
    """
    def __init__(self, api_key: Optional[str] = None, on_speech_break: Optional[Callable] = None):
        self.api_key = api_key
        self.on_speech_break = on_speech_break

    def connect(self) -> None:
        """
        TODO (Phase 1): Connect to AssemblyAI Streaming WebSocket API.
        """
        raise NotImplementedError("Phase 1 Feature: AssemblyAI real-time connection.")

    def process_transcript_event(self, event_data: dict) -> None:
        """
        TODO (Phase 1): Parse speaker labels and trigger downstream pipeline on speech break.
        """
        raise NotImplementedError("Phase 1 Feature: Transcript event handling.")

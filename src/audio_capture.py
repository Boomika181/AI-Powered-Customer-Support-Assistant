"""
src/audio_capture.py

Microphone Audio Capture Module.
Scheduled for Phase 1 implementation.

Phase 0 Status:
- Standalone verification script in `scripts/test_microphone.py` verified hardware detection,
  stream opening, frame capture, and zero disk storage.

Phase 1 Responsibilities:
- Continuous background audio capture from microphone using sounddevice.
- Streaming raw PCM frames into an in-memory queue/generator.
- Interfacing with AssemblyAI Streaming API WebSocket.
- Clean shutdown on call termination without storing audio to disk.
"""

from typing import Generator, Optional
import numpy as np


class AudioStreamer:
    """
    Placeholder for continuous live microphone streaming in Phase 1.
    """
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._is_streaming = False

    def start_stream(self) -> None:
        """
        TODO (Phase 1): Initialize sounddevice InputStream and start producer thread.
        """
        raise NotImplementedError("Phase 1 Feature: Continuous audio streaming pipeline.")

    def stream_generator(self) -> Generator[bytes, None, None]:
        """
        TODO (Phase 1): Yield raw PCM audio bytes to AssemblyAI WebSocket.
        """
        raise NotImplementedError("Phase 1 Feature: Audio frame generator.")

    def stop_stream(self) -> None:
        """
        TODO (Phase 1): Safely stop and close the sounddevice InputStream.
        """
        self._is_streaming = False

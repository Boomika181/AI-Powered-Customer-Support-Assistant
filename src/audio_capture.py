"""
src/audio_capture.py

Microphone Audio Capture Module for Phase 1.
Captures live audio from system microphone using sounddevice,
converts frames to raw 16-bit linear PCM little-endian bytes (pcm_s16le),
and buffers them in a thread-safe in-memory queue for real-time streaming.

Key Design Principles:
- Pure in-memory streaming: ZERO audio is ever written to disk or recorded permanently.
- Clean lifecycle: start(), stop(), and generator stream for consumers.
- Explicit audio parameters: 16000 Hz, mono, 16-bit signed PCM (int16).
- Graceful error handling for missing/disconnected microphones.
"""

import logging
import queue
from typing import Generator, Optional
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCaptureError(Exception):
    """Raised when audio capture initialization or streaming fails."""
    pass


class AudioCapture:
    """
    Captures live audio from system microphone and streams raw PCM bytes.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index

        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._stream: Optional[sd.InputStream] = None
        self._is_running: bool = False
        self._device_name: str = "Default Input Device"

    @property
    def is_running(self) -> bool:
        """Returns True if the microphone stream is currently active."""
        return self._is_running

    @property
    def device_name(self) -> str:
        """Returns the human-readable name of the active audio device."""
        return self._device_name

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Internal callback invoked by sounddevice for each audio block."""
        if status:
            logger.warning("Sounddevice status flag: %s", status)
        if self._is_running:
            # indata is numpy.ndarray with dtype=int16, mono
            raw_pcm_bytes = indata.tobytes()
            try:
                self._audio_queue.put_nowait(raw_pcm_bytes)
            except queue.Full:
                # Drop oldest frame if queue overflows to avoid blocking audio thread
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    pass
                self._audio_queue.put_nowait(raw_pcm_bytes)

    def start(self) -> None:
        """
        Initializes and starts the audio input stream.
        """
        if self._is_running:
            logger.warning("AudioCapture is already running.")
            return

        try:
            # Query device info
            if self.device_index is not None:
                dev_info = sd.query_devices(self.device_index)
            else:
                dev_info = sd.query_devices(kind="input")

            self._device_name = dev_info.get("name", "Unknown Microphone")

            # Validate input channel capability
            if dev_info.get("max_input_channels", 0) < self.channels:
                raise AudioCaptureError(
                    f"Selected device '{self._device_name}' does not support {self.channels} input channel(s)."
                )

            # Check settings compatibility
            dev_idx = self.device_index if self.device_index is not None else dev_info.get("index")
            sd.check_input_settings(
                device=dev_idx,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16"
            )

            # Open input stream
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.chunk_size,
                device=dev_idx,
                callback=self._audio_callback
            )

            self._is_running = True
            self._stream.start()
            logger.info(
                "Microphone started: '%s' (Rate: %d Hz, Channels: %d, Chunk size: %d frames)",
                self._device_name,
                self.sample_rate,
                self.channels,
                self.chunk_size
            )

        except Exception as exc:
            self._is_running = False
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            logger.error("Failed to start audio capture: %s", exc)
            raise AudioCaptureError(f"Failed to start audio capture: {exc}") from exc

    def read_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Reads a single PCM chunk from the queue. Returns None on timeout.
        """
        if not self._is_running and self._audio_queue.empty():
            return None
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream_generator(self) -> Generator[bytes, None, None]:
        """
        Yields raw PCM audio chunks continuously while capture is active.
        Suitable for piping directly to AssemblyAI streaming transcriber.
        """
        while self._is_running:
            chunk = self.read_chunk(timeout=0.2)
            if chunk is not None:
                yield chunk

    def stop(self) -> None:
        """
        Safely stops the audio stream and releases hardware resources.
        """
        if not self._is_running and self._stream is None:
            return

        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error closing audio stream: %s", e)
            finally:
                self._stream = None

        # Drain remaining buffer in memory
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Microphone stopped: '%s' released cleanly.", self._device_name)

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

#!/usr/bin/env python3
"""
scripts/test_microphone.py

Phase 0 Microphone Capture Test:
Verifies system microphone hardware detection, stream initialization,
and frame reception without transcription or disk storage.

Key Objectives:
1. Detect available input audio devices.
2. Select default or designated microphone.
3. Open an audio input stream using sounddevice.
4. Capture raw audio frames in memory for verification.
5. Compute basic signal metrics (frame count, peak amplitude) to verify live input.
6. Stop and close the stream cleanly.
7. Discard all captured frames immediately (Zero Audio Storage).
"""

import sys
import argparse
import numpy as np
import sounddevice as sd


def test_microphone(device_id: int | None = None, duration_sec: float = 2.0, sample_rate: int = 16000):
    print("=" * 65)
    print("Phase 0 - Basic Microphone Capture Test")
    print("=" * 65)

    # 1. Query available devices
    all_devices = sd.query_devices()
    input_devices = [
        (idx, dev)
        for idx, dev in enumerate(all_devices)
        if dev["max_input_channels"] > 0
    ]

    print("\n[1] Detected Audio Input Devices:")
    if not input_devices:
        print("  [ERROR] No input audio devices detected on this system!")
        print("=" * 65)
        return False

    for idx, dev in input_devices:
        is_default = (idx == sd.default.device[0])
        default_marker = " (System Default)" if is_default else ""
        print(f"  [{idx}] {dev['name']}{default_marker}")
        print(f"      Channels: {dev['max_input_channels']}, Default Rate: {int(dev['default_samplerate'])} Hz")

    # 2. Select Microphone
    selected_device_idx = device_id if device_id is not None else sd.default.device[0]
    if selected_device_idx is None or selected_device_idx < 0:
        selected_device_idx = input_devices[0][0]

    selected_dev = sd.query_devices(selected_device_idx)
    print(f"\n[2] Selected Microphone:")
    print(f"  Name:        {selected_dev['name']}")
    print(f"  Device ID:   {selected_device_idx}")
    print(f"  Sample Rate: {sample_rate} Hz (Target) | {int(selected_dev['default_samplerate'])} Hz (Native)")
    print(f"  Channels:    1 (Mono)")

    # 3. Check sample rate compatibility
    try:
        sd.check_input_settings(device=selected_device_idx, samplerate=sample_rate, channels=1)
        actual_sample_rate = sample_rate
    except Exception:
        # Fallback to device's native rate if 16000 is not supported
        actual_sample_rate = int(selected_dev["default_samplerate"])
        print(f"  [NOTE] Target sample rate {sample_rate} Hz adjusted to device native: {actual_sample_rate} Hz")

    # 4. Open Stream & Capture Frames
    print(f"\n[3] Opening Audio Stream (capturing {duration_sec}s for verification)...")
    stream_opened_successfully = False
    frames_received = 0
    peak_amplitude = 0.0
    buffer_chunks = []

    block_size = 1024
    total_chunks = int((actual_sample_rate * duration_sec) / block_size)

    try:
        stream = sd.InputStream(
            device=selected_device_idx,
            samplerate=actual_sample_rate,
            channels=1,
            blocksize=block_size,
            dtype="float32"
        )
        stream.start()
        stream_opened_successfully = True
        print("  -> Audio stream opened successfully: YES")

        print("  -> Capturing audio frames...")
        for _ in range(total_chunks):
            data, overflowed = stream.read(block_size)
            frames_received += len(data)
            max_amp = float(np.max(np.abs(data)))
            if max_amp > peak_amplitude:
                peak_amplitude = max_amp
            buffer_chunks.append(len(data))

        stream.stop()
        stream.close()
        print("  -> Audio stream stopped and closed successfully: YES")

    except Exception as e:
        print(f"  [ERROR] Failed to stream from microphone: {e}")
        return False

    finally:
        # Guarantee no persistent storage
        buffer_chunks.clear()

    # 5. Report Results
    audio_frames_received = frames_received > 0
    print("\n" + "=" * 65)
    print("Microphone Test Results Summary")
    print("=" * 65)
    print(f"• Input Devices Found:       {len(input_devices)}")
    print(f"• Selected Microphone:       {selected_dev['name']}")
    print(f"• Sample Rate Used:          {actual_sample_rate} Hz")
    print(f"• Stream Opened:             {'YES (PASS)' if stream_opened_successfully else 'NO (FAIL)'}")
    print(f"• Audio Frames Received:     {'YES (PASS)' if audio_frames_received else 'NO (FAIL)'} ({frames_received} frames)")
    print(f"• Peak Signal Amplitude:     {peak_amplitude:.6f}")
    print(f"• Audio Stored on Disk:      NO (Zero Audio Storage Confirmed)")
    print("=" * 65)

    if stream_opened_successfully and audio_frames_received:
        print("PHASE 0 MICROPHONE TEST: PASS\n")
        return True
    else:
        print("PHASE 0 MICROPHONE TEST: FAIL\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test microphone capture for Phase 0.")
    parser.add_argument("--device", type=int, default=None, help="Device index to test (default: system default)")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds to test capture (default: 2.0)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate (default: 16000)")
    args = parser.parse_args()

    success = test_microphone(device_id=args.device, duration_sec=args.duration, sample_rate=args.sample_rate)
    sys.exit(0 if success else 1)

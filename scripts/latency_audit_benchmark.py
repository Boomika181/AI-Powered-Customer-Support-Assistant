"""
scripts/latency_audit_benchmark.py

Controlled 10-call latency benchmark for Gemini Semantic Speaker-Role Classifier.
Instruments timing breakdown without modifying production code:
- Pre-request preparation time
- Gemini API HTTP/model generation duration
- Post-request parsing & threshold gating duration
- Total duration
Uses the production classify_speaker_role() path.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.speaker_role_classifier import (
    classify_speaker_role,
    SpeakerContinuityTracker,
    build_role_classifier_prompt,
    _strip_markdown_fences,
)


def run_benchmark():
    raw_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not raw_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    turns = [
        {"turn": 1, "text": "My WP400 has stopped working.", "raw_speaker": "A", "expected": "CUSTOMER"},
        {"turn": 2, "text": "Okay, what error code is displayed?", "raw_speaker": "B", "expected": "AGENT"},
        {"turn": 3, "text": "It is showing error E102.", "raw_speaker": "A", "expected": "CUSTOMER"},
        {"turn": 4, "text": "Please inspect the extraction breaker.", "raw_speaker": "B", "expected": "AGENT"},
        {"turn": 5, "text": "The extraction fan isn't running either.", "raw_speaker": "A", "expected": "CUSTOMER"},
        {"turn": 6, "text": "Let me check the technical troubleshooting manual.", "raw_speaker": "B", "expected": "AGENT"},
        {"turn": 7, "text": "What should I check next on the unit?", "raw_speaker": "A", "expected": "CUSTOMER"},
        {"turn": 8, "text": "Check whether filter cartridge WP4-FL-DE1 is blocked.", "raw_speaker": "B", "expected": "AGENT"},
        {"turn": 9, "text": "The filter cartridge looks completely clogged.", "raw_speaker": "A", "expected": "CUSTOMER"},
        {"turn": 10, "text": "Replace the filter and reset breaker CB-2.", "raw_speaker": "B", "expected": "AGENT"},
    ]

    print("==================================================")
    print("STARTING 10-CALL CONTROLLED LATENCY BENCHMARK")
    print(f"Model: {model_name}")
    print("==================================================\n")

    tracker = SpeakerContinuityTracker()
    recent_history = []
    results = []
    http_429_count = 0
    api_error_count = 0

    for item in turns:
        t_start = time.perf_counter()

        # Step A: Measure pre-request preparation
        t_pre_start = time.perf_counter()
        continuity_summary = tracker.get_continuity_summary()
        prompt = build_role_classifier_prompt(
            current_turn_text=item["text"],
            current_raw_speaker=item["raw_speaker"],
            recent_turns=recent_history,
            continuity_summary=continuity_summary
        )
        t_pre_end = time.perf_counter()
        pre_duration_ms = (t_pre_end - t_pre_start) * 1000.0

        # Step B: Call production classify_speaker_role()
        t_call_start = time.perf_counter()
        res = classify_speaker_role(
            current_turn={"text": item["text"], "raw_speaker": item["raw_speaker"]},
            recent_turns=recent_history,
            tracker=tracker
        )
        t_call_end = time.perf_counter()
        total_duration_ms = (t_call_end - t_call_start) * 1000.0

        # The internal latency recorded by classify_speaker_role
        internal_latency_ms = res["latency_ms"]

        if "429" in res.get("rationale", ""):
            http_429_count += 1
        elif "error" in res.get("rationale", "").lower():
            api_error_count += 1

        results.append({
            "turn": item["turn"],
            "text": item["text"],
            "role": res["role"],
            "expected": item["expected"],
            "confidence": res["confidence"],
            "prompt_chars": len(prompt),
            "pre_ms": pre_duration_ms,
            "total_ms": total_duration_ms,
            "internal_ms": internal_latency_ms,
            "rationale": res.get("rationale", "")
        })

        print(f"Call {item['turn']}/10: role={res['role']} (conf={res['confidence']:.2f}) | prompt_len={len(prompt)} chars | latency={total_duration_ms:.1f} ms")

        # Update history
        recent_history.append({
            "text": item["text"],
            "raw_speaker": item["raw_speaker"],
            "role": res["role"],
            "speaker": res["role"]
        })

    # Summary statistics
    total_latencies = [r["total_ms"] for r in results]
    min_lat = min(total_latencies)
    max_lat = max(total_latencies)
    avg_lat = sum(total_latencies) / len(total_latencies)
    sorted_lat = sorted(total_latencies)
    p50_lat = sorted_lat[len(sorted_lat) // 2]
    # 95th percentile
    p95_idx = int(0.95 * len(sorted_lat))
    p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

    avg_prompt_chars = sum(r["prompt_chars"] for r in results) / len(results)
    avg_pre_ms = sum(r["pre_ms"] for r in results) / len(results)

    print("\n==================================================")
    print("BENCHMARK METRICS SUMMARY")
    print("==================================================")
    print(f"Total calls: {len(results)}")
    print(f"Min latency: {min_lat:.1f} ms")
    print(f"P50 (Median) latency: {p50_lat:.1f} ms")
    print(f"Average latency: {avg_lat:.1f} ms")
    print(f"P95 latency: {p95_lat:.1f} ms")
    print(f"Max latency: {max_lat:.1f} ms")
    print(f"Average prompt character count: {avg_prompt_chars:.1f} chars (~{avg_prompt_chars/4:.0f} tokens)")
    print(f"Average local pre-processing time: {avg_pre_ms:.3f} ms")
    print(f"HTTP 429 errors: {http_429_count}")
    print(f"Other API errors: {api_error_count}")
    print("==================================================")

    # Breakdown per call
    print("\nIndividual Call Breakdown:")
    for r in results:
        print(f"Call {r['turn']:2d}: total={r['total_ms']:6.1f} ms | pre={r['pre_ms']:5.3f} ms | prompt={r['prompt_chars']:4d} chars | role={r['role']:8s} | rationale=\"{r['rationale'][:60]}...\"")


if __name__ == "__main__":
    run_benchmark()

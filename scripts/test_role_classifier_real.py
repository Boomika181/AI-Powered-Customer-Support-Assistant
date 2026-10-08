"""
scripts/test_role_classifier_real.py

Live API verification for Semantic Speaker-Role Classifier using Google Gemini 3.5 Flash Lite.
Runs:
1. Customer-First 4-turn conversation
2. Agent-First 4-turn conversation
3. Ambiguous short-turn test ("Okay.")
Records latency, structured JSON output, role validity, and accuracy metrics.
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
    get_confidence_threshold,
)


def run_live_verification():
    # --------------------------------------------------------------------------
    # 1. Environment Verification
    # --------------------------------------------------------------------------
    raw_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not raw_key or raw_key.startswith("your_"):
        print("ERROR: GEMINI_API_KEY is not configured or invalid in .env")
        sys.exit(1)

    model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    if model_name != "gemini-3.5-flash-lite":
        print(f"ERROR: Model is configured as '{model_name}', expected 'gemini-3.5-flash-lite'")
        sys.exit(1)

    threshold = get_confidence_threshold()

    print("==================================================")
    print("ENVIRONMENT VERIFIED")
    print(f"API Key Present: YES (length={len(raw_key)})")
    print(f"Configured Model: {model_name}")
    print(f"Confidence Threshold: {threshold:.2f}")
    print("==================================================\n")

    latencies = []
    http_429_errors = 0
    other_errors = 0
    total_calls = 0
    correct_calls = 0

    # --------------------------------------------------------------------------
    # 2. Customer-First Test (4 Turns)
    # --------------------------------------------------------------------------
    print("==================================================")
    print("2. REAL LIVE TEST — CUSTOMER FIRST")
    print("==================================================")

    tracker_cust_first = SpeakerContinuityTracker()
    recent_turns_cust = []

    cust_first_dialogue = [
        {
            "turn": 1,
            "raw_speaker": "A",
            "text": "My WP400 has stopped working and it is showing error E102.",
            "expected": "CUSTOMER"
        },
        {
            "turn": 2,
            "raw_speaker": "B",
            "text": "Okay, let me help you troubleshoot that.",
            "expected": "AGENT"
        },
        {
            "turn": 3,
            "raw_speaker": "A",
            "text": "The extraction fan isn't running either.",
            "expected": "CUSTOMER"
        },
        {
            "turn": 4,
            "raw_speaker": "B",
            "text": "Please check whether the extraction breaker is switched on.",
            "expected": "AGENT"
        }
    ]

    cust_first_correct = 0

    for item in cust_first_dialogue:
        total_calls += 1
        turn_data = {"text": item["text"], "raw_speaker": item["raw_speaker"]}
        
        res = classify_speaker_role(
            current_turn=turn_data,
            recent_turns=recent_turns_cust,
            tracker=tracker_cust_first
        )

        latencies.append(res["latency_ms"])
        if "429" in res.get("rationale", ""):
            http_429_errors += 1
        elif "error" in res.get("rationale", "").lower():
            other_errors += 1

        is_correct = (res["role"] == item["expected"])
        if is_correct:
            cust_first_correct += 1
            correct_calls += 1

        print(f"[ROLE CLASSIFICATION LIVE]")
        print(f"turn={item['turn']}")
        print(f"raw_speaker={item['raw_speaker']}")
        print(f"text=\"{item['text']}\"")
        print(f"model={model_name}")
        print(f"role={res['role']}")
        print(f"expected={item['expected']}")
        print(f"confidence={res['confidence']:.2f}")
        print(f"latency_ms={res['latency_ms']:.1f}")
        print(f"rationale=\"{res['rationale']}\"")
        print(f"match={'PASS' if is_correct else 'FAIL'}\n")

        # Update recent turns history
        recent_turns_cust.append({
            "text": item["text"],
            "raw_speaker": item["raw_speaker"],
            "role": res["role"],
            "speaker": res["role"]
        })

    # --------------------------------------------------------------------------
    # 3. Agent-First Test (4 Turns)
    # --------------------------------------------------------------------------
    print("==================================================")
    print("3. REAL LIVE TEST — AGENT FIRST")
    print("==================================================")

    tracker_agent_first = SpeakerContinuityTracker()
    recent_turns_agent = []

    agent_first_dialogue = [
        {
            "turn": 1,
            "raw_speaker": "A",
            "text": "Hello, technical support. How can I help you today?",
            "expected": "AGENT"
        },
        {
            "turn": 2,
            "raw_speaker": "B",
            "text": "My WP400 is showing error E102 and the extraction fan has stopped.",
            "expected": "CUSTOMER"
        },
        {
            "turn": 3,
            "raw_speaker": "A",
            "text": "Okay. Please check whether the extraction filter is blocked.",
            "expected": "AGENT"
        },
        {
            "turn": 4,
            "raw_speaker": "B",
            "text": "I can see that the filter is blocked.",
            "expected": "CUSTOMER"
        }
    ]

    agent_first_correct = 0

    for item in agent_first_dialogue:
        total_calls += 1
        turn_data = {"text": item["text"], "raw_speaker": item["raw_speaker"]}

        res = classify_speaker_role(
            current_turn=turn_data,
            recent_turns=recent_turns_agent,
            tracker=tracker_agent_first
        )

        latencies.append(res["latency_ms"])
        if "429" in res.get("rationale", ""):
            http_429_errors += 1
        elif "error" in res.get("rationale", "").lower():
            other_errors += 1

        is_correct = (res["role"] == item["expected"])
        if is_correct:
            agent_first_correct += 1
            correct_calls += 1

        print(f"[ROLE CLASSIFICATION LIVE]")
        print(f"turn={item['turn']}")
        print(f"raw_speaker={item['raw_speaker']}")
        print(f"text=\"{item['text']}\"")
        print(f"model={model_name}")
        print(f"role={res['role']}")
        print(f"expected={item['expected']}")
        print(f"confidence={res['confidence']:.2f}")
        print(f"latency_ms={res['latency_ms']:.1f}")
        print(f"rationale=\"{res['rationale']}\"")
        print(f"match={'PASS' if is_correct else 'FAIL'}\n")

        recent_turns_agent.append({
            "text": item["text"],
            "raw_speaker": item["raw_speaker"],
            "role": res["role"],
            "speaker": res["role"]
        })

    # --------------------------------------------------------------------------
    # 4. Ambiguous Turn Test ("Okay.")
    # --------------------------------------------------------------------------
    print("==================================================")
    print("4. AMBIGUOUS TURN TEST")
    print("==================================================")

    ambiguous_preceding = [
        {
            "text": "My machine has stopped and I need help.",
            "role": "CUSTOMER",
            "speaker": "CUSTOMER",
            "raw_speaker": "A"
        }
    ]
    ambiguous_turn = {"text": "Okay.", "raw_speaker": "B"}

    total_calls += 1
    res_ambiguous = classify_speaker_role(
        current_turn=ambiguous_turn,
        recent_turns=ambiguous_preceding
    )
    latencies.append(res_ambiguous["latency_ms"])
    if "429" in res_ambiguous.get("rationale", ""):
        http_429_errors += 1
    elif "error" in res_ambiguous.get("rationale", "").lower():
        other_errors += 1

    # In this context, "Okay." from Speaker B following customer problem can be AGENT or UNKNOWN,
    # but must NOT be falsely classified as high-confidence CUSTOMER.
    ambiguous_pass = (res_ambiguous["role"] in {"UNKNOWN", "AGENT"})
    if ambiguous_pass:
        correct_calls += 1

    print(f"[ROLE CLASSIFICATION LIVE - AMBIGUOUS]")
    print(f"context=\"My machine has stopped and I need help.\" (CUSTOMER)")
    print(f"turn=\"Okay.\" (Speaker B)")
    print(f"model={model_name}")
    print(f"role={res_ambiguous['role']}")
    print(f"confidence={res_ambiguous['confidence']:.2f}")
    print(f"latency_ms={res_ambiguous['latency_ms']:.1f}")
    print(f"rationale=\"{res_ambiguous['rationale']}\"")
    print(f"handling={'PASS (Safely handled as ' + res_ambiguous['role'] + ')' if ambiguous_pass else 'FAIL (Unsafely attributed to CUSTOMER)'}\n")

    # --------------------------------------------------------------------------
    # 6. Accuracy Summary
    # --------------------------------------------------------------------------
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    accuracy_pct = (correct_calls / total_calls) * 100.0 if total_calls else 0.0

    print("==================================================")
    print("ACCURACY SUMMARY")
    print("==================================================")
    print(f"Total live role classifications: {total_calls}")
    print(f"Correct expected classifications: {correct_calls}")
    print(f"Accuracy: {accuracy_pct:.1f}%")
    print(f"Customer-first accuracy: {cust_first_correct}/4")
    print(f"Agent-first accuracy: {agent_first_correct}/4")
    print(f"Ambiguous-turn handling: {'PASS' if ambiguous_pass else 'FAIL'}")
    print(f"Average Gemini role-classification latency: {avg_latency:.1f} ms")
    print(f"HTTP 429 errors: {http_429_errors}")
    print(f"Other API errors: {other_errors}")
    print("==================================================")


if __name__ == "__main__":
    run_live_verification()

"""
scripts/verify_bounded_context_live.py

Live verification script for bounded downstream customer context.
Executes the production PipelineCoordinator path with LIVE Gemini API,
real ChromaDB retrieval, real sentiment analysis, real categorization,
and real support suggestion generation.
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

from src.transcription import ConversationState
from src.pipeline_coordinator import PipelineCoordinator
from src.sentiment_analysis import analyze_sentiment
from src.query_categorization import classify_query
from src.rag_search import search_knowledge_base
from src.llm_suggestions import generate_support_suggestion


def run_live_verification():
    print("==================================================")
    print("STARTING LIVE BOUNDED CUSTOMER-CONTEXT VERIFICATION")
    print("==================================================")

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY is not configured.")
        sys.exit(1)

    # Recorded calls & timings for downstream
    captured_downstream_calls = {
        "sentiment": [],
        "category": [],
        "rag": [],
        "suggestion": []
    }

    latencies = {
        "sentiment": [],
        "category": [],
        "rag_embedding": [],
        "rag_retrieval": [],
        "rag_total": [],
        "suggestion": [],
        "total_downstream": []
    }

    http_429_count = 0
    api_error_count = 0

    # Wrapped real functions to measure timing and capture exact inputs
    def instrumented_sentiment(text: str):
        nonlocal http_429_count, api_error_count
        captured_downstream_calls["sentiment"].append(text)
        t0 = time.perf_counter()
        try:
            res = analyze_sentiment(text)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies["sentiment"].append(lat)
            return res
        except Exception as e:
            if "429" in str(e):
                http_429_count += 1
            else:
                api_error_count += 1
            raise

    def instrumented_category(text: str):
        nonlocal http_429_count, api_error_count
        captured_downstream_calls["category"].append(text)
        t0 = time.perf_counter()
        try:
            res = classify_query(text)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies["category"].append(lat)
            return res
        except Exception as e:
            if "429" in str(e):
                http_429_count += 1
            else:
                api_error_count += 1
            raise

    def instrumented_rag(text: str, top_k: int = 3):
        nonlocal http_429_count, api_error_count
        captured_downstream_calls["rag"].append(text)
        t0 = time.perf_counter()
        try:
            res = search_knowledge_base(text, top_k=top_k)
            total_lat = (time.perf_counter() - t0) * 1000.0
            latencies["rag_total"].append(total_lat)
            if res and isinstance(res, list) and len(res) > 0:
                latencies["rag_embedding"].append(res[0].get("embedding_latency_ms", 0.0))
                latencies["rag_retrieval"].append(res[0].get("retrieval_latency_ms", 0.0))
            return res
        except Exception as e:
            if "429" in str(e):
                http_429_count += 1
            else:
                api_error_count += 1
            raise

    def instrumented_suggestion(conversation_text: str, retrieved_context, category=None):
        nonlocal http_429_count, api_error_count
        captured_downstream_calls["suggestion"].append(conversation_text)
        t0 = time.perf_counter()
        try:
            res = generate_support_suggestion(
                conversation_text=conversation_text,
                retrieved_context=retrieved_context,
                category=category
            )
            lat = (time.perf_counter() - t0) * 1000.0
            latencies["suggestion"].append(lat)
            return res
        except Exception as e:
            if "429" in str(e):
                http_429_count += 1
            else:
                api_error_count += 1
            raise

    # Instantiate coordinator with production path (async_downstream=False for sequential measurement)
    coordinator = PipelineCoordinator(
        sentiment_fn=instrumented_sentiment,
        category_fn=instrumented_category,
        rag_fn=instrumented_rag,
        suggestion_fn=instrumented_suggestion,
        max_customer_turns=4,
        async_downstream=False
    )

    state = coordinator.conversation_state

    # 8-turn test conversation
    conversation_turns = [
        {"turn": 1, "speaker": "Customer", "raw_speaker": "A", "text": "My WP400 has stopped working."},
        {"turn": 2, "speaker": "Agent",    "raw_speaker": "B", "text": "Okay, let me help you troubleshoot it."},
        {"turn": 3, "speaker": "Customer", "raw_speaker": "A", "text": "It is showing error E102."},
        {"turn": 4, "speaker": "Agent",    "raw_speaker": "B", "text": "Please check whether the extraction breaker is on."},
        {"turn": 5, "speaker": "Customer", "raw_speaker": "A", "text": "The extraction fan isn't running."},
        {"turn": 6, "speaker": "Agent",    "raw_speaker": "B", "text": "Please inspect the filter."},
        {"turn": 7, "speaker": "Customer", "raw_speaker": "A", "text": "The filter is blocked."},
        {"turn": 8, "speaker": "Customer", "raw_speaker": "A", "text": "The machine still won't operate."},
    ]

    last_events = []
    coordinator.add_event_listener(lambda ev: last_events.append(ev))

    print("\nExecuting live turns sequentially...\n")

    for t in conversation_turns:
        print(f"--- Turn {t['turn']} ({t['speaker']}) ---")
        print(f"Text: \"{t['text']}\"")
        seg = state.add_final_segment(text=t["text"], raw_speaker=t["raw_speaker"])
        
        t_turn_start = time.perf_counter()
        coordinator._handle_speech_break(seg, state)
        t_turn_elapsed = (time.perf_counter() - t_turn_start) * 1000.0
        print(f"Role assigned: {seg.get('role')} (confidence={seg.get('role_confidence', 0.0):.2f})")
        print(f"Turn dispatch completed in: {t_turn_elapsed:.1f} ms\n")
        
        # Pacing delay between turns to respect free-tier 15 RPM quota
        time.sleep(4.5)

    # =========================================================================
    # Evaluation on Turn 8 (Final Customer Turn)
    # =========================================================================
    final_sentiment_context = captured_downstream_calls["sentiment"][-1]
    final_category_context = captured_downstream_calls["category"][-1]
    final_rag_context = captured_downstream_calls["rag"][-1]
    final_suggestion_context = captured_downstream_calls["suggestion"][-1]

    expected_bounded_context = (
        "It is showing error E102. "
        "The extraction fan isn't running. "
        "The filter is blocked. "
        "The machine still won't operate."
    )

    all_same = (
        final_sentiment_context == final_category_context ==
        final_rag_context == final_suggestion_context ==
        expected_bounded_context
    )

    has_turn1_customer = "My WP400 has stopped working" in final_sentiment_context
    has_agent_turn2 = "troubleshoot it" in final_sentiment_context
    has_agent_turn4 = "extraction breaker is on" in final_sentiment_context
    has_agent_turn6 = "inspect the filter" in final_sentiment_context

    # Get latest result payloads
    latest_sentiment_ev = next((e for e in reversed(last_events) if e.get("type") == "sentiment_update"), {})
    latest_category_ev = next((e for e in reversed(last_events) if e.get("type") == "category_update"), {})
    latest_suggestion_ev = next((e for e in reversed(last_events) if e.get("type") == "suggestion_update"), {})

    # Compute final turn downstream latency
    final_sent_lat = latencies["sentiment"][-1] if latencies["sentiment"] else 0.0
    final_cat_lat = latencies["category"][-1] if latencies["category"] else 0.0
    final_rag_total_lat = latencies["rag_total"][-1] if latencies["rag_total"] else 0.0
    final_rag_embed_lat = latencies["rag_embedding"][-1] if latencies["rag_embedding"] else 0.0
    final_rag_retrieval_lat = latencies["rag_retrieval"][-1] if latencies["rag_retrieval"] else 0.0
    final_sugg_lat = latencies["suggestion"][-1] if latencies["suggestion"] else 0.0
    final_total_downstream = final_sent_lat + final_cat_lat + final_rag_total_lat + final_sugg_lat

    print("\n==================================================")
    print("LIVE VERIFICATION ANALYSIS")
    print("==================================================")
    print(f"Total state segments preserved: {len(state.segments)} (Expected: 8)")
    print(f"Full transcript contains all turns: {'My WP400 has stopped working' in state.get_full_transcript()}")
    print(f"All 4 downstream modules received identical bounded context: {all_same}")
    print(f"Turn 1 Customer excluded from bounded context: {not has_turn1_customer}")
    print(f"Agent turns excluded from bounded context: {not (has_agent_turn2 or has_agent_turn4 or has_agent_turn6)}")
    print("\nExact Bounded Context String:")
    print(f"\"{final_sentiment_context}\"")
    print("\nLatest Downstream Results:")
    print(f"Sentiment: {latest_sentiment_ev.get('sentiment')} ({latest_sentiment_ev.get('confidence', 0)*100:.0f}% conf)")
    print(f"Category: {latest_category_ev.get('category')} ({latest_category_ev.get('confidence', 0)*100:.0f}% conf)")
    print(f"Suggestion: {latest_suggestion_ev.get('suggestion')}")
    print(f"Grounded: {latest_suggestion_ev.get('is_grounded')}")
    print(f"Sources: {latest_suggestion_ev.get('sources')}")
    print(f"\nTimings for Final Turn Downstream Stages:")
    print(f"Sentiment Latency: {final_sent_lat:.1f} ms")
    print(f"Category Latency: {final_cat_lat:.1f} ms")
    print(f"RAG Embedding Latency: {final_rag_embed_lat:.1f} ms")
    print(f"RAG Retrieval Latency: {final_rag_retrieval_lat:.1f} ms")
    print(f"RAG Total Latency: {final_rag_total_lat:.1f} ms")
    print(f"Suggestion Latency: {final_sugg_lat:.1f} ms")
    print(f"Total Downstream Latency: {final_total_downstream:.1f} ms")
    print(f"HTTP 429 Errors: {http_429_count}")
    print(f"API Errors: {api_error_count}")


if __name__ == "__main__":
    run_live_verification()

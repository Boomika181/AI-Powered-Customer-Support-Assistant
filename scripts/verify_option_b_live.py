"""
scripts/verify_option_b_live.py

Live verification of Option B:
Consolidated Sentiment Analysis + Query Categorization into ONE Gemini call,
running concurrently with ChromaDB RAG retrieval, followed by Support Suggestion Generation.

Uses real production Gemini API and ChromaDB.
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline_coordinator import PipelineCoordinator
from src.transcription import ConversationState
from src.customer_context_analyzer import analyze_customer_context
from src.rag_search import search_knowledge_base
from src.llm_suggestions import generate_support_suggestion


def run_live_verification():
    print("=" * 60)
    print("LIVE VERIFICATION: CONSOLIDATED SENTIMENT + CATEGORY (OPTION B)")
    print("=" * 60)

    customer_context = (
        "My WP400 is showing error E102. The extraction fan isn't running. "
        "The filter is blocked. The machine still won't operate."
    )
    print(f"Customer Context:\n\"{customer_context}\"\n")

    timings = {}
    errors = []
    counts_429 = 0

    # Timing wrapper for combined analysis
    def timed_context_analyzer(text):
        nonlocal counts_429
        t0 = time.perf_counter()
        try:
            res = analyze_customer_context(text)
            timings["combined_analysis_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        except Exception as e:
            timings["combined_analysis_ms"] = (time.perf_counter() - t0) * 1000.0
            if "429" in str(e):
                counts_429 += 1
            errors.append(f"Combined analysis error: {e}")
            raise

    # Timing wrapper for RAG
    def timed_rag(text, top_k=3):
        nonlocal counts_429
        t0 = time.perf_counter()
        try:
            res = search_knowledge_base(text, top_k=top_k)
            timings["rag_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        except Exception as e:
            timings["rag_ms"] = (time.perf_counter() - t0) * 1000.0
            if "429" in str(e):
                counts_429 += 1
            errors.append(f"RAG error: {e}")
            raise

    # Timing wrapper for Suggestions
    def timed_suggestion(conversation_text, retrieved_context, category):
        nonlocal counts_429
        t0 = time.perf_counter()
        try:
            res = generate_support_suggestion(
                conversation_text=conversation_text,
                retrieved_context=retrieved_context,
                category=category
            )
            timings["suggestion_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        except Exception as e:
            timings["suggestion_ms"] = (time.perf_counter() - t0) * 1000.0
            if "429" in str(e):
                counts_429 += 1
            errors.append(f"Suggestion error: {e}")
            raise

    coordinator = PipelineCoordinator(
        context_analyzer_fn=timed_context_analyzer,
        rag_fn=timed_rag,
        suggestion_fn=timed_suggestion,
        async_downstream=False
    )

    captured_events = []
    coordinator.add_event_listener(lambda ev: captured_events.append(ev))

    state = ConversationState(session_id="option_b_live_session")
    segment = state.add_final_segment(
        text=customer_context,
        speaker="Customer",
        raw_speaker="A"
    )
    segment["role"] = "CUSTOMER"

    print("Executing downstream processing pipeline (2-way parallel: Combined Analysis + RAG)...")
    t_downstream_start = time.perf_counter()
    coordinator._process_downstream_turn(segment, state)
    t_downstream_end = time.perf_counter()

    total_downstream_ms = (t_downstream_end - t_downstream_start) * 1000.0
    timings["total_downstream_ms"] = total_downstream_ms

    parallel_phase_ms = total_downstream_ms - timings.get("suggestion_ms", 0.0)
    timings["parallel_phase_ms"] = parallel_phase_ms

    sentiment_event = next((ev for ev in captured_events if ev["type"] == "sentiment_update"), {})
    category_event = next((ev for ev in captured_events if ev["type"] == "category_update"), {})
    suggestion_event = next((ev for ev in captured_events if ev["type"] == "suggestion_update"), {})

    print("\n" + "-" * 50)
    print("LIVE VERIFICATION RESULTS (OPTION B)")
    print("-" * 50)
    print(f"Sentiment Event:   {sentiment_event.get('sentiment')} (conf={sentiment_event.get('confidence')})")
    print(f"   Rationale:      {sentiment_event.get('rationale')}")
    print(f"Category Event:    {category_event.get('category')} (conf={category_event.get('confidence')})")
    print(f"   Rationale:      {category_event.get('rationale')}")
    print(f"Grounded Status:   {suggestion_event.get('is_grounded')}")
    print(f"RAG Sources Count: {len(suggestion_event.get('sources', []))}")
    for idx, src in enumerate(suggestion_event.get('sources', []), 1):
        print(f"   [{idx}] {src.get('source_file')} | {src.get('document_reference')} | Page {src.get('page')}")
    print(f"Suggestion:        {suggestion_event.get('suggestion')}")

    print("\nLATENCY MEASUREMENTS:")
    print(f"  Combined Sentiment + Category Latency: {timings.get('combined_analysis_ms', 0):.1f} ms")
    print(f"  RAG Latency (Embedding + ChromaDB):     {timings.get('rag_ms', 0):.1f} ms")
    print(f"  Parallel Pre-Suggestion Phase Latency:  {timings.get('parallel_phase_ms', 0):.1f} ms")
    print(f"  Suggestion Latency:                     {timings.get('suggestion_ms', 0):.1f} ms")
    print(f"  TOTAL Downstream Latency:               {timings.get('total_downstream_ms', 0):.1f} ms")

    print("\nRELIABILITY:")
    print(f"  429 Count:  {counts_429}")
    print(f"  API Errors: {len(errors)}")

    return {
        "timings": timings,
        "sentiment_event": sentiment_event,
        "category_event": category_event,
        "suggestion_event": suggestion_event,
        "counts_429": counts_429,
        "errors": errors
    }


if __name__ == "__main__":
    run_live_verification()

"""
scripts/verify_parallel_downstream_live.py

Live verification of concurrent downstream execution in PipelineCoordinator:
Sentiment Analysis, Query Categorization, and RAG Knowledge-Base Retrieval run concurrently,
followed by Support Suggestion Generation.

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
from src.sentiment_analysis import analyze_sentiment
from src.query_categorization import classify_query
from src.rag_search import search_knowledge_base
from src.llm_suggestions import generate_support_suggestion


def run_live_verification():
    print("=" * 60)
    print("LIVE VERIFICATION: PARALLEL DOWNSTREAM EXECUTION")
    print("=" * 60)

    customer_context = (
        "My WP400 is showing error E102. The extraction fan isn't running. "
        "The filter is blocked. The machine still won't operate."
    )
    print(f"Customer Context:\n\"{customer_context}\"\n")

    # Metrics trackers
    timings = {}
    errors = []
    counts_429 = 0

    # Wrap functions to measure individual execution times while calling real implementations
    def timed_sentiment(text):
        nonlocal counts_429
        t0 = time.perf_counter()
        try:
            res = analyze_sentiment(text)
            timings["individual_sentiment_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        except Exception as e:
            timings["individual_sentiment_ms"] = (time.perf_counter() - t0) * 1000.0
            if "429" in str(e):
                counts_429 += 1
            errors.append(f"Sentiment error: {e}")
            raise

    def timed_category(text):
        nonlocal counts_429
        t0 = time.perf_counter()
        try:
            res = classify_query(text)
            timings["individual_category_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        except Exception as e:
            timings["individual_category_ms"] = (time.perf_counter() - t0) * 1000.0
            if "429" in str(e):
                counts_429 += 1
            errors.append(f"Category error: {e}")
            raise

    def timed_rag(text, top_k=3):
        nonlocal counts_429
        t0 = time.perf_counter()
        try:
            res = search_knowledge_base(text, top_k=top_k)
            timings["individual_rag_ms"] = (time.perf_counter() - t0) * 1000.0
            return res
        except Exception as e:
            timings["individual_rag_ms"] = (time.perf_counter() - t0) * 1000.0
            if "429" in str(e):
                counts_429 += 1
            errors.append(f"RAG error: {e}")
            raise

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

    # Initialize coordinator with timed real functions
    coordinator = PipelineCoordinator(
        sentiment_fn=timed_sentiment,
        category_fn=timed_category,
        rag_fn=timed_rag,
        suggestion_fn=timed_suggestion,
        async_downstream=False
    )

    captured_events = []
    coordinator.add_event_listener(lambda ev: captured_events.append(ev))

    # Set up ConversationState
    state = ConversationState(session_id="live_verification_session")
    segment = state.add_final_segment(
        text=customer_context,
        speaker="Customer",
        raw_speaker="A"
    )
    segment["role"] = "CUSTOMER"

    print("Executing downstream processing pipeline concurrently...")
    t_downstream_start = time.perf_counter()
    coordinator._process_downstream_turn(segment, state)
    t_downstream_end = time.perf_counter()

    total_downstream_ms = (t_downstream_end - t_downstream_start) * 1000.0
    timings["total_downstream_ms"] = total_downstream_ms

    # Parallel stage latency is total downstream minus suggestion time
    parallel_stage_ms = total_downstream_ms - timings.get("suggestion_ms", 0.0)
    timings["parallel_stage_ms"] = parallel_stage_ms

    # Extract event outputs
    sentiment_event = next((ev for ev in captured_events if ev["type"] == "sentiment_update"), {})
    category_event = next((ev for ev in captured_events if ev["type"] == "category_update"), {})
    suggestion_event = next((ev for ev in captured_events if ev["type"] == "suggestion_update"), {})

    print("\n" + "-" * 50)
    print("LIVE VERIFICATION RESULTS")
    print("-" * 50)
    print(f"Sentiment Result:  {sentiment_event.get('sentiment')} (conf={sentiment_event.get('confidence')})")
    print(f"Category Result:   {category_event.get('category')} (conf={category_event.get('confidence')})")
    print(f"Grounded Status:   {suggestion_event.get('is_grounded')}")
    print(f"RAG Sources Count: {len(suggestion_event.get('sources', []))}")
    for idx, src in enumerate(suggestion_event.get('sources', []), 1):
        print(f"   [{idx}] {src.get('source_file')} | {src.get('document_reference')} | Page {src.get('page')}")
    print(f"Suggestion:        {suggestion_event.get('suggestion')}")
    print("\nLATENCY MEASUREMENTS:")
    print(f"  Individual Sentiment:   {timings.get('individual_sentiment_ms', 0):.1f} ms")
    print(f"  Individual Category:    {timings.get('individual_category_ms', 0):.1f} ms")
    print(f"  Individual RAG:         {timings.get('individual_rag_ms', 0):.1f} ms")
    print(f"  Concurrent Phase Time:  {timings.get('parallel_stage_ms', 0):.1f} ms")
    print(f"  Suggestion Time:        {timings.get('suggestion_ms', 0):.1f} ms")
    print(f"  TOTAL Downstream Time:  {timings.get('total_downstream_ms', 0):.1f} ms")
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

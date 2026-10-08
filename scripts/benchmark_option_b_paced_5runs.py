"""
scripts/benchmark_option_b_paced_5runs.py

5-Run Live Latency Benchmark of Option B with 22s Inter-Iteration Pacing.
Ensures generation requests (2 per run) stay well within the Free Tier 15 RPM quota (5.4 RPM),
providing clean, unthrottled per-iteration latency measurements.
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


def run_paced_benchmark():
    print("=" * 60)
    print("OPTION B — 5-RUN LIVE BENCHMARK (PACED TO PREVENT 15 RPM BACKOFF)")
    print("=" * 60)

    customer_context = (
        "My WP400 is showing error E102. The extraction fan isn't running. "
        "The filter is blocked. The machine still won't operate."
    )

    runs = []
    total_429 = 0
    total_errors = 0
    total_retries = 0

    # Ensure clean starting bucket
    print("Waiting 20 seconds for quota window to clear before Run 1...")
    time.sleep(20.0)

    for i in range(1, 6):
        print(f"\n>>> Running Benchmark Iteration {i}/5...")
        run_record = {
            "run": i,
            "429": 0,
            "errors": 0,
            "retries": 0
        }

        timings = {}

        def timed_context_analyzer(text):
            nonlocal total_429, total_errors
            t0 = time.perf_counter()
            try:
                res = analyze_customer_context(text)
                timings["combined_ms"] = (time.perf_counter() - t0) * 1000.0
                return res
            except Exception as e:
                timings["combined_ms"] = (time.perf_counter() - t0) * 1000.0
                if "429" in str(e):
                    run_record["429"] += 1
                    total_429 += 1
                else:
                    run_record["errors"] += 1
                    total_errors += 1
                raise

        def timed_rag(text, top_k=3):
            nonlocal total_429, total_errors
            t0 = time.perf_counter()
            try:
                res = search_knowledge_base(text, top_k=top_k)
                timings["rag_ms"] = (time.perf_counter() - t0) * 1000.0
                return res
            except Exception as e:
                timings["rag_ms"] = (time.perf_counter() - t0) * 1000.0
                if "429" in str(e):
                    run_record["429"] += 1
                    total_429 += 1
                else:
                    run_record["errors"] += 1
                    total_errors += 1
                raise

        def timed_suggestion(conversation_text, retrieved_context, category):
            nonlocal total_429, total_errors
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
                    run_record["429"] += 1
                    total_429 += 1
                else:
                    run_record["errors"] += 1
                    total_errors += 1
                raise

        coordinator = PipelineCoordinator(
            context_analyzer_fn=timed_context_analyzer,
            rag_fn=timed_rag,
            suggestion_fn=timed_suggestion,
            async_downstream=False
        )

        captured_events = []
        coordinator.add_event_listener(lambda ev: captured_events.append(ev))

        state = ConversationState(session_id=f"benchmark_paced_run_{i}")
        segment = state.add_final_segment(
            text=customer_context,
            speaker="Customer",
            raw_speaker="A"
        )
        segment["role"] = "CUSTOMER"

        t_start = time.perf_counter()
        coordinator._process_downstream_turn(segment, state)
        t_end = time.perf_counter()

        total_downstream_ms = (t_end - t_start) * 1000.0
        sugg_ms = timings.get("suggestion_ms", 0.0)
        parallel_phase_ms = total_downstream_ms - sugg_ms

        sentiment_event = next((ev for ev in captured_events if ev["type"] == "sentiment_update"), {})
        category_event = next((ev for ev in captured_events if ev["type"] == "category_update"), {})
        suggestion_event = next((ev for ev in captured_events if ev["type"] == "suggestion_update"), {})

        run_record["combined_ms"] = timings.get("combined_ms", 0.0)
        run_record["rag_ms"] = timings.get("rag_ms", 0.0)
        run_record["parallel_phase_ms"] = parallel_phase_ms
        run_record["suggestion_ms"] = sugg_ms
        run_record["total_ms"] = total_downstream_ms

        run_record["sentiment"] = sentiment_event.get("sentiment")
        run_record["sentiment_conf"] = sentiment_event.get("confidence")
        run_record["category"] = category_event.get("category")
        run_record["category_conf"] = category_event.get("confidence")
        run_record["is_grounded"] = suggestion_event.get("is_grounded")
        run_record["sources_count"] = len(suggestion_event.get("sources", []))

        runs.append(run_record)

        print(f"Run {i} Results:")
        print(f"  Combined Classification: {run_record['combined_ms']:.1f} ms")
        print(f"  RAG Retrieval:           {run_record['rag_ms']:.1f} ms")
        print(f"  Parallel Phase:          {run_record['parallel_phase_ms']:.1f} ms")
        print(f"  Suggestion:              {run_record['suggestion_ms']:.1f} ms")
        print(f"  Total Downstream:        {run_record['total_ms']:.1f} ms")
        print(f"  Sentiment: {run_record['sentiment']} ({run_record['sentiment_conf']}) | Category: {run_record['category']} ({run_record['category_conf']}) | Grounded: {run_record['is_grounded']}")

        if i < 5:
            print("  [Pacing delay: cooling down for 22.0s to maintain ~5.4 RPM < 15 RPM limit...]")
            time.sleep(22.0)

    totals = [r["total_ms"] for r in runs]
    sorted_totals = sorted(totals)
    min_total = sorted_totals[0]
    max_total = sorted_totals[-1]
    p50_total = sorted_totals[2]
    p95_total = sorted_totals[-1]
    avg_total = sum(totals) / len(totals)

    avg_combined = sum(r["combined_ms"] for r in runs) / len(runs)
    avg_rag = sum(r["rag_ms"] for r in runs) / len(runs)
    avg_parallel = sum(r["parallel_phase_ms"] for r in runs) / len(runs)
    avg_suggestion = sum(r["suggestion_ms"] for r in runs) / len(runs)

    runs_under_2s = sum(1 for t in totals if t < 2000.0)
    pct_under_2s = (runs_under_2s / len(totals)) * 100.0

    baseline_parallel = 3963.5
    baseline_original = 5773.0

    imp_vs_parallel = baseline_parallel - avg_total
    pct_vs_parallel = (imp_vs_parallel / baseline_parallel) * 100.0

    imp_vs_original = baseline_original - avg_total
    pct_vs_original = (imp_vs_original / baseline_original) * 100.0

    print("\n" + "=" * 60)
    print("PACED BENCHMARK AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"Combined classification avg: {avg_combined:.1f} ms")
    print(f"RAG avg:                     {avg_rag:.1f} ms")
    print(f"Parallel phase avg:          {avg_parallel:.1f} ms")
    print(f"Suggestion avg:              {avg_suggestion:.1f} ms")
    print(f"Total average:               {avg_total:.1f} ms")
    print(f"Total P50:                   {p50_total:.1f} ms")
    print(f"Total P95:                   {p95_total:.1f} ms")
    print(f"Total minimum:               {min_total:.1f} ms")
    print(f"Total maximum:               {max_total:.1f} ms")
    print(f"429s:                        {total_429}")
    print(f"API errors:                  {total_errors}")
    print(f"Retries:                     {total_retries}")
    print(f"Runs under 2 seconds:        {runs_under_2s}")
    print(f"Percentage under 2s:         {pct_under_2s:.1f}%")
    print(f"Improvement vs 3963.5 ms:    {imp_vs_parallel:.1f} ms ({pct_vs_parallel:.1f}%)")
    print(f"Improvement vs 5773.0 ms:    {imp_vs_original:.1f} ms ({pct_vs_original:.1f}%)")

    results = {
        "runs": runs,
        "avg_combined": avg_combined,
        "avg_rag": avg_rag,
        "avg_parallel": avg_parallel,
        "avg_suggestion": avg_suggestion,
        "avg_total": avg_total,
        "p50_total": p50_total,
        "p95_total": p95_total,
        "min_total": min_total,
        "max_total": max_total,
        "total_429": total_429,
        "total_errors": total_errors,
        "total_retries": total_retries,
        "runs_under_2s": runs_under_2s,
        "pct_under_2s": pct_under_2s,
        "imp_vs_parallel": imp_vs_parallel,
        "pct_vs_parallel": pct_vs_parallel,
        "imp_vs_original": imp_vs_original,
        "pct_vs_original": pct_vs_original
    }

    with open(PROJECT_ROOT / "scripts" / "option_b_paced_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_paced_benchmark()

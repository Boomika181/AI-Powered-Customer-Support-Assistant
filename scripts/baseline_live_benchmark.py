"""
scripts/baseline_live_benchmark.py

Comprehensive Baseline Live Latency Benchmark for Phase 2 Production Pipeline.
Measures real execution latencies, input/output payload sizes, reliability,
and stage contributions using existing production implementations.
"""

import os
import sys
import time
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.speaker_role_classifier import (
    classify_speaker_role,
    SpeakerContinuityTracker,
    build_role_classifier_prompt
)
from src.sentiment_analysis import analyze_sentiment
from src.query_categorization import classify_query
from src.rag_search import search_knowledge_base
from src.llm_suggestions import (
    generate_support_suggestion,
    _format_context_for_prompt
)


def run_benchmark():
    print("=" * 60)
    print("BASELINE LIVE LATENCY BENCHMARK — PHASE 2 PRODUCTION PIPELINE")
    print("=" * 60)

    python_version = platform.python_version()
    gen_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    emb_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    vdb = "ChromaDB (PersistentClient)"

    print(f"Python: {python_version}")
    print(f"Gemini generation model: {gen_model}")
    print(f"Embedding model: {emb_model}")
    print(f"Vector database: {vdb}")
    print()

    # =========================================================================
    # PART 1: SPEAKER ROLE CLASSIFIER (10 CALLS)
    # =========================================================================
    print("---------------------------------------------------------")
    print("PART 1: SPEAKER ROLE CLASSIFIER (10 LIVE CALLS)")
    print("---------------------------------------------------------")

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

    tracker = SpeakerContinuityTracker()
    history = []
    role_results = []
    role_429 = 0
    role_errors = 0
    role_prompt_sizes = []
    role_output_sizes = []

    for t in turns:
        prompt_str = build_role_classifier_prompt(
            current_turn_text=t["text"],
            current_raw_speaker=t["raw_speaker"],
            recent_turns=history,
            continuity_summary=tracker.get_continuity_summary()
        )
        role_prompt_sizes.append(len(prompt_str))

        t0 = time.perf_counter()
        res = classify_speaker_role(
            current_turn={"text": t["text"], "raw_speaker": t["raw_speaker"]},
            recent_turns=history,
            tracker=tracker
        )
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        rationale = res.get("rationale", "")
        role_output_sizes.append(len(rationale) + 40) # Approximate JSON output size
        is_429 = "429" in rationale
        is_err = "error" in rationale.lower() and not is_429 and res["role"] == "UNKNOWN"

        if is_429:
            role_429 += 1
        if is_err:
            role_errors += 1

        is_match = (res["role"] == t["expected"])

        role_results.append({
            "turn": t["turn"],
            "text": t["text"],
            "expected": t["expected"],
            "role": res["role"],
            "confidence": res["confidence"],
            "latency": t_elapsed,
            "match": is_match,
            "prompt_len": len(prompt_str),
            "output_len": len(rationale),
            "is_429": is_429,
            "is_err": is_err
        })

        print(f"Call {t['turn']:2d}: role={res['role']:8s} (exp={t['expected']:8s}) | conf={res['confidence']:.2f} | lat={t_elapsed:6.1f} ms | match={is_match}")

        # Update history and tracker
        history.append({
            "text": t["text"],
            "raw_speaker": t["raw_speaker"],
            "role": res["role"],
            "speaker": res["role"]
        })
        tracker.record_turn(t["raw_speaker"], res["role"], res["confidence"])

        time.sleep(4.2) # Pacing to stay under 15 RPM

    role_latencies = [r["latency"] for r in role_results]
    role_accuracy = sum(1 for r in role_results if r["match"]) / len(role_results) * 100.0

    sorted_role_lat = sorted(role_latencies)
    role_min = sorted_role_lat[0]
    role_max = sorted_role_lat[-1]
    role_avg = sum(role_latencies) / len(role_latencies)
    role_p50 = sorted_role_lat[len(sorted_role_lat) // 2]
    p95_idx = int(0.95 * len(sorted_role_lat))
    role_p95 = sorted_role_lat[min(p95_idx, len(sorted_role_lat) - 1)]

    print(f"\nRole Classifier Summary: Min={role_min:.1f}ms, P50={role_p50:.1f}ms, Avg={role_avg:.1f}ms, P95={role_p95:.1f}ms, Max={role_max:.1f}ms, Accuracy={role_accuracy:.1f}%\n")

    # =========================================================================
    # PART 2 & 3: DOWNSTREAM ANALYSIS (5 FULL ITERATIONS)
    # =========================================================================
    print("---------------------------------------------------------")
    print("PART 2 & 3: DOWNSTREAM ANALYSIS (5 COMPLETE ITERATIONS)")
    print("---------------------------------------------------------")

    customer_context = (
        "My WP400 is showing error E102. The extraction fan isn't running. "
        "The filter is blocked. The machine still won't operate."
    )

    iterations = []
    total_429 = role_429
    total_api_errors = role_errors
    total_gemini_calls = len(turns) # 10 from role classifier

    sent_input_sizes = []
    sent_output_sizes = []
    cat_input_sizes = []
    cat_output_sizes = []
    rag_input_sizes = []
    sugg_input_sizes = []
    sugg_output_sizes = []

    for i in range(1, 6):
        print(f"\n>>> Running Downstream Iteration {i}/5...")
        iter_record = {"iteration": i, "429": 0, "api_errors": 0}

        # Stage 1: Role classification latency on the final turn
        t0_role = time.perf_counter()
        role_res = classify_speaker_role(
            current_turn={"text": "The machine still won't operate.", "raw_speaker": "A"},
            recent_turns=history[-4:],
            tracker=tracker
        )
        lat_role = (time.perf_counter() - t0_role) * 1000.0
        iter_record["role_lat"] = lat_role
        total_gemini_calls += 1
        time.sleep(4.2)

        # Stage 2: Sentiment Analysis
        sent_input_sizes.append(len(customer_context))
        t0_sent = time.perf_counter()
        try:
            sent_res = analyze_sentiment(customer_context)
            lat_sent = (time.perf_counter() - t0_sent) * 1000.0
            iter_record["sent_lat"] = lat_sent
            iter_record["sentiment"] = sent_res.get("sentiment")
            sent_output_sizes.append(len(str(sent_res)))
        except Exception as e:
            lat_sent = (time.perf_counter() - t0_sent) * 1000.0
            iter_record["sent_lat"] = lat_sent
            if "429" in str(e):
                iter_record["429"] += 1
                total_429 += 1
            else:
                iter_record["api_errors"] += 1
                total_api_errors += 1
            sent_res = {}
        total_gemini_calls += 1
        time.sleep(4.2)

        # Stage 3: Query Categorization
        cat_input_sizes.append(len(customer_context))
        t0_cat = time.perf_counter()
        try:
            cat_res = classify_query(customer_context)
            lat_cat = (time.perf_counter() - t0_cat) * 1000.0
            iter_record["cat_lat"] = lat_cat
            iter_record["category"] = cat_res.get("category")
            cat_output_sizes.append(len(str(cat_res)))
        except Exception as e:
            lat_cat = (time.perf_counter() - t0_cat) * 1000.0
            iter_record["cat_lat"] = lat_cat
            if "429" in str(e):
                iter_record["429"] += 1
                total_429 += 1
            else:
                iter_record["api_errors"] += 1
                total_api_errors += 1
            cat_res = {}
        total_gemini_calls += 1
        time.sleep(4.2)

        # Stage 4: RAG Search (Gemini Embedding 2 + ChromaDB retrieval)
        rag_input_sizes.append(len(customer_context))
        t0_rag = time.perf_counter()
        try:
            rag_docs = search_knowledge_base(customer_context, top_k=3)
            lat_rag_total = (time.perf_counter() - t0_rag) * 1000.0
            iter_record["rag_total_lat"] = lat_rag_total
            if rag_docs and len(rag_docs) > 0:
                iter_record["rag_embed_lat"] = rag_docs[0].get("embedding_latency_ms", 0.0)
                iter_record["rag_retrieval_lat"] = rag_docs[0].get("retrieval_latency_ms", 0.0)
            else:
                iter_record["rag_embed_lat"] = 0.0
                iter_record["rag_retrieval_lat"] = lat_rag_total
        except Exception as e:
            lat_rag_total = (time.perf_counter() - t0_rag) * 1000.0
            iter_record["rag_total_lat"] = lat_rag_total
            iter_record["rag_embed_lat"] = 0.0
            iter_record["rag_retrieval_lat"] = 0.0
            if "429" in str(e):
                iter_record["429"] += 1
                total_429 += 1
            else:
                iter_record["api_errors"] += 1
                total_api_errors += 1
            rag_docs = []
        total_gemini_calls += 1 # 1 embedding call
        time.sleep(4.2)

        # Stage 5: Support Suggestion Generation
        formatted_context = _format_context_for_prompt(rag_docs)
        sugg_prompt_len = len(customer_context) + len(formatted_context) + 500
        sugg_input_sizes.append(sugg_prompt_len)

        t0_sugg = time.perf_counter()
        try:
            sugg_res = generate_support_suggestion(
                conversation_text=customer_context,
                retrieved_context=rag_docs,
                category=iter_record.get("category")
            )
            lat_sugg = (time.perf_counter() - t0_sugg) * 1000.0
            iter_record["sugg_lat"] = lat_sugg
            iter_record["grounded"] = sugg_res.get("is_grounded")
            sugg_output_sizes.append(len(sugg_res.get("suggestion", "")))
        except Exception as e:
            lat_sugg = (time.perf_counter() - t0_sugg) * 1000.0
            iter_record["sugg_lat"] = lat_sugg
            if "429" in str(e):
                iter_record["429"] += 1
                total_429 += 1
            else:
                iter_record["api_errors"] += 1
                total_api_errors += 1
            sugg_res = {}
        total_gemini_calls += 1
        time.sleep(4.2)

        # Actual sequential downstream total in PipelineCoordinatorWorker:
        # Sentiment + Category + RAG (Embedding + Retrieval) + Suggestion
        total_downstream = lat_sent + lat_cat + lat_rag_total + lat_sugg
        iter_record["total_downstream"] = total_downstream

        iterations.append(iter_record)

        print(f"Iteration {i} Results:")
        print(f"  Role Classifier: {lat_role:.1f} ms")
        print(f"  Sentiment:       {lat_sent:.1f} ms")
        print(f"  Category:        {lat_cat:.1f} ms")
        print(f"  Embedding:       {iter_record['rag_embed_lat']:.1f} ms")
        print(f"  Retrieval:       {iter_record['rag_retrieval_lat']:.1f} ms")
        print(f"  Suggestion:      {lat_sugg:.1f} ms")
        print(f"  Total Downstream:{total_downstream:.1f} ms")
        print(f"  429s: {iter_record['429']}, API errors: {iter_record['api_errors']}")

    # =========================================================================
    # PART 4, 5, 6: AGGREGATIONS & STATISTICS
    # =========================================================================
    downstream_totals = [it["total_downstream"] for it in iterations]
    sorted_ds = sorted(downstream_totals)

    avg_sent = sum(it["sent_lat"] for it in iterations) / len(iterations)
    avg_cat = sum(it["cat_lat"] for it in iterations) / len(iterations)
    avg_embed = sum(it["rag_embed_lat"] for it in iterations) / len(iterations)
    avg_retrieval = sum(it["rag_retrieval_lat"] for it in iterations) / len(iterations)
    avg_sugg = sum(it["sugg_lat"] for it in iterations) / len(iterations)
    avg_total = sum(downstream_totals) / len(downstream_totals)

    min_total = sorted_ds[0]
    max_total = sorted_ds[-1]
    p50_total = sorted_ds[len(sorted_ds) // 2]
    p95_total = sorted_ds[int(0.95 * len(sorted_ds))]

    under_2s_count = sum(1 for t in downstream_totals if t < 2000.0)
    under_2s_pct = (under_2s_count / len(downstream_totals)) * 100.0

    print("\n" + "=" * 60)
    print("BENCHMARK AGGREGATE SUMMARY COMPLETE")
    print("=" * 60)

    return {
        "python_version": python_version,
        "gen_model": gen_model,
        "emb_model": emb_model,
        "vdb": vdb,
        "role_min": role_min,
        "role_p50": role_p50,
        "role_avg": role_avg,
        "role_p95": role_p95,
        "role_max": role_max,
        "role_accuracy": role_accuracy,
        "role_429": role_429,
        "role_errors": role_errors,
        "avg_sent": avg_sent,
        "avg_cat": avg_cat,
        "avg_embed": avg_embed,
        "avg_retrieval": avg_retrieval,
        "avg_sugg": avg_sugg,
        "avg_total": avg_total,
        "min_total": min_total,
        "p50_total": p50_total,
        "p95_total": p95_total,
        "max_total": max_total,
        "under_2s_count": under_2s_count,
        "under_2s_pct": under_2s_pct,
        "total_gemini_calls": total_gemini_calls,
        "total_429": total_429,
        "total_api_errors": total_api_errors,
        "role_prompt_size": sum(role_prompt_sizes) / len(role_prompt_sizes),
        "sent_input_size": sum(sent_input_sizes) / len(sent_input_sizes),
        "cat_input_size": sum(cat_input_sizes) / len(cat_input_sizes),
        "rag_input_size": sum(rag_input_sizes) / len(rag_input_sizes),
        "sugg_input_size": sum(sugg_input_sizes) / len(sugg_input_sizes),
        "sugg_output_size": sum(sugg_output_sizes) / len(sugg_output_sizes),
        "iterations": iterations,
        "role_results": role_results
    }


if __name__ == "__main__":
    results = run_benchmark()

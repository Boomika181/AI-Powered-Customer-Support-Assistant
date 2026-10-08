"""
scripts/test_support_suggestions_dataset.py

Evaluation Runner for Support Suggestion Test Cases (RAG + Gemini).
Executes the 12 evaluation cases from data/test_cases/support_suggestion_test_cases.json
against the production RAG retrieval and Gemini suggestion pipeline, produces deterministic
evaluation metrics, and writes summary reports in JSON and Markdown.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_search import search_knowledge_base
from src.llm_suggestions import generate_support_suggestion

logger = logging.getLogger("dataset_evaluation")

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "test_cases" / "support_suggestion_test_cases.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "data" / "test_cases" / "support_suggestion_evaluation_results.json"
DEFAULT_MD_REPORT = PROJECT_ROOT / "data" / "test_cases" / "support_suggestion_evaluation_results.md"


def evaluate_single_case(
    case: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    llm_result: Dict[str, Any],
    latencies: Dict[str, float]
) -> Dict[str, Any]:
    """
    Deterministically evaluates a single test case result without invoking another LLM.

    Args:
        case: Test case dictionary from dataset.
        retrieved_chunks: Chunks returned by search_knowledge_base.
        llm_result: Output from generate_support_suggestion.
        latencies: Measured execution latencies (retrieval_ms, generation_ms, total_ms).

    Returns:
        Evaluated record dictionary with pass/fail status and audit fields.
    """
    case_id = case.get("id", "UNKNOWN")
    expected_grounded = bool(case.get("expected_grounded", True))
    is_cross_doc = (case.get("document") == "Both" and expected_grounded)
    expected_sources = set(case.get("expected_source_refs", []))

    actual_grounded = bool(llm_result.get("is_grounded", False))
    suggestion = (llm_result.get("suggestion") or "").strip()
    confidence = float(llm_result.get("confidence", 0.0))
    is_429 = bool(llm_result.get("is_429", False))
    api_error = llm_result.get("error")

    # Extract source references from retrieved chunks
    retrieved_refs = set()
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        doc_ref = meta.get("document_reference")
        if doc_ref:
            retrieved_refs.add(doc_ref)

    # Extract source references returned by LLM module
    returned_refs = set()
    for src in llm_result.get("sources", []):
        doc_ref = src.get("document_reference") or src.get("source")
        if doc_ref:
            returned_refs.add(doc_ref)

    all_available_sources = retrieved_refs | returned_refs

    passed = True
    failure_reasons = []

    if api_error:
        passed = False
        failure_reasons.append(f"API Error: {api_error}")

    if is_429:
        passed = False
        failure_reasons.append("API Quota Exhausted (429)")

    # 1. Grounded Cases (Single-doc and Cross-doc)
    if expected_grounded:
        if not suggestion:
            passed = False
            failure_reasons.append("Empty suggestion returned.")

        if not actual_grounded:
            passed = False
            failure_reasons.append("Expected is_grounded=True, but system returned is_grounded=False.")

        # Disallow obvious no-context disclaimers when grounded answer was expected
        disclaimer_phrases = [
            "no relevant information was found",
            "documentation does not provide sufficient information",
            "insufficient information"
        ]
        if any(dp in suggestion.lower() for dp in disclaimer_phrases):
            passed = False
            failure_reasons.append("Suggestion returned an ungrounded disclaimer for an expected grounded case.")

        # Source reference check
        if expected_sources:
            matched_sources = expected_sources.intersection(all_available_sources)
            if is_cross_doc:
                # For cross-doc cases, all expected sources (across docs) should be represented
                if len(matched_sources) < len(expected_sources):
                    passed = False
                    missing = expected_sources - matched_sources
                    failure_reasons.append(
                        f"Cross-document source missing: {sorted(list(missing))}. "
                        f"Expected {sorted(list(expected_sources))}, found {sorted(list(matched_sources))}."
                    )
            else:
                # For single-doc grounded cases, at least one expected reference must be retrieved/represented
                if len(matched_sources) == 0:
                    passed = False
                    failure_reasons.append(
                        f"None of the expected source references were retrieved/returned. "
                        f"Expected {sorted(list(expected_sources))}, retrieved {sorted(list(retrieved_refs))}."
                    )

    # 2. No-Context Cases
    else:
        if actual_grounded:
            passed = False
            failure_reasons.append("Expected is_grounded=False, but system returned is_grounded=True.")

        disclaimer_phrases = [
            "does not provide sufficient information",
            "not provide sufficient information",
            "no relevant information",
            "not documented",
            "insufficient documentation",
            "insufficient information",
            "escalate"
        ]
        if not any(dp in suggestion.lower() for dp in disclaimer_phrases):
            passed = False
            failure_reasons.append("No-context response failed to clearly state insufficient documentation.")

        if len(returned_refs) > 0 and actual_grounded:
            passed = False
            failure_reasons.append(f"No-context case returned grounded source references: {sorted(list(returned_refs))}.")

    return {
        "id": case_id,
        "machine": case.get("machine"),
        "document": case.get("document"),
        "customer_query": case.get("customer_query"),
        "expected_category": case.get("expected_category"),
        "expected_grounded": expected_grounded,
        "actual_grounded": actual_grounded,
        "expected_source_refs": sorted(list(expected_sources)),
        "retrieved_source_refs": sorted(list(retrieved_refs)),
        "returned_source_refs": sorted(list(returned_refs)),
        "suggestion": suggestion,
        "confidence": confidence,
        "retrieval_ms": round(latencies.get("retrieval_ms", 0.0), 2),
        "generation_ms": round(latencies.get("generation_ms", 0.0), 2),
        "total_ms": round(latencies.get("total_ms", 0.0), 2),
        "passed": passed,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
        "api_status": "ERROR" if (api_error or is_429) else "OK",
        "is_429": is_429
    }


def compute_summary_metrics(evaluated_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes overall benchmark and safety metrics from evaluated case records.
    """
    total = len(evaluated_cases)
    if total == 0:
        return {
            "total_cases": 0, "passed": 0, "failed": 0,
            "grounded_cases_passed": 0, "cross_doc_cases_passed": 0, "no_context_cases_passed": 0,
            "grounding_accuracy": 0.0, "source_reference_match_rate": 0.0, "no_context_safety_rate": 0.0,
            "avg_generation_latency_ms": 0.0, "avg_total_latency_ms": 0.0, "num_429_errors": 0
        }

    passed_count = sum(1 for c in evaluated_cases if c["passed"])
    failed_count = total - passed_count

    # Breakdown by category
    single_doc_grounded = [c for c in evaluated_cases if c["expected_grounded"] and c.get("document") != "Both"]
    cross_doc_grounded = [c for c in evaluated_cases if c["expected_grounded"] and c.get("document") == "Both"]
    no_context_cases = [c for c in evaluated_cases if not c["expected_grounded"]]

    grounded_passed = sum(1 for c in single_doc_grounded if c["passed"])
    cross_doc_passed = sum(1 for c in cross_doc_grounded if c["passed"])
    no_context_passed = sum(1 for c in no_context_cases if c["passed"])

    all_grounded_total = len(single_doc_grounded) + len(cross_doc_grounded)
    all_grounded_passed = grounded_passed + cross_doc_passed

    grounding_acc = (all_grounded_passed / all_grounded_total * 100.0) if all_grounded_total else 0.0

    # Source reference match rate across all grounded cases
    source_matches = 0
    for c in single_doc_grounded + cross_doc_grounded:
        exp = set(c["expected_source_refs"])
        avail = set(c["retrieved_source_refs"]) | set(c["returned_source_refs"])
        if c.get("document") == "Both":
            if exp.issubset(avail):
                source_matches += 1
        else:
            if exp.intersection(avail):
                source_matches += 1

    source_match_rate = (source_matches / all_grounded_total * 100.0) if all_grounded_total else 0.0
    no_context_safety_rate = (no_context_passed / len(no_context_cases) * 100.0) if no_context_cases else 0.0

    avg_gen_latency = sum(c["generation_ms"] for c in evaluated_cases) / total
    avg_total_latency = sum(c["total_ms"] for c in evaluated_cases) / total
    num_429 = sum(1 for c in evaluated_cases if c["is_429"])

    return {
        "total_cases": total,
        "passed": passed_count,
        "failed": failed_count,
        "grounded_cases_passed": f"{grounded_passed}/{len(single_doc_grounded)}",
        "cross_doc_cases_passed": f"{cross_doc_passed}/{len(cross_doc_grounded)}",
        "no_context_cases_passed": f"{no_context_passed}/{len(no_context_cases)}",
        "grounding_accuracy": round(grounding_acc, 2),
        "source_reference_match_rate": round(source_match_rate, 2),
        "no_context_safety_rate": round(no_context_safety_rate, 2),
        "avg_generation_latency_ms": round(avg_gen_latency, 2),
        "avg_total_latency_ms": round(avg_total_latency, 2),
        "num_429_errors": num_429
    }


def save_reports(
    evaluated_cases: List[Dict[str, Any]],
    summary: Dict[str, Any],
    json_path: Path,
    md_path: Path
) -> None:
    """
    Saves evaluation results to JSON and Markdown reports.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_model": os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"),
        "embedding_model": "gemini-embedding-2",
        "summary": summary,
        "cases": evaluated_cases
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Markdown report
    lines = [
        "# Support Suggestion Dataset Evaluation Results",
        "",
        f"- **Execution Timestamp:** {report_data['timestamp']}",
        f"- **Active LLM Generation Model:** `{report_data['generation_model']}`",
        f"- **Embedding Model:** `{report_data['embedding_model']}` (768 dimensions)",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Cases** | {summary['total_cases']} |",
        f"| **Passed** | **{summary['passed']}** |",
        f"| **Failed** | {summary['failed']} |",
        f"| **Single-Document Grounded Passed** | {summary['grounded_cases_passed']} |",
        f"| **Cross-Document Grounded Passed** | {summary['cross_doc_cases_passed']} |",
        f"| **No-Context Safety Passed** | {summary['no_context_cases_passed']} |",
        f"| **Grounding Accuracy** | {summary['grounding_accuracy']}% |",
        f"| **Source Reference Match Rate** | {summary['source_reference_match_rate']}% |",
        f"| **No-Context Safety Rate** | {summary['no_context_safety_rate']}% |",
        f"| **Average Gemini Generation Latency** | {summary['avg_generation_latency_ms']} ms |",
        f"| **Average Total Pipeline Latency** | {summary['avg_total_latency_ms']} ms |",
        f"| **429 Rate Limit Errors** | {summary['num_429_errors']} |",
        "",
        "## Detailed Case Results",
        ""
    ]

    for c in evaluated_cases:
        status_badge = "✅ PASS" if c["passed"] else "❌ FAIL"
        lines.append(f"### {c['id']}: {c['machine']} — {status_badge}")
        lines.append(f"- **Query:** *\"{c['customer_query']}\"*")
        lines.append(f"- **Expected Grounded:** `{c['expected_grounded']}` | **Actual Grounded:** `{c['actual_grounded']}`")
        lines.append(f"- **Expected Sources:** `{c['expected_source_refs']}`")
        lines.append(f"- **Retrieved Sources:** `{c['retrieved_source_refs']}`")
        lines.append(f"- **Returned Sources:** `{c['returned_source_refs']}`")
        lines.append(f"- **Confidence:** `{c['confidence']}`")
        lines.append(f"- **Latency:** Generation: `{c['generation_ms']} ms` | Total: `{c['total_ms']} ms`")
        lines.append(f"- **Generated Suggestion:**")
        lines.append(f"  > {c['suggestion']}")
        if not c["passed"]:
            lines.append(f"- **Failure Reason:** ⚠️ {c['failure_reason']}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_dataset_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    json_report_path: Path = DEFAULT_JSON_REPORT,
    md_report_path: Path = DEFAULT_MD_REPORT
) -> Dict[str, Any]:
    """
    Runs the complete dataset evaluation.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("=" * 80)
    print("=== SUPPORT SUGGESTION DATASET EVALUATION ===")
    print(f"Total Test Cases: {len(cases)}")
    active_model = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
    print(f"LLM Generation Model: {active_model}")
    print(f"Embedding Model     : gemini-embedding-2 (768d)")
    print("=" * 80)

    evaluated_cases = []

    for idx, case in enumerate(cases, start=1):
        case_id = case["id"]
        query = case["customer_query"]
        category = case.get("expected_category")

        print(f"\n[{idx}/{len(cases)}] Case {case_id} ({case.get('machine')})")
        print(f"Query: \"{query}\"")

        # 1. Semantic Retrieval
        retrieval_start = time.perf_counter()
        try:
            chunks = search_knowledge_base(query, top_k=5)
        except Exception as e:
            logger.error("RAG retrieval failed for %s: %s", case_id, e)
            chunks = []
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # 2. Suggestion Generation
        gen_start = time.perf_counter()
        llm_result = {}
        try:
            llm_result = generate_support_suggestion(
                conversation_text=query,
                retrieved_context=chunks,
                category=category
            )
        except Exception as e:
            logger.error("Suggestion generation failed for %s: %s", case_id, e)
            err_str = str(e)
            llm_result = {
                "suggestion": "",
                "confidence": 0.0,
                "sources": [],
                "is_grounded": False,
                "error": err_str,
                "is_429": "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            }
        gen_ms = (time.perf_counter() - gen_start) * 1000.0
        total_ms = retrieval_ms + gen_ms

        latencies = {
            "retrieval_ms": retrieval_ms,
            "generation_ms": gen_ms,
            "total_ms": total_ms
        }

        # 3. Deterministic Evaluation
        eval_record = evaluate_single_case(case, chunks, llm_result, latencies)
        evaluated_cases.append(eval_record)

        # Print per-case evaluation summary
        status_str = "PASS" if eval_record["passed"] else "FAIL"
        print(f"Expected Grounded: {eval_record['expected_grounded']} | Actual: {eval_record['actual_grounded']}")
        print(f"Expected Sources : {eval_record['expected_source_refs']}")
        print(f"Retrieved Sources: {eval_record['retrieved_source_refs']}")
        print(f"Returned Sources : {eval_record['returned_source_refs']}")
        print(f"Suggestion       : {eval_record['suggestion']}")
        print(f"Confidence       : {eval_record['confidence']}")
        print(f"Generation Latency: {eval_record['generation_ms']:.2f} ms | Total: {eval_record['total_ms']:.2f} ms")
        print(f"Result           : {status_str}")
        if not eval_record["passed"]:
            print(f"Failure Reason   : {eval_record['failure_reason']}")

    # 4. Summary Metrics
    summary = compute_summary_metrics(evaluated_cases)

    print("\n" + "=" * 80)
    print("=== EVALUATION SUMMARY ===")
    print(f"Total Cases                 : {summary['total_cases']}")
    print(f"Passed                      : {summary['passed']}")
    print(f"Failed                      : {summary['failed']}")
    print(f"Single-Doc Grounded Passed  : {summary['grounded_cases_passed']}")
    print(f"Cross-Document Passed       : {summary['cross_doc_cases_passed']}")
    print(f"No-Context Cases Passed     : {summary['no_context_cases_passed']}")
    print(f"Grounding Accuracy          : {summary['grounding_accuracy']}%")
    print(f"Source Reference Match Rate : {summary['source_reference_match_rate']}%")
    print(f"No-Context Safety Rate      : {summary['no_context_safety_rate']}%")
    print(f"Avg Gemini Latency          : {summary['avg_generation_latency_ms']} ms")
    print(f"Avg Total Latency           : {summary['avg_total_latency_ms']} ms")
    print(f"Quota Exhausted (429)       : {summary['num_429_errors']}")
    print("=" * 80)

    # 5. Save Reports
    save_reports(evaluated_cases, summary, json_report_path, md_report_path)
    print(f"Saved machine-readable report to: {json_report_path}")
    print(f"Saved human-readable report to   : {md_report_path}")

    return {
        "summary": summary,
        "cases": evaluated_cases
    }


if __name__ == "__main__":
    run_dataset_evaluation()

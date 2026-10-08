"""
scripts/verify_suggestion_client_reuse_live.py

Controlled live verification of persistent Gemini client reuse in src/llm_suggestions.py.
Executes exactly 2 suggestion calls with a safe inter-call pause to protect free-tier RPM quota.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_search import search_knowledge_base
from src.llm_suggestions import generate_support_suggestion, _shared_client


def main():
    print("=" * 70)
    print("CONTROLLED LIVE VERIFICATION: PERSISTENT GEMINI CLIENT REUSE")
    print("=" * 70)

    # 1. Environment & API key check
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        print("ERROR: Valid GEMINI_API_KEY is not set.")
        sys.exit(1)

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    print(f"Model: {model_name}")

    # 2. Context setup
    query_text = (
        "My WP400 is showing error E102. The extraction fan isn't running. "
        "The filter is blocked. The machine still won't operate."
    )
    print(f"\nQuery Text: {query_text}")

    print("\n[Step 1] Retrieving top-3 RAG chunks from ChromaDB...")
    rag_start = time.perf_counter()
    chunks = search_knowledge_base(query_text, top_k=3)
    rag_elapsed = (time.perf_counter() - rag_start) * 1000.0
    print(f"Retrieved {len(chunks)} chunks in {rag_elapsed:.1f} ms")

    if not chunks:
        print("ERROR: No chunks retrieved from ChromaDB.")
        sys.exit(1)

    print("\nInitial client state in llm_suggestions module:")
    import src.llm_suggestions as ls
    print(f"_shared_client is None: {ls._shared_client is None}")

    # 3. Call 1 (Cold / First shared call)
    print("\n" + "-" * 70)
    print("[Call 1] Executing first suggestion call (initializes shared client)...")
    t0 = time.perf_counter()
    res1 = generate_support_suggestion(
        conversation_text=query_text,
        retrieved_context=chunks,
        category="Technical Troubleshooting"
    )
    latency1_s = time.perf_counter() - t0
    client_after_1 = ls._shared_client

    print(f"Call 1 Completed in: {latency1_s:.3f} s ({latency1_s * 1000:.1f} ms)")
    print(f"Call 1 is_grounded: {res1.get('is_grounded')}")
    print(f"Call 1 confidence: {res1.get('confidence')}")
    print(f"Call 1 sources count: {len(res1.get('sources', []))}")
    sources1 = [s.get("source_file") or s.get("document_reference") for s in res1.get("sources", [])]
    print(f"Call 1 sources: {sources1}")
    print(f"Call 1 suggestion preview: {str(res1.get('suggestion', ''))[:100]}...")
    print(f"Shared client created: {client_after_1 is not None}")

    # Assert Call 1
    assert res1.get("is_grounded") is True, "Call 1 must be grounded"
    assert len(res1.get("sources", [])) > 0, "Call 1 must have sources"
    assert client_after_1 is not None, "Shared client must be initialized after Call 1"

    # 4. Safe pause to avoid triggering free-tier RPM limit
    pause_s = 15
    print(f"\nPausing {pause_s}s to maintain clean quota margin under free-tier 15 RPM...")
    time.sleep(pause_s)

    # 5. Call 2 (Warm / Reused persistent client)
    print("\n" + "-" * 70)
    print("[Call 2] Executing second suggestion call (reuses existing persistent client)...")
    t1 = time.perf_counter()
    res2 = generate_support_suggestion(
        conversation_text=query_text,
        retrieved_context=chunks,
        category="Technical Troubleshooting"
    )
    latency2_s = time.perf_counter() - t1
    client_after_2 = ls._shared_client

    print(f"Call 2 Completed in: {latency2_s:.3f} s ({latency2_s * 1000:.1f} ms)")
    print(f"Call 2 is_grounded: {res2.get('is_grounded')}")
    print(f"Call 2 confidence: {res2.get('confidence')}")
    print(f"Call 2 sources count: {len(res2.get('sources', []))}")
    sources2 = [s.get("source_file") or s.get("document_reference") for s in res2.get("sources", [])]
    print(f"Call 2 sources: {sources2}")
    print(f"Call 2 suggestion preview: {str(res2.get('suggestion', ''))[:100]}...")
    print(f"Shared client is identical instance: {client_after_1 is client_after_2}")

    # Assert Call 2
    assert res2.get("is_grounded") is True, "Call 2 must be grounded"
    assert len(res2.get("sources", [])) > 0, "Call 2 must have sources"
    assert client_after_1 is client_after_2, "Client must be reused identically"

    print("\n" + "=" * 70)
    print("LIVE VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Call 1 Latency (Initial connection): {latency1_s:.3f} s")
    print(f"Call 2 Latency (Reused connection):  {latency2_s:.3f} s")
    diff_s = latency1_s - latency2_s
    pct = (diff_s / latency1_s) * 100 if latency1_s > 0 else 0
    print(f"Latency delta on reused call:        -{diff_s:.3f} s ({pct:.1f}% reduction)")
    print(f"Baseline clean latency reference:    ~1.420 s")
    print("All functional assertions: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()

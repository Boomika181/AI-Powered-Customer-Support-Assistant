#!/usr/bin/env python3
"""
scripts/test_gemini_embeddings_real.py

Live Integration Test for Gemini Embedding 2 (Real API Call).
Tests generating real vector embeddings for:
1. A technical document chunk (containing technical IDs and numbers)
2. A customer technical query

Outputs:
- Configured model name
- Vector dimensionality (target: 768)
- API call latency
- Success / Failure / Quota blocked status
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.embeddings import (
    embed_text,
    embed_query,
    get_embedding_dimension,
    get_embedding_model_name,
    EmbeddingError,
    DimensionMismatchError
)

SAMPLE_DOCUMENT_CHUNK = (
    "Vantor Industrial WP-400 Panel Saw: Error code E-102 indicates dust extraction fault. "
    "Working pressure must read 6.0 bar. When differential pressure across filter cartridge "
    "WP4-FL-DE1 exceeds 1200 Pa, replace cartridge immediately. Follow Lock-Out/Tag-Out (LOTO)."
)

SAMPLE_QUERY = "The WP-400 is showing E-102 with high filter pressure. What should I replace?"


def run_real_embedding_test():
    print("=" * 70)
    print("REAL GEMINI EMBEDDING 2 API INTEGRATION TEST")
    print("=" * 70)

    model_name = get_embedding_model_name()
    expected_dim = get_embedding_dimension()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[FAIL] Neither GEMINI_API_KEY nor GOOGLE_API_KEY is configured in .env.")
        sys.exit(1)

    print(f"Target Model      : {model_name}")
    print(f"Target Dimension  : {expected_dim}")
    print(f"API Key           : Configured ({api_key[:6]}...{api_key[-4:]})")
    print("-" * 70)

    # Test 1: Embed technical document chunk
    print("\n[TEST 1] Embedding Technical Document Chunk...")
    print(f"Input Preview: \"{SAMPLE_DOCUMENT_CHUNK[:80]}...\"")

    t0 = time.perf_counter()
    try:
        doc_vector = embed_text(SAMPLE_DOCUMENT_CHUNK)
        doc_latency = (time.perf_counter() - t0) * 1000
        doc_dim = len(doc_vector)
        doc_success = (doc_dim == expected_dim)
        print(f"  Status        : {'PASS' if doc_success else 'FAIL'}")
        print(f"  Model Used    : {model_name}")
        print(f"  Dimension     : {doc_dim} (Expected: {expected_dim})")
        print(f"  Latency       : {doc_latency:.2f} ms")
        print(f"  Vector Sample : [{', '.join(f'{v:.4f}' for v in doc_vector[:5])}, ...]")
    except Exception as exc:
        doc_latency = (time.perf_counter() - t0) * 1000
        err_msg = str(exc)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            print(f"  Status        : BLOCKED (429 Quota Exhausted)")
        elif "503" in err_msg or "UNAVAILABLE" in err_msg:
            print(f"  Status        : BLOCKED (503 Service Unavailable)")
        else:
            print(f"  Status        : FAIL ({err_msg})")
        print(f"  Latency       : {doc_latency:.2f} ms")
        sys.exit(1)

    # Test 2: Embed customer technical query
    print("\n[TEST 2] Embedding Customer Technical Query...")
    print(f"Input Query: \"{SAMPLE_QUERY}\"")

    t1 = time.perf_counter()
    try:
        query_vector = embed_query(SAMPLE_QUERY)
        query_latency = (time.perf_counter() - t1) * 1000
        query_dim = len(query_vector)
        query_success = (query_dim == expected_dim)
        print(f"  Status        : {'PASS' if query_success else 'FAIL'}")
        print(f"  Model Used    : {model_name}")
        print(f"  Dimension     : {query_dim} (Expected: {expected_dim})")
        print(f"  Latency       : {query_latency:.2f} ms")
        print(f"  Vector Sample : [{', '.join(f'{v:.4f}' for v in query_vector[:5])}, ...]")
    except Exception as exc:
        query_latency = (time.perf_counter() - t1) * 1000
        err_msg = str(exc)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            print(f"  Status        : BLOCKED (429 Quota Exhausted)")
        elif "503" in err_msg or "UNAVAILABLE" in err_msg:
            print(f"  Status        : BLOCKED (503 Service Unavailable)")
        else:
            print(f"  Status        : FAIL ({err_msg})")
        print(f"  Latency       : {query_latency:.2f} ms")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("EMBEDDING VERIFICATION SUMMARY")
    print(f"  Model               : {model_name}")
    print(f"  Document Dimension  : {doc_dim} (Expected: {expected_dim})")
    print(f"  Query Dimension     : {query_dim} (Expected: {expected_dim})")
    print(f"  Document Latency    : {doc_latency:.2f} ms")
    print(f"  Query Latency       : {query_latency:.2f} ms")
    print(f"  Overall Status      : {'PASS' if (doc_success and query_success) else 'FAIL'}")
    print("=" * 70)

    if doc_success and query_success:
        print("\n>>> RESULT: SUCCESSFUL REAL GEMINI EMBEDDING 2 VERIFICATION")
        return 0
    else:
        print("\n>>> RESULT: VERIFICATION FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_real_embedding_test())

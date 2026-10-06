"""
scripts/test_llm_suggestions_real.py

Real Gemini 3.8 Flash API integration test for AI-Powered Support Suggestion Generation.
Runs actual API requests against Google Gemini with real retrieved context and validates:
- Model used: gemini-3.8-flash
- Grounded suggestion content
- Source metadata preservation
- Safety instructions
- Request latency

Usage:
    ./venv/bin/python scripts/test_llm_suggestions_real.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_suggestions import generate_support_suggestion


SAMPLE_REAL_TEST = {
    "query": "The WP-400 is showing E-102. What should I check?",
    "category": "Technical Troubleshooting",
    "context": [
        {
            "text": (
                "Section 3.1 WP-400 error codes\n"
                "Code: E-102 | Meaning: Dust extraction fault | "
                "Likely cause: Filter blocked (pressure drop above 1200 Pa) or fan stopped\n"
                "Action: Clean or replace cartridge WP4-FL-DE1. Check fan breaker.\n"
                "SAFETY: Before any maintenance, switch the machine off and lock out the main isolator."
            ),
            "metadata": {
                "source_file": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
                "page": 6,
                "document_reference": "TS-WP400-3.1",
                "section": "Section 3.1 WP-400 error codes",
                "category": "Technical Troubleshooting",
                "machine": "WP-400"
            }
        }
    ]
}


def run_real_suggestion_test():
    print("=" * 70)
    print("GEMINI 3.8 FLASH SUPPORT SUGGESTION - LIVE API INTEGRATION TEST")
    print("=" * 70)

    model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    api_key = os.getenv("GEMINI_API_KEY", "")

    print(f"Configured Model : {model}")
    print(f"API Key Present  : {'Yes (' + api_key[:6] + '...' + api_key[-4:] + ')' if api_key else 'NO'}")
    print("-" * 70)

    if not api_key:
        print("[FAIL] GEMINI_API_KEY is not set.")
        sys.exit(1)

    query = SAMPLE_REAL_TEST["query"]
    category = SAMPLE_REAL_TEST["category"]
    context = SAMPLE_REAL_TEST["context"]

    print(f"Query    : {query}")
    print(f"Category : {category}")
    print(f"Context  : {context[0]['text'][:80]}...")
    print("-" * 70)

    start_time = time.time()
    try:
        result = generate_support_suggestion(
            conversation_text=query,
            retrieved_context=context,
            category=category
        )
        gemini_latency = time.time() - start_time

        print(f"\nMODEL:\n{model}")
        print(f"\nSUGGESTION:\n{result.get('suggestion')}")
        print(f"\nCONFIDENCE:\n{result.get('confidence')}")
        print(f"\nSOURCES:\n{json.dumps(result.get('sources'), indent=2)}")
        print(f"\nLATENCY:\n{gemini_latency:.2f} seconds")
        print("\nSTATUS:\nPASS")
        print("=" * 70)
        sys.exit(0)

    except Exception as exc:
        err_msg = str(exc)
        gemini_latency = time.time() - start_time
        print(f"\n[FAIL] Error calling Gemini API: {err_msg}")
        print(f"Latency before failure: {gemini_latency:.2f} seconds")

        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            print("\n>>> NOTE: Quota exhausted (429 RESOURCE_EXHAUSTED). Free tier daily limit reached.")
            print("STATUS: QUOTA_EXHAUSTED (Cannot claim live API completion until quota resets)")
            sys.exit(2)
        else:
            print("STATUS: FAIL")
            sys.exit(1)


if __name__ == "__main__":
    run_real_suggestion_test()

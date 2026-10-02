"""
src/llm_suggestions.py

Grounded Support Suggestion Generation using Google Gemini Flash.
Scheduled for Phase 1 implementation.

Phase 1 Responsibilities:
- Formulate grounded prompt using retrieved documentation chunks.
- Generate 2–3 actionable, step-by-step suggestions.
- Ensure strict grounding: reference document codes and page numbers.
- Provide explicit fallback: "Not found in the provided manual" if documentation does not cover the query.
"""

from typing import List, Dict, Any, Optional


def generate_suggestions(
    query_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    TODO (Phase 1): Generate grounded suggestion cards via Gemini Flash.
    
    Expected return structure:
    {
        "category": "Machine Operation Issues" | "Maintenance & Parts" | "Technical Troubleshooting",
        "suggestions": [
            {
                "title": str,
                "steps": List[str],
                "doc_ref": str,
                "page": int
            }
        ],
        "is_grounded": bool
    }
    """
    raise NotImplementedError("Phase 1 Feature: Gemini grounded suggestions generation.")

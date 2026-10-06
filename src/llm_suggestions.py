"""
src/llm_suggestions.py

AI-Powered Customer Support Suggestion Generation using Google Gemini 3.8 Flash.
Synthesizes concise, actionable, strictly grounded customer-support suggestions based
on retrieved knowledge-base chunks from ChromaDB.

Key Principles:
- Strictly grounded in retrieved context: will NOT invent error codes, part numbers,
  pressures, or repair procedures.
- Strictly uses Gemini 3.8 Flash via the google-genai SDK. No fallback models.
- If retrieved_context is empty, safely returns an ungrounded disclaimer without
  calling Gemini, preventing hallucinations and conserving quota.
- Preserves industrial safety instructions (lock-out/tag-out, interlocks, eye/ear protection).
- Formats provenance metadata (sources, page numbers, document references).
- Employs direct text generation with robust JSON parsing (avoids SDK structured-output auth conflicts).
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from google import genai

logger = logging.getLogger("llm_suggestions")


class SuggestionGenerationError(ValueError):
    """Raised when Gemini suggestion generation fails or returns malformed output."""
    pass


def _strip_markdown_fences(text: str) -> str:
    """
    Safely strips markdown code block fences (```json ... ``` or ``` ... ```)
    from the raw response text before JSON parsing.
    """
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _format_context_for_prompt(retrieved_context: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved ChromaDB chunks into structured provenance blocks for Gemini.
    """
    formatted_blocks = []
    for idx, item in enumerate(retrieved_context, 1):
        meta = item.get("metadata", {})
        doc_text = item.get("text") or item.get("document", "")

        source_file = meta.get("source_file") or meta.get("source", "Unknown Document")
        page = meta.get("page", -1)
        page_str = str(page) if page != -1 else "N/A"
        category = meta.get("category", "General")
        machine = meta.get("machine", "General")
        doc_ref = meta.get("document_reference", "N/A")
        section = meta.get("section", "N/A")

        block = (
            f"SOURCE {idx}\n"
            f"--------\n"
            f"Document: {source_file}\n"
            f"Page: {page_str}\n"
            f"Category: {category}\n"
            f"Machine: {machine}\n"
            f"Reference: {doc_ref}\n"
            f"Section: {section}\n"
            f"\nContent:\n{doc_text.strip()}\n"
        )
        formatted_blocks.append(block)

    return "\n".join(formatted_blocks)


def _extract_sources(retrieved_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts deduplicated source metadata from retrieved chunks.
    """
    sources = []
    seen = set()
    for item in retrieved_context:
        meta = item.get("metadata", {})
        source_file = meta.get("source_file") or meta.get("source", "")
        doc_ref = meta.get("document_reference", "")
        page = meta.get("page", -1)

        key = (source_file, doc_ref, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_file": source_file,
                "document_reference": doc_ref,
                "page": page,
                "section": meta.get("section", ""),
                "category": meta.get("category", ""),
                "machine": meta.get("machine", "")
            })
    return sources


def generate_support_suggestion(
    conversation_text: str,
    retrieved_context: List[Dict[str, Any]],
    category: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a concise, strictly grounded customer-support suggestion using Gemini 3.8 Flash.

    Args:
        conversation_text: Customer conversation or query text.
        retrieved_context: List of retrieved chunks from ChromaDB (search_knowledge_base).
        category: Optional detected query category.
        api_key: Optional Google Gemini API key. Defaults to GEMINI_API_KEY from environment.

    Returns:
        Dict[str, Any] with:
            - suggestion: str (clear, actionable, grounded support suggestion)
            - confidence: float between 0.0 and 1.0
            - sources: List[Dict] with provenance metadata
            - is_grounded: bool
            - category: str (optional)

    Raises:
        ValueError: If conversation_text is empty or API key is missing.
        SuggestionGenerationError: If the Gemini API call fails or produces invalid output.
    """
    # 1. Validate conversation text
    if not conversation_text or not conversation_text.strip():
        raise ValueError("conversation_text cannot be empty")

    # 2. Handle empty retrieved context safely without calling Gemini
    if not retrieved_context:
        logger.info("Empty retrieved context provided; returning fallback disclaimer without calling LLM.")
        return {
            "suggestion": (
                "No relevant information was found in the knowledge base. "
                "Please refer to the machine documentation or escalate the issue to a support engineer."
            ),
            "confidence": 0.0,
            "sources": [],
            "is_grounded": False,
            "category": category
        }

    # 3. Resolve API key
    if api_key is not None:
        resolved_key = api_key.strip()
    else:
        resolved_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )

    if not resolved_key or resolved_key.startswith("your_"):
        raise ValueError("GEMINI_API_KEY is required but not found in the environment or parameters.")

    # 4. Resolve configured model (enforces gemini-3.8-flash)
    model_name = os.getenv("LLM_MODEL", "gemini-3.8-flash")

    # 5. Format context and prompt
    context_str = _format_context_for_prompt(retrieved_context)
    category_instruction = f"The query is classified under: {category}." if category else ""

    prompt = f"""You are an expert AI customer support assistant for industrial machinery.
{category_instruction}

Your task is to generate a concise, actionable, and strictly grounded customer support suggestion to address the customer's issue.

CRITICAL GROUNDING AND SAFETY RULES:
1. Answer using ONLY the supplied knowledge-base context below.
2. DO NOT invent or fabricate any technical procedures, error codes, part numbers, pressures, speeds, temperatures, or maintenance intervals.
3. If the knowledge-base context does NOT contain enough information to address the query, state clearly that the available documentation does not provide sufficient information.
4. Prioritize safety: For any maintenance, blade replacement, consumable change, or mechanical troubleshooting, remind the user of required safety procedures (e.g. switch off, lock out the isolator, wait for parts to stop) as documented. Never suggest bypassing safety interlocks.
5. Provide a helpful, direct response explaining the likely cause and step-by-step resolution steps.

KNOWLEDGE BASE CONTEXT:
=======================
{context_str}
=======================

CUSTOMER QUERY:
\"\"\"
{conversation_text.strip()}
\"\"\"

Return ONLY a valid JSON object with no additional text or commentary using this exact schema:
{{
  "suggestion": "Clear, grounded step-by-step suggestion with necessary safety precautions",
  "confidence": 0.95,
  "is_grounded": true
}}
"""

    # 6. Execute Gemini call with retries for transient errors
    client = genai.Client(api_key=resolved_key)
    max_retries = 3
    last_error = None
    response = None

    for attempt in range(max_retries):
        try:
            logger.info("Calling Gemini (%s) for suggestion generation (attempt %d/%d)", model_name, attempt + 1, max_retries)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            break
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                logger.error("Gemini API quota exhausted: %s", exc)
                raise SuggestionGenerationError(f"Gemini API request failed: {exc}") from exc
            if "503" in err_str or "UNAVAILABLE" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = 1
                    logger.warning("Transient error (%s), retrying in %ds...", err_str[:60], sleep_time)
                    time.sleep(sleep_time)
                    continue
            logger.error("Gemini API call failed: %s", exc)
            raise SuggestionGenerationError(f"Gemini API request failed: {exc}") from exc

    if response is None:
        raise SuggestionGenerationError(f"Gemini API request failed after {max_retries} attempts: {last_error}")

    # 7. Parse response text
    raw_text = getattr(response, "text", None)
    if not raw_text or not raw_text.strip():
        raise SuggestionGenerationError("Gemini suggestion response was empty.")

    cleaned_json_text = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned_json_text)
    except json.JSONDecodeError as exc:
        raise SuggestionGenerationError(f"Gemini suggestion response was invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SuggestionGenerationError("Gemini suggestion response must be a JSON object.")

    suggestion = data.get("suggestion")
    if not isinstance(suggestion, str) or not suggestion.strip():
        raise SuggestionGenerationError("Invalid or empty suggestion returned by Gemini.")

    confidence = data.get("confidence", 0.9)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence_val = 0.85
    else:
        confidence_val = max(0.0, min(1.0, float(confidence)))

    is_grounded = bool(data.get("is_grounded", True))
    sources = _extract_sources(retrieved_context)

    return {
        "suggestion": suggestion.strip(),
        "confidence": round(confidence_val, 4),
        "sources": sources,
        "is_grounded": is_grounded,
        "category": category
    }


def generate_suggestions(
    query_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for Phase 1 suggestions generation.
    """
    return generate_support_suggestion(
        conversation_text=query_text,
        retrieved_context=retrieved_chunks,
        api_key=api_key
    )

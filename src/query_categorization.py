"""
src/query_categorization.py

Customer Query Categorization Module using Google Gemini 3.8 Flash.
Classifies support conversation queries into one of the three core SOW categories:
1. Machine Operation Issues
2. Maintenance & Parts
3. Technical Troubleshooting

Key Principles:
- Uses direct text generation call via google-genai SDK (no response_schema or response_mime_type).
- Output format enforced via prompt instructions and safely parsed/validated in Python.
- Strips markdown fences if returned by the model.
- Strictly uses conversation context rather than rigid keyword matching.
- Zero audio or conversation text written to disk.
- Never logs or exposes raw API keys.
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Ensure project root is loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from google import genai

logger = logging.getLogger("query_categorization")

VALID_CATEGORIES = {
    "Machine Operation Issues",
    "Maintenance & Parts",
    "Technical Troubleshooting",
}


class QueryCategorizationError(ValueError):
    """Raised when Gemini returns an invalid or malformed query categorization response."""
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


def classify_query(
    conversation_text: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classifies a customer support query into one of three standard categories using Gemini 3.8 Flash.

    Args:
        conversation_text: The transcript or dialogue history to evaluate.
        api_key: Optional Google Gemini API key. Defaults to GEMINI_API_KEY from environment.

    Returns:
        Dict[str, Any] with:
            - category: "Machine Operation Issues" | "Maintenance & Parts" | "Technical Troubleshooting"
            - confidence: float between 0.0 and 1.0
            - rationale: str explaining the classification based strictly on context

    Raises:
        ValueError: If conversation_text is empty or if no API key is available.
        QueryCategorizationError: If the Gemini response is invalid or malformed.
    """
    # 1. Validate input text
    if not conversation_text or not conversation_text.strip():
        raise ValueError("conversation_text cannot be empty")

    # 2. Resolve API key
    if api_key is not None:
        resolved_key = api_key.strip()
    else:
        resolved_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )

    if not resolved_key or resolved_key.startswith("your_"):
        raise ValueError("GEMINI_API_KEY is required but not found in the environment or parameters.")

    # 3. Resolve configured model (defaults to gemini-3.8-flash)
    model_name = os.getenv("LLM_MODEL", "gemini-3.8-flash")

    # 4. Construct prompt requesting ONLY raw JSON
    prompt = f"""You are an expert customer support query categorizer for industrial machinery support calls.

Your task is to analyze the customer's issue from the conversation below and categorize the query into EXACTLY ONE of the following three categories:

1. Machine Operation Issues:
Questions or problems related to operating, using, configuring, or understanding normal machine operation (e.g., startup procedures, operating modes, selecting feed rates/spindle speeds, operating controls).

2. Maintenance & Parts:
Issues involving preventive maintenance schedules, servicing routines, replacement parts, consumable wear, ordering spare parts, lubrication, or physical component replacement.

3. Technical Troubleshooting:
Technical faults, error codes, alarms, diagnostic problems, component failure, machine stalls, overheating, sensor malfunctions, or abnormal behavior requiring troubleshooting procedures.

Conversation:
\"\"\"
{conversation_text.strip()}
\"\"\"

Return ONLY a valid JSON object with no additional text, markdown, or commentary. Use this exact structure:
{{
  "category": "Machine Operation Issues|Maintenance & Parts|Technical Troubleshooting",
  "confidence": 0.0,
  "rationale": "short explanation explaining why this query belongs to the chosen category"
}}
"""

    # 5. Execute normal text generation call with retries on transient errors
    client = genai.Client(api_key=resolved_key)
    max_retries = 3
    last_error = None
    response = None

    for attempt in range(max_retries):
        try:
            logger.info("Calling Gemini (%s) for query categorization (attempt %d/%d)", model_name, attempt + 1, max_retries)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            break
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.warning("Transient error (%s), retrying in %ds...", err_str[:60], sleep_time)
                    time.sleep(sleep_time)
                    continue
            logger.error("Gemini API call failed: %s", exc)
            raise QueryCategorizationError(f"Gemini API request failed: {exc}") from exc

    if response is None:
        raise QueryCategorizationError(f"Gemini API request failed after {max_retries} attempts: {last_error}")

    # 6. Parse and validate response text
    raw_text = getattr(response, "text", None)
    if not raw_text or not raw_text.strip():
        raise QueryCategorizationError("Gemini query categorization response was empty.")

    cleaned_json_text = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned_json_text)
    except json.JSONDecodeError as exc:
        raise QueryCategorizationError(f"Gemini query categorization response was invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise QueryCategorizationError("Gemini query categorization response must be a JSON object.")

    category = data.get("category")
    if category not in VALID_CATEGORIES:
        raise QueryCategorizationError(
            f"Invalid category '{category}' returned by Gemini. Must be one of {sorted(VALID_CATEGORIES)}."
        )

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise QueryCategorizationError(
            f"Invalid confidence '{confidence}' returned by Gemini. Must be a numeric float."
        )

    confidence_val = float(confidence)
    if confidence_val < 0.0 or confidence_val > 1.0:
        raise QueryCategorizationError(
            f"Confidence '{confidence_val}' out of bounds. Must be between 0.0 and 1.0."
        )

    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise QueryCategorizationError("Invalid rationale returned by Gemini. Must be a non-empty string.")

    return {
        "category": category,
        "confidence": round(confidence_val, 4),
        "rationale": rationale.strip()
    }

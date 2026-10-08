"""
src/customer_context_analyzer.py

Consolidated Customer Context Analysis using Google Gemini.
Combines Sentiment Analysis and Query Categorization into a single Gemini generation call,
returning structured JSON with both emotional sentiment and query category.

Reduces Gemini generation calls on CUSTOMER turns from 4 to 3:
1. Speaker Role Classification (gemini-3.5-flash-lite)
2. Combined Customer Context Analysis (gemini-3.5-flash-lite)
3. Support Suggestion Generation (gemini-3.5-flash-lite)
+ 1 Gemini Embedding 2 call for RAG retrieval
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from google import genai

logger = logging.getLogger("customer_context_analyzer")

VALID_SENTIMENTS: Set[str] = {"Positive", "Neutral", "Negative", "Agitated"}
VALID_CATEGORIES: Set[str] = {
    "Machine Operation Issues",
    "Maintenance & Parts",
    "Technical Troubleshooting",
}


class CustomerContextAnalysisError(ValueError):
    """Raised when combined customer context analysis fails or returns invalid output."""
    pass


def _strip_markdown_fences(text: str) -> str:
    """Safely extracts JSON content from markdown code fences if present."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def analyze_customer_context(
    conversation_text: str,
    api_key: Optional[str] = None,
    client: Optional[genai.Client] = None
) -> Dict[str, Any]:
    """
    Analyzes customer tone/sentiment AND query category in ONE Gemini generation call.

    Args:
        conversation_text: The bounded customer conversation transcript to evaluate.
        api_key: Optional Google Gemini API key. Defaults to GEMINI_API_KEY from environment.
        client: Optional persistent genai.Client instance for reuse.

    Returns:
        Dict[str, Any] with:
            - sentiment: "Positive" | "Neutral" | "Negative" | "Agitated"
            - sentiment_confidence: float between 0.0 and 1.0
            - sentiment_rationale: str
            - category: "Machine Operation Issues" | "Maintenance & Parts" | "Technical Troubleshooting"
            - category_confidence: float between 0.0 and 1.0
            - category_rationale: str

    Raises:
        ValueError: If conversation_text is empty or if no API key is available.
        CustomerContextAnalysisError: If the Gemini response is invalid or malformed.
    """
    if not conversation_text or not conversation_text.strip():
        raise ValueError("conversation_text cannot be empty")

    if client is None:
        if api_key is not None:
            resolved_key = api_key.strip()
        else:
            resolved_key = (
                os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
            )

        if not resolved_key or resolved_key.startswith("your_"):
            raise ValueError("GEMINI_API_KEY is required but not found in the environment or parameters.")

        client = genai.Client(api_key=resolved_key)

    model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    prompt = f"""You are an expert customer support tone, sentiment, and query intent analyzer for industrial machinery support calls.

Your task is to analyze the CUSTOMER'S dialogue from the conversation below and determine BOTH their emotional sentiment and their query category.
Do NOT classify the support agent's tone. Focus strictly on the customer.
Use the entire supplied conversation context.

1. SENTIMENT CLASSIFICATION:
Choose exactly one sentiment label from:
- Positive: Customer expresses satisfaction, appreciation, relief, approval, or clearly positive language.
- Neutral: Customer is factual, informational, calm, or emotionally unclear.
- Negative: Customer expresses dissatisfaction, frustration, disappointment, or a problem without strong hostility.
- Agitated: Customer shows strong anger, escalation, hostility, repeated frustration, urgency, or highly emotionally charged language.

2. QUERY CATEGORIZATION:
Categorize the query into EXACTLY ONE of the following three categories:
- Machine Operation Issues: Questions or problems related to operating, using, configuring, or understanding normal machine operation (e.g., startup procedures, operating modes, selecting feed rates/spindle speeds, operating controls).
- Maintenance & Parts: Issues involving preventive maintenance schedules, servicing routines, replacement parts, consumable wear, ordering spare parts, lubrication, or physical component replacement.
- Technical Troubleshooting: Technical faults, error codes, alarms, diagnostic problems, component failure, machine stalls, overheating, sensor malfunctions, or abnormal behavior requiring troubleshooting procedures.

Conversation:
\"\"\"
{conversation_text.strip()}
\"\"\"

Return ONLY a valid JSON object with no additional text, markdown, or commentary. Use this exact structure:
{{
  "sentiment": "Positive|Neutral|Negative|Agitated",
  "sentiment_confidence": 0.0,
  "sentiment_rationale": "short explanation based only on customer statements",
  "category": "Machine Operation Issues|Maintenance & Parts|Technical Troubleshooting",
  "category_confidence": 0.0,
  "category_rationale": "short explanation explaining why this query belongs to the chosen category"
}}
"""

    max_retries = 3
    last_error = None
    response = None

    for attempt in range(max_retries):
        try:
            logger.info("Calling Gemini (%s) for combined context analysis (attempt %d/%d)", model_name, attempt + 1, max_retries)
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
                    import time
                    time.sleep(sleep_time)
                    continue
            logger.error("Gemini API call failed: %s", exc)
            raise CustomerContextAnalysisError(f"Gemini API request failed: {exc}") from exc

    if response is None:
        raise CustomerContextAnalysisError(f"Gemini API request failed after {max_retries} attempts: {last_error}")

    raw_text = getattr(response, "text", None)
    if not raw_text or not raw_text.strip():
        raise CustomerContextAnalysisError("Gemini combined context analysis response was empty.")

    cleaned_json_text = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned_json_text)
    except json.JSONDecodeError as exc:
        raise CustomerContextAnalysisError(f"Gemini response was invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise CustomerContextAnalysisError("Gemini response must be a JSON object.")

    # 1. Validate Sentiment
    sentiment = data.get("sentiment")
    if sentiment not in VALID_SENTIMENTS:
        raise CustomerContextAnalysisError(
            f"Invalid sentiment '{sentiment}' returned by Gemini. Must be one of {sorted(VALID_SENTIMENTS)}."
        )

    sent_conf = data.get("sentiment_confidence", data.get("confidence"))
    if not isinstance(sent_conf, (int, float)) or isinstance(sent_conf, bool):
        raise CustomerContextAnalysisError(
            f"Invalid sentiment_confidence '{sent_conf}' returned by Gemini. Must be a numeric float."
        )
    sent_conf_val = float(sent_conf)
    if sent_conf_val < 0.0 or sent_conf_val > 1.0:
        raise CustomerContextAnalysisError(
            f"Sentiment confidence '{sent_conf_val}' out of bounds. Must be between 0.0 and 1.0."
        )

    sent_rationale = data.get("sentiment_rationale", data.get("rationale"))
    if not isinstance(sent_rationale, str) or not sent_rationale.strip():
        raise CustomerContextAnalysisError("Invalid sentiment_rationale returned by Gemini. Must be a non-empty string.")

    # 2. Validate Category
    category = data.get("category")
    if category not in VALID_CATEGORIES:
        raise CustomerContextAnalysisError(
            f"Invalid category '{category}' returned by Gemini. Must be one of {sorted(VALID_CATEGORIES)}."
        )

    cat_conf = data.get("category_confidence", data.get("confidence"))
    if not isinstance(cat_conf, (int, float)) or isinstance(cat_conf, bool):
        raise CustomerContextAnalysisError(
            f"Invalid category_confidence '{cat_conf}' returned by Gemini. Must be a numeric float."
        )
    cat_conf_val = float(cat_conf)
    if cat_conf_val < 0.0 or cat_conf_val > 1.0:
        raise CustomerContextAnalysisError(
            f"Category confidence '{cat_conf_val}' out of bounds. Must be between 0.0 and 1.0."
        )

    cat_rationale = data.get("category_rationale", data.get("rationale"))
    if not isinstance(cat_rationale, str) or not cat_rationale.strip():
        raise CustomerContextAnalysisError("Invalid category_rationale returned by Gemini. Must be a non-empty string.")

    return {
        "sentiment": sentiment,
        "sentiment_confidence": round(sent_conf_val, 4),
        "sentiment_rationale": sent_rationale.strip(),
        "category": category,
        "category_confidence": round(cat_conf_val, 4),
        "category_rationale": cat_rationale.strip()
    }

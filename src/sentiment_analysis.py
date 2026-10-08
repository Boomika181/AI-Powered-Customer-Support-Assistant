"""
src/sentiment_analysis.py

Customer Sentiment Analysis Module using Google Gemini.
Analyzes customer tone and sentiment from support dialogue context into four discrete categories:
- Positive
- Neutral
- Negative
- Agitated

Key Principles:
- Uses direct text generation call via google-genai SDK (no response_schema or response_mime_type).
- Output format enforced via prompt instructions and safely parsed/validated in Python.
- Strips markdown fences if returned by the model.
- Strictly classifies the CUSTOMER'S tone, ignoring the support agent's tone.
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
from google.genai import types

logger = logging.getLogger("sentiment_analysis")

VALID_SENTIMENTS = {"Positive", "Neutral", "Negative", "Agitated"}


class SentimentAnalysisError(ValueError):
    """Raised when Gemini returns an invalid or malformed sentiment response."""
    pass


def _strip_markdown_fences(text: str) -> str:
    """
    Safely strips markdown code block fences (```json ... ``` or ``` ... ```)
    from the raw response text before JSON parsing.
    """
    cleaned = text.strip()
    # Check for markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    
    # Fallback line-by-line fence stripping
    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def analyze_sentiment(
    conversation_text: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes customer sentiment using Gemini via normal text generation.

    Args:
        conversation_text: The transcript or dialogue history to evaluate.
        api_key: Optional Google Gemini API key. Defaults to GEMINI_API_KEY from environment.

    Returns:
        Dict[str, Any] with:
            - sentiment: "Positive" | "Neutral" | "Negative" | "Agitated"
            - confidence: float between 0.0 and 1.0
            - rationale: str explaining the classification based strictly on context

    Raises:
        ValueError: If conversation_text is empty or if no API key is available.
        SentimentAnalysisError: If the Gemini response is invalid or malformed.
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

    # 3. Resolve configured model (defaults to gemini-3.5-flash-lite)
    model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    # 4. Construct prompt requesting ONLY raw JSON
    prompt = f"""You are an expert customer support tone and sentiment analyzer for industrial machinery support calls.

Your task is to analyze the CUSTOMER'S tone and sentiment from the conversation below.
Do NOT classify the support agent's tone. Focus strictly on the customer.
Use the entire supplied conversation context.

Choose exactly one sentiment label from:
- Positive: Customer expresses satisfaction, appreciation, relief, approval, or clearly positive language.
- Neutral: Customer is factual, informational, calm, or emotionally unclear.
- Negative: Customer expresses dissatisfaction, frustration, disappointment, or a problem without strong hostility.
- Agitated: Customer shows strong anger, escalation, hostility, repeated frustration, urgency, or highly emotionally charged language.

Conversation:
\"\"\"
{conversation_text.strip()}
\"\"\"

Return ONLY a valid JSON object with no additional text, markdown, or commentary. Use this exact structure:
{{
  "sentiment": "Positive|Neutral|Negative|Agitated",
  "confidence": 0.0,
  "rationale": "short explanation based only on the customer statements"
}}
"""

    # 5. Execute normal text generation call with retries on transient errors
    client = genai.Client(api_key=resolved_key)
    max_retries = 3
    last_error = None
    response = None

    for attempt in range(max_retries):
        try:
            logger.info("Calling Gemini (%s) for sentiment analysis (attempt %d/%d)", model_name, attempt + 1, max_retries)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            break
        except Exception as exc:
            last_error = exc
            err_str = str(exc)
            # If transient 503 high demand or rate limit, wait and retry
            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.warning("Transient error (%s), retrying in %ds...", err_str[:60], sleep_time)
                    time.sleep(sleep_time)
                    continue
            logger.error("Gemini API call failed: %s", exc)
            raise SentimentAnalysisError(f"Gemini API request failed: {exc}") from exc

    if response is None:
        raise SentimentAnalysisError(f"Gemini API request failed after {max_retries} attempts: {last_error}")

    # 6. Parse and validate response text
    raw_text = getattr(response, "text", None)
    if not raw_text or not raw_text.strip():
        raise SentimentAnalysisError("Gemini sentiment response was empty.")

    cleaned_json_text = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned_json_text)
    except json.JSONDecodeError as exc:
        raise SentimentAnalysisError(f"Gemini sentiment response was invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SentimentAnalysisError("Gemini sentiment response must be a JSON object.")

    sentiment = data.get("sentiment")
    if sentiment not in VALID_SENTIMENTS:
        raise SentimentAnalysisError(
            f"Invalid sentiment '{sentiment}' returned by Gemini. Must be one of {sorted(VALID_SENTIMENTS)}."
        )

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise SentimentAnalysisError(
            f"Invalid confidence '{confidence}' returned by Gemini. Must be a numeric float."
        )

    confidence_val = float(confidence)
    if confidence_val < 0.0 or confidence_val > 1.0:
        raise SentimentAnalysisError(
            f"Confidence '{confidence_val}' out of bounds. Must be between 0.0 and 1.0."
        )

    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SentimentAnalysisError("Invalid rationale returned by Gemini. Must be a non-empty string.")

    return {
        "sentiment": sentiment,
        "confidence": round(confidence_val, 4),
        "rationale": rationale.strip()
    }

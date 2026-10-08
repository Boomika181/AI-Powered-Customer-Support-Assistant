"""
src/speaker_role_classifier.py

Semantic Speaker-Role Classifier for AI-Powered Customer Support Assistant.
Uses Google Gemini 3.5 Flash Lite to infer whether a finalized turn was spoken
by CUSTOMER, AGENT, or UNKNOWN based on conversational discourse and dialogue context.

Key Architectural Guarantees:
- Output Contract:
    {
      "role": "CUSTOMER" | "AGENT" | "UNKNOWN",
      "confidence": 0.0 - 1.0,
      "raw_role": str,
      "latency_ms": float,
      "rationale": str
    }
- Conservative Decision Gate:
    confidence >= 0.80 -> accept CUSTOMER or AGENT
    confidence < 0.80  -> fallback to UNKNOWN
- Speaker Continuity:
    Uses raw AssemblyAI A/B labels as continuity evidence without hardcoding.
- Safe Error Handling:
    Never silently converts errors or uncertainty into CUSTOMER.
- Latency Measurement:
    Explicitly tracks model response latency in milliseconds.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Ensure environment variables are loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from google import genai

logger = logging.getLogger("speaker_role_classifier")

# Valid discrete role classifications
VALID_ROLES = {"CUSTOMER", "AGENT", "UNKNOWN"}

# Configurable decision threshold (defaults to 0.80)
DEFAULT_CONFIDENCE_THRESHOLD = 0.80

# Module-level persistent Gemini client cache
_shared_client: Optional[genai.Client] = None
_shared_client_key: Optional[str] = None


def get_shared_gemini_client(api_key: Optional[str] = None) -> Optional[genai.Client]:
    """
    Returns a shared, persistent genai.Client instance for speaker-role classification.
    Reuses the client across calls and sessions unless the API key changes.
    """
    global _shared_client, _shared_client_key
    resolved_key = (
        (api_key or "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    if not resolved_key or resolved_key.startswith("your_"):
        return None
    if _shared_client is None or _shared_client_key != resolved_key:
        _shared_client = genai.Client(api_key=resolved_key)
        _shared_client_key = resolved_key
    return _shared_client


def get_confidence_threshold() -> float:
    """Returns the configured confidence threshold from environment or default."""
    try:
        val = float(os.getenv("SPEAKER_ROLE_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)))
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return DEFAULT_CONFIDENCE_THRESHOLD


class SpeakerContinuityTracker:
    """
    Maintains an ephemeral sliding window of confirmed role evidence for raw
    acoustic speaker clusters (e.g. 'A', 'B') without permanently locking them in.
    Supports dynamic re-clustering if speaker roles evolve or if AssemblyAI drifts.
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._history: Dict[str, List[Tuple[str, float]]] = {}

    def record_turn(self, raw_speaker: Optional[str], role: str, confidence: float) -> None:
        """Records a confirmed role assignment for a raw acoustic speaker label."""
        if not raw_speaker:
            return
        clean_raw = str(raw_speaker).strip().upper()
        clean_role = str(role).strip().upper()
        if clean_role not in {"CUSTOMER", "AGENT"}:
            return

        if clean_raw not in self._history:
            self._history[clean_raw] = []
        self._history[clean_raw].append((clean_role, float(confidence)))
        if len(self._history[clean_raw]) > self.window_size:
            self._history[clean_raw] = self._history[clean_raw][-self.window_size:]

    def get_continuity_summary(self) -> Dict[str, str]:
        """
        Summarizes acoustic continuity evidence for the classifier prompt.
        Example: {'A': 'Strong evidence of CUSTOMER', 'B': 'Tentative evidence of AGENT'}
        """
        summary: Dict[str, str] = {}
        for spk, records in self._history.items():
            if not records:
                continue
            cust_weight = sum(c for r, c in records if r == "CUSTOMER")
            agent_weight = sum(c for r, c in records if r == "AGENT")
            if cust_weight > agent_weight:
                strength = "Strong" if cust_weight >= 1.6 else "Tentative"
                summary[spk] = f"{strength} evidence of CUSTOMER"
            elif agent_weight > cust_weight:
                strength = "Strong" if agent_weight >= 1.6 else "Tentative"
                summary[spk] = f"{strength} evidence of AGENT"
            elif cust_weight > 0 or agent_weight > 0:
                summary[spk] = "Mixed dialogue evidence"
        return summary

    def reset(self) -> None:
        """Clears all continuity history when a new support session starts."""
        self._history.clear()


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


def build_role_classifier_prompt(
    current_turn_text: str,
    current_raw_speaker: Optional[str] = None,
    recent_turns: Optional[List[Dict[str, Any]]] = None,
    continuity_summary: Optional[Dict[str, str]] = None,
) -> str:
    """Constructs the prompt for Gemini 3.5 Flash Lite."""
    recent_turns = recent_turns or []
    continuity_summary = continuity_summary or {}

    history_lines: List[str] = []
    if recent_turns:
        for i, t in enumerate(recent_turns[-4:], start=1):
            t_text = t.get("text", "").strip()
            t_raw = t.get("raw_speaker") or t.get("speaker_label") or "Unknown"
            t_role = t.get("role") or t.get("speaker") or "Unknown"
            history_lines.append(f"Turn {i} [Acoustic Speaker {t_raw} | Assigned Role: {t_role}]: \"{t_text}\"")
    history_str = "\n".join(history_lines) if history_lines else "None (this is the opening turn of the call)."

    continuity_lines: List[str] = []
    if continuity_summary:
        for spk, desc in continuity_summary.items():
            continuity_lines.append(f"- Speaker {spk}: {desc}")
    continuity_str = "\n".join(continuity_lines) if continuity_lines else "No established continuity history."

    cur_raw = current_raw_speaker or "Unknown"

    prompt = f"""You are an expert conversation analyst for industrial machinery technical support calls.
Your objective is to identify whether the CURRENT speech turn was spoken by the CUSTOMER or the SUPPORT AGENT.

DO NOT apply rigid keyword matching (e.g. do not assume questions are always Customer, or helpful sentences are always Agent).
Reason carefully from conversational context, dialogue turn-taking, and technical semantics.

Role Definitions & Signals:
- CUSTOMER:
  * Reports machine faults, breakdowns, or operational problems (e.g. WP-400, MC-250, error codes like E-102).
  * Describes symptoms, machine noises, fan status, physical observations ("my machine", "our unit", "extraction fan stopped").
  * Requests assistance, asks what to inspect, or expresses confusion/distress.
  * Responds to troubleshooting questions with machine status observations.

- AGENT:
  * Greets the caller or introduces support ("Welcome to technical support", "How can I help you?").
  * Offers assistance or acknowledges the reported problem ("Let me look into that for you").
  * Asks diagnostic questions ("What error code is showing?", "Is the LED flashing?").
  * Provides procedural instructions or troubleshooting steps ("Check the breaker", "Inspect filter cartridge").
  * Refers to verifying manuals or technical documentation.

- UNKNOWN:
  * Use UNKNOWN if the turn is brief (e.g. "Okay", "Yes", "Right") and cannot be confidently attributed from context.
  * Use UNKNOWN if dialogue evidence is ambiguous, contradictory, or lacks high confidence.

Acoustic Speaker Continuity Context:
{continuity_str}

Recent Dialogue History:
{history_str}

Current Turn to Classify:
Acoustic Speaker Label: {cur_raw}
Transcript: "{current_turn_text.strip()}"

Instructions:
1. Determine the role: CUSTOMER, AGENT, or UNKNOWN.
2. Provide a confidence score between 0.0 and 1.0 based on how clear the evidence is.
3. Return ONLY a valid JSON object matching this exact schema:
{{
  "role": "CUSTOMER|AGENT|UNKNOWN",
  "confidence": 0.95,
  "rationale": "Brief contextual reason"
}}"""
    return prompt


def classify_speaker_role(
    current_turn: Dict[str, Any],
    recent_turns: Optional[List[Dict[str, Any]]] = None,
    tracker: Optional[SpeakerContinuityTracker] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Classifies the semantic speaker role of a finalized speech turn using Gemini 3.5 Flash Lite.

    Args:
        current_turn: Dict containing 'text' and optional 'raw_speaker'.
        recent_turns: Optional list of previous 2-4 finalized turn dictionaries.
        tracker: Optional SpeakerContinuityTracker instance.
        api_key: Optional Gemini API key override.
        client: Optional pre-configured genai.Client instance (useful for unit tests).
        threshold: Optional confidence threshold override (defaults to 0.80).

    Returns:
        Dict[str, Any] with:
            - role: 'CUSTOMER' | 'AGENT' | 'UNKNOWN'
            - confidence: float (0.0 to 1.0)
            - raw_role: str (model's original role)
            - latency_ms: float (measured execution time in milliseconds)
            - rationale: str
    """
    start_time = time.perf_counter()
    effective_threshold = threshold if threshold is not None else get_confidence_threshold()

    turn_text = str(current_turn.get("text", "")).strip()
    raw_speaker = current_turn.get("raw_speaker") or current_turn.get("speaker_label")

    # If turn text is completely empty, immediately return UNKNOWN
    if not turn_text:
        return {
            "role": "UNKNOWN",
            "confidence": 0.0,
            "raw_role": "UNKNOWN",
            "latency_ms": 0.0,
            "rationale": "Empty transcript text."
        }

    # Resolve API key if client not pre-injected
    resolved_key = (
        (api_key or "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )

    if client is None and (not resolved_key or resolved_key.startswith("your_")):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning("No valid GEMINI_API_KEY available for role classification; falling back to UNKNOWN.")
        return {
            "role": "UNKNOWN",
            "confidence": 0.0,
            "raw_role": "UNKNOWN",
            "latency_ms": elapsed_ms,
            "rationale": "GEMINI_API_KEY not configured."
        }

    # Resolve model
    model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    continuity_summary = tracker.get_continuity_summary() if tracker else {}
    prompt = build_role_classifier_prompt(
        current_turn_text=turn_text,
        current_raw_speaker=raw_speaker,
        recent_turns=recent_turns,
        continuity_summary=continuity_summary,
    )

    try:
        genai_client = client or get_shared_gemini_client(api_key=resolved_key) or genai.Client(api_key=resolved_key)
        response = genai_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error("Gemini speaker role classification API call failed: %s", exc)
        return {
            "role": "UNKNOWN",
            "confidence": 0.0,
            "raw_role": "UNKNOWN",
            "latency_ms": elapsed_ms,
            "rationale": f"Gemini API error: {exc}"
        }

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    raw_response_text = getattr(response, "text", None)
    if not raw_response_text or not raw_response_text.strip():
        logger.warning("Empty response from Gemini speaker role classifier.")
        return {
            "role": "UNKNOWN",
            "confidence": 0.0,
            "raw_role": "UNKNOWN",
            "latency_ms": elapsed_ms,
            "rationale": "Empty response from model."
        }

    # Parse and validate JSON
    cleaned_json = _strip_markdown_fences(raw_response_text)
    try:
        parsed_data = json.loads(cleaned_json)
    except Exception as exc:
        logger.error("Failed to parse JSON from speaker role classifier: %s (raw text: %r)", exc, raw_response_text[:120])
        return {
            "role": "UNKNOWN",
            "confidence": 0.0,
            "raw_role": "UNKNOWN",
            "latency_ms": elapsed_ms,
            "rationale": f"Malformed model JSON: {exc}"
        }

    raw_role = str(parsed_data.get("role", "UNKNOWN")).strip().upper()
    try:
        raw_confidence = float(parsed_data.get("confidence", 0.0))
        raw_confidence = max(0.0, min(1.0, raw_confidence))
    except (ValueError, TypeError):
        raw_confidence = 0.0

    rationale = str(parsed_data.get("rationale", "")).strip()

    # Decision Gate:
    # confidence >= threshold -> accept CUSTOMER or AGENT
    # confidence < threshold  -> UNKNOWN
    if raw_role in {"CUSTOMER", "AGENT"} and raw_confidence >= effective_threshold:
        final_role = raw_role
    else:
        final_role = "UNKNOWN"

    # Record continuity evidence if tracker provided and confidence is high
    if tracker and final_role in {"CUSTOMER", "AGENT"}:
        tracker.record_turn(raw_speaker, final_role, raw_confidence)

    return {
        "role": final_role,
        "confidence": raw_confidence,
        "raw_role": raw_role,
        "latency_ms": elapsed_ms,
        "rationale": rationale
    }

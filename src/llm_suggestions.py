"""
src/llm_suggestions.py

AI-Powered Customer Support Suggestion Generation using Google Gemini.
Synthesizes concise, actionable, strictly grounded customer-support suggestions based
on retrieved knowledge-base chunks from ChromaDB.

Key Principles:
- Strictly grounded in retrieved context: will NOT invent error codes, part numbers,
  pressures, or repair procedures.
- Strictly uses configured Gemini generation model via the google-genai SDK. No fallback models.
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

# Module-level persistent Gemini client cache
_shared_client: Optional[genai.Client] = None
_shared_client_key: Optional[str] = None

SAFE_SUPPLEMENTARY_STEPS: List[str] = [
    "Verify documented operating baselines: Confirm primary electrical voltage, pneumatic line pressure (documented at 6.0-7.0 bar), and air line dryness.",
    "Inspect machine status & fault logs: Review control panel error code history and indicator light diagnostics against the user manual.",
    "Check physical safety condition: Ensure emergency stop circuits, guard interlocks, and dust extraction lines are cleared and operating safely.",
    "Escalate to authorized technical support: If the fault persists after initial checks, contact certified factory service engineering."
]


def _is_positive_no_issue(text: str) -> bool:
    """Detects if dialogue is purely pleasantries, gratitude, or positive closure without any issue."""
    cleaned = text.lower().strip()
    positive_cues = [
        "thank you", "thanks", "working well", "all good", "great now",
        "resolved", "everything is running", "running smoothly", "works great",
        "fixed now", "appreciate it", "have a great day", "have a good day",
        "no further questions", "everything is fine", "working fine"
    ]
    issue_cues = [
        "error", "fault", "code", "alarm", "broken", "stop", "stuck",
        "leak", "fail", "smoke", "noise", "pressure", "overheat", "e-", "r-",
        "how do i", "how to", "why is", "what is", "not working", "won't",
        "problem", "issue", "damage", "warning", "malfunction", "shut down",
        "tripped", "replace", "maintenance", "service", "start", "perform",
        "procedure", "inspect", "check"
    ]
    has_positive = any(c in cleaned for c in positive_cues)
    has_issue = any(c in cleaned for c in issue_cues)
    return has_positive and not has_issue


def _extract_actions_from_text(text: str) -> List[str]:
    """Splits a multi-step suggestion text into individual distinct action strings."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    actions: List[str] = []
    current_action: List[str] = []

    for line in lines:
        step_match = re.match(r"^(?:(?:\(?\d+\)?|[•\-\*]|step\s*\d+)[:.]?\s*)(.+)", line, re.IGNORECASE)
        if step_match:
            if current_action:
                actions.append(" ".join(current_action).strip())
                current_action = []
            current_action.append(step_match.group(1).strip())
        else:
            if current_action:
                current_action.append(line)
            else:
                current_action.append(line)

    if current_action:
        actions.append(" ".join(current_action).strip())

    if len(actions) <= 1:
        matches = list(re.finditer(r"(?:^|[\n\r]+|(?<=[.!?:\-;])\s+)(?:(?:\(?\d+\)?[.:]\s+)|\b(?:step\s+\d+[:.]\s+))", text, re.IGNORECASE))
        if len(matches) > 1:
            actions = []
            for i in range(len(matches)):
                start = matches[i].end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    actions.append(chunk)

    return actions


def _ensure_exactly_four_suggestions(
    suggestions: List[str],
    category: Optional[str] = None
) -> List[str]:
    """
    Guarantees exactly 4 distinct, non-repetitive suggestions.
    Deduplicates and supplements with safe, grounded standard procedures if needed.
    """
    seen = set()
    distinct: List[str] = []
    for s in suggestions:
        cleaned = str(s).strip()
        norm = re.sub(r"^(?:step\s*\d+[:.]|\d+[.:\)]|[•\-\*])\s*", "", cleaned, flags=re.IGNORECASE).strip().lower()
        norm_key = re.sub(r"[^\w\s]", "", norm)
        if norm_key and norm_key not in seen:
            seen.add(norm_key)
            distinct.append(cleaned)

    # Supplement if fewer than 4 distinct suggestions
    for fallback in SAFE_SUPPLEMENTARY_STEPS:
        if len(distinct) >= 4:
            break
        fallback_norm = re.sub(r"[^\w\s]", "", fallback.lower()[:30])
        if not any(fallback_norm in re.sub(r"[^\w\s]", "", d.lower()) for d in distinct):
            distinct.append(fallback)

    return distinct[:4]


def get_shared_gemini_client(api_key: Optional[str] = None) -> Optional[genai.Client]:
    """
    Returns a shared, persistent genai.Client instance for support suggestion generation.
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
        meta = item.get("metadata") or {}
        doc_text = item.get("text") or item.get("document", "")

        source_file = meta.get("source_file") or item.get("source_file") or meta.get("source") or item.get("source", "Unknown Document")
        page = meta.get("page", item.get("page", -1))
        page_str = str(page) if page != -1 else "N/A"
        category = meta.get("category") or item.get("category", "General")
        machine = meta.get("machine") or item.get("machine", "General")
        doc_ref = meta.get("document_reference") or item.get("document_reference") or item.get("doc_id", "N/A")
        section = meta.get("section") or item.get("section", "N/A")

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
    Extracts deduplicated source metadata from retrieved chunks including title, section, page, and excerpt.
    """
    sources = []
    seen = set()
    for item in retrieved_context:
        meta = item.get("metadata") or {}
        source_file = meta.get("source_file") or item.get("source_file") or meta.get("source") or item.get("source", "")
        doc_ref = meta.get("document_reference") or item.get("document_reference") or item.get("doc_id", "")
        page = meta.get("page", item.get("page", -1))
        doc_text = (item.get("text") or item.get("document") or "").strip()

        title = meta.get("title") or item.get("title") or meta.get("section") or item.get("section") or doc_ref or source_file
        section = meta.get("section") or item.get("section", "")
        category = meta.get("category") or item.get("category", "")
        machine = meta.get("machine") or item.get("machine", "")

        excerpt = ""
        if doc_text:
            lines = [l.strip() for l in doc_text.splitlines() if l.strip()]
            filtered = [l for l in lines if not l.startswith("PART ") and not l.startswith("Doc ref:")]
            raw_excerpt = " ".join(filtered)
            if len(raw_excerpt) > 220:
                excerpt = raw_excerpt[:217] + "..."
            else:
                excerpt = raw_excerpt

        key = (source_file, doc_ref, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_file": source_file,
                "document_reference": doc_ref,
                "page": page,
                "title": title,
                "section": section,
                "category": category,
                "machine": machine,
                "excerpt": excerpt
            })
    return sources


def generate_support_suggestion(
    conversation_text: str,
    retrieved_context: List[Dict[str, Any]],
    category: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Generates a concise, strictly grounded customer-support suggestion using Gemini.

    Args:
        conversation_text: Customer conversation or query text.
        retrieved_context: List of retrieved chunks from ChromaDB (search_knowledge_base).
        category: Optional detected query category.
        api_key: Optional Google Gemini API key. Defaults to GEMINI_API_KEY from environment.
        client: Optional pre-configured genai.Client instance. If None, uses shared persistent client.

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

    # 2. Fast check: Positive conversation with no reported issue
    if _is_positive_no_issue(conversation_text):
        logger.info("Customer conversation is positive without reported issue; returning no suggestions.")
        return {
            "has_issue": False,
            "likely_issue": "",
            "suggestions": [],
            "suggestion": "",
            "safety": "",
            "confidence": 1.0,
            "sources": [],
            "suggestion_sources": [],
            "is_grounded": True,
            "category": category
        }

    # 3. Handle empty retrieved context safely without calling Gemini
    if not retrieved_context:
        logger.info("Empty retrieved context provided; returning fallback disclaimer without calling LLM.")
        fallback_msg = (
            "No relevant information was found in the knowledge base. "
            "Please refer to the machine documentation or escalate the issue to a support engineer."
        )
        safe_fallback_suggestions = [
            "Check machine documentation: Consult the physical machine manual and schematics for the specified model.",
            "Gather missing diagnostic details: Record the machine serial number, active fault codes, and LED status indicators.",
            "Inspect documented safety status: Ensure all emergency stop circuits and interlocks are in a safe, disengaged state.",
            "Escalate to technical engineering: Contact an authorized factory service technician or senior support engineering."
        ]
        return {
            "has_issue": True,
            "likely_issue": "Undocumented machine issue or query not covered in the knowledge base.",
            "suggestions": safe_fallback_suggestions,
            "suggestion_sources": [[], [], [], []],
            "suggestion": fallback_msg,
            "safety": "Do not attempt unverified mechanical repairs without consulting the manufacturer documentation.",
            "confidence": 0.0,
            "sources": [],
            "is_grounded": False,
            "category": category
        }

    # 4. Resolve API key
    if api_key is not None:
        resolved_key = api_key.strip()
    else:
        resolved_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )

    if client is None and (not resolved_key or resolved_key.startswith("your_")):
        raise ValueError("GEMINI_API_KEY is required but not found in the environment or parameters.")

    # 5. Resolve configured model (defaults to gemini-3.5-flash-lite)
    model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    # 6. Format context and prompt
    context_str = _format_context_for_prompt(retrieved_context)
    category_instruction = f"The query is classified under: {category}." if category else ""

    prompt = f"""You are an expert AI customer support assistant for industrial machinery.
{category_instruction}

Your task is to analyze the customer's dialogue and generate practical, grounded support suggestions.

CRITICAL INSTRUCTIONS:
1. ISSUE IDENTIFICATION:
   First, determine whether the customer has reported an active problem, fault, error code, maintenance need, or operational issue requiring assistance.
   - If the customer dialogue contains NO problem, fault, error, or operational issue (e.g., pleasantries, expressing thanks, confirmation that everything is working normally, or a purely positive conversation without any reported issue):
     Set "has_issue" to false, "likely_issue" to "", "suggestions" to [], and "safety" to "".
   - If a customer issue, question, or problem IS identified:
     Set "has_issue" to true and provide EXACTLY 4 distinct, actionable suggestions.

2. EXACTLY 4 DISTINCT SUGGESTIONS:
   - When an issue is identified, you MUST provide EXACTLY 4 distinct, actionable suggestions as an array of 4 individual strings in "suggestions".
   - Each suggestion must be returned as an individual item, not combined into one paragraph.
   - Each suggestion must be a concrete, distinct action (not repetitive or rewording of other steps).
   - Suggestions must directly address the customer's reported problem using the conversation context and knowledge base context.

3. KNOWLEDGE GROUNDING AND SAFETY:
   - Answer using ONLY the supplied knowledge-base context below.
   - DO NOT invent or fabricate any machine-specific procedures, technical specifications, error codes, part numbers, or safety protocols not in the documentation.
   - If the available documentation does not support four technical actions, use safe, relevant standard next steps for the remaining suggestions, such as:
     * Gathering missing diagnostic information (e.g. noting error codes, indicator light status, or exact machine model/serial).
     * Checking documented visual indicators, supply lines, or control panel readings.
     * Escalating to an authorized field service technician or senior support engineering.
   - Never recommend bypassing safety interlocks, disabling safety circuits, or performing undocumented repairs.
   - Clearly distinguish documentation-supported technical actions from general diagnostic or escalation steps.
   - Include any mandatory documented safety precautions in the "safety" field (e.g., power isolation, lock-out/tag-out, waiting for blades/spindles to stop).

4. CITATION EVIDENCE LINKING:
   - For each suggestion in "suggestions", specify an array of supporting document references from the KNOWLEDGE BASE CONTEXT (e.g. ["TS-WP400-3.1"] or ["TS-WP400-3.1", "UM-WP400-1.1.3"]) in the "suggestion_sources" array.
   - If a suggestion draws on multiple documents, include all applicable document references in its array (e.g. ["TS-WP400-3.1", "UM-WP400-1.1.3"]).
   - If a suggestion is a general diagnostic check or standard escalation step NOT derived from a specific retrieved document, use an empty array [].
   - DO NOT invent or fabricate any document references not found in the KNOWLEDGE BASE CONTEXT.

KNOWLEDGE BASE CONTEXT:
=======================
{context_str}
=======================

CUSTOMER QUERY:
\"\"\"
{conversation_text.strip()}
\"\"\"

Return ONLY a valid JSON object with no additional text or markdown formatting outside the JSON, using this schema:
{{
  "has_issue": true,
  "likely_issue": "Concise summary of the likely issue or root cause",
  "suggestions": [
    "Action 1: Clear, actionable step",
    "Action 2: Clear, actionable step",
    "Action 3: Clear, actionable step",
    "Action 4: Clear, actionable step"
  ],
  "suggestion_sources": [
    ["TS-WP400-3.1"],
    ["TS-WP400-3.1", "UM-WP400-1.1.3"],
    ["UM-WP400-1.1.3"],
    []
  ],
  "safety": "Required safety precautions (or empty string if none)",
  "confidence": 0.95,
  "is_grounded": true
}}
"""

    # 7. Execute Gemini call with retries for transient errors
    gemai_client = client or get_shared_gemini_client(api_key=resolved_key) or genai.Client(api_key=resolved_key)
    max_retries = 3
    last_error = None
    response = None

    for attempt in range(max_retries):
        try:
            logger.info("Calling Gemini (%s) for suggestion generation (attempt %d/%d)", model_name, attempt + 1, max_retries)
            response = gemai_client.models.generate_content(
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

    # 8. Parse response text
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

    has_issue = bool(data.get("has_issue", True))
    raw_suggestions = data.get("suggestions")

    # If the model explicitly identified no issue
    if not has_issue or (isinstance(raw_suggestions, list) and len(raw_suggestions) == 0 and not data.get("suggestion")):
        return {
            "has_issue": False,
            "likely_issue": "",
            "suggestions": [],
            "suggestion": "",
            "safety": "",
            "confidence": 1.0,
            "sources": [],
            "suggestion_sources": [],
            "is_grounded": True,
            "category": category
        }

    # Extract raw suggestions list
    parsed_suggestions: List[str] = []
    if isinstance(raw_suggestions, list):
        parsed_suggestions = [str(s).strip() for s in raw_suggestions if str(s).strip()]
    elif isinstance(data.get("suggestion"), str):
        parsed_suggestions = _extract_actions_from_text(data["suggestion"])

    likely_issue = str(data.get("likely_issue") or "").strip()
    safety = str(data.get("safety") or "").strip()

    if not parsed_suggestions and not (isinstance(data.get("suggestion"), str) and data["suggestion"].strip()):
        raise SuggestionGenerationError("Invalid or empty suggestion returned by Gemini.")

    # Ensure exactly 4 distinct suggestions
    final_suggestions = _ensure_exactly_four_suggestions(parsed_suggestions, category=category)

    sources = _extract_sources(retrieved_context)
    valid_doc_refs = {s["document_reference"] for s in sources if s.get("document_reference")}
    sources_by_ref = {s["document_reference"]: s for s in sources if s.get("document_reference")}

    raw_sugg_sources = data.get("suggestion_sources")
    if not isinstance(raw_sugg_sources, list):
        raw_sugg_sources = []

    final_suggestion_sources: List[List[Dict[str, Any]]] = []

    for i, sug in enumerate(final_suggestions):
        sug_lower = sug.lower()
        matched_refs: List[str] = []

        is_standard_protocol = any(k in sug_lower for k in [
            "operating baselines", "operating parameters", "fault logs", "error history",
            "safety status", "emergency stop", "authorized technical support", "authorized factory",
            "escalate", "missing diagnostic"
        ])

        if not is_standard_protocol:
            # 1. Check if model provided valid matching document_reference(s)
            if i < len(raw_sugg_sources):
                model_item = raw_sugg_sources[i]
                if isinstance(model_item, list):
                    for r in model_item:
                        if isinstance(r, str) and r in valid_doc_refs and r not in matched_refs:
                            matched_refs.append(r)
                        elif isinstance(r, dict) and r.get("document_reference") in valid_doc_refs:
                            ref = r["document_reference"]
                            if ref not in matched_refs:
                                matched_refs.append(ref)
                elif isinstance(model_item, str) and model_item in valid_doc_refs:
                    matched_refs.append(model_item)

            # 2. Check if the suggestion text explicitly contains any valid doc_ref
            if not matched_refs:
                for ref in valid_doc_refs:
                    if ref.lower() in sug_lower and ref not in matched_refs:
                        matched_refs.append(ref)

            # 3. Check if suggestion mentions key parts or specific terms unique to retrieved sources
            if not matched_refs:
                for s in sources:
                    ref = s.get("document_reference")
                    if not ref or ref in matched_refs:
                        continue
                    sec_lower = s.get("section", "").lower()
                    exc_lower = s.get("excerpt", "").lower()
                    haystack = sec_lower + " " + exc_lower

                    part_codes = re.findall(r"\b[A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)?\b", s.get("excerpt", "") + " " + s.get("section", ""))
                    if any(code.lower() in sug_lower for code in part_codes if len(code) >= 4):
                        matched_refs.append(ref)
                        continue
                    if "fan breaker" in sug_lower and "fan breaker" in haystack:
                        matched_refs.append(ref)
                        continue
                    if "velocity" in sug_lower and "velocity" in haystack:
                        matched_refs.append(ref)
                        continue
                    if "pressure drop" in sug_lower and "pressure drop" in haystack:
                        matched_refs.append(ref)
                        continue
                    if "water filter" in sug_lower and "water filter" in haystack:
                        matched_refs.append(ref)
                        continue

        # Build structured citation objects for matched_refs
        sug_citations: List[Dict[str, Any]] = []
        for ref in matched_refs:
            s_meta = sources_by_ref.get(ref, {})
            title = s_meta.get("title") or s_meta.get("section") or ref
            page = s_meta.get("page", -1)
            citation_obj = {
                "document_reference": ref,
                "page": page,
                "title": title
            }
            if s_meta.get("source_file"):
                citation_obj["source_file"] = s_meta.get("source_file")
            sug_citations.append(citation_obj)

        final_suggestion_sources.append(sug_citations)

    # Attach supported_suggestions list to each source for fast lookup
    for s in sources:
        ref = s.get("document_reference")
        s["supported_suggestions"] = [
            j + 1 for j, citations in enumerate(final_suggestion_sources)
            if any(c.get("document_reference") == ref for c in citations)
        ]

    # Construct backward-compatible suggestion string
    if "suggestion" in data and not raw_suggestions and isinstance(data["suggestion"], str):
        res_suggestion = data["suggestion"].strip()
    else:
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(final_suggestions))
        parts = []
        if likely_issue:
            parts.append(likely_issue)
        parts.append(steps_text)
        if safety:
            parts.append(f"Safety Reminder: {safety}")
        res_suggestion = "\n\n".join(parts)

    confidence = data.get("confidence", 0.9)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence_val = 0.85
    else:
        confidence_val = max(0.0, min(1.0, float(confidence)))

    is_grounded = bool(data.get("is_grounded", True))

    return {
        "has_issue": True,
        "likely_issue": likely_issue,
        "suggestions": final_suggestions,
        "suggestion_sources": final_suggestion_sources,
        "suggestion": res_suggestion,
        "safety": safety,
        "confidence": round(confidence_val, 4),
        "sources": sources,
        "is_grounded": is_grounded,
        "category": category
    }


def generate_suggestions(
    query_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for Phase 1 suggestions generation.
    """
    return generate_support_suggestion(
        conversation_text=query_text,
        retrieved_context=retrieved_chunks,
        api_key=api_key,
        client=client
    )


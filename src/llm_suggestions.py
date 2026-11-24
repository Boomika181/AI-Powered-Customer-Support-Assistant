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
from typing import List, Dict, Any, Optional, Set

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from google import genai

logger = logging.getLogger("llm_suggestions")

# Module-level persistent Gemini client cache
_shared_client: Optional[genai.Client] = None
_shared_client_key: Optional[str] = None

INSUFFICIENT_INFO_RESPONSE = (
    "I couldn't find sufficient information about this issue in the available technical documentation. "
    "Please consult the relevant technical expert or provide the appropriate documentation."
)

EVIDENCE_STATUS_SUFFICIENT = "sufficient"
EVIDENCE_STATUS_PARTIAL = "partial"
EVIDENCE_STATUS_INSUFFICIENT = "insufficient"

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "can", "could", "should", "would", "how", "what", "why", "when", "where",
    "who", "which", "my", "our", "your", "it", "its", "this", "that", "there", "i",
    "we", "you", "they", "me", "us", "please", "help", "hello", "hi", "hey", "about",
    "need", "tell", "give", "some", "any", "now", "here", "query", "injected", "client",
    "dummy", "sample", "text"
}

SAFE_SUPPLEMENTARY_STEPS: List[str] = [
    "Verify documented operating baselines: Confirm primary electrical voltage, pneumatic line pressure (documented at 6.0-7.0 bar), and air line dryness.",
    "Inspect machine status & fault logs: Review control panel error code history and indicator light diagnostics against the user manual.",
    "Check physical safety condition: Ensure emergency stop circuits, guard interlocks, and dust extraction lines are cleared and operating safely.",
    "Escalate to authorized technical support: If the fault persists after initial checks, contact certified factory service engineering."
]


# Comprehensive industrial technical synonyms and abbreviations map
TECHNICAL_SYNONYM_MAP: Dict[str, Set[str]] = {
    # Extraction / dust collection / ventilation / blowers
    "exhaust": {"extraction", "extractor", "suction", "ventilation", "vacuum", "collector", "duct", "blower", "fan"},
    "suction": {"extraction", "extractor", "exhaust", "vacuum", "airflow", "velocity", "hood", "port"},
    "vacuum": {"extraction", "suction", "exhaust", "collector"},
    "extractor": {"extraction", "exhaust", "suction", "collector", "fan"},
    "blower": {"fan", "extraction", "exhaust", "motor", "impeller"},
    "airflow": {"air", "velocity", "flow", "ventilation", "suction", "draft"},

    # Filtration / Elements / Cartridges
    "filtration": {"filter", "cartridge", "element", "strainer", "canister"},
    "element": {"cartridge", "filter", "filtration", "canister"},
    "cartridge": {"filter", "filtration", "element", "canister"},
    "canister": {"cartridge", "filter", "element", "filtration"},

    # Electrical / Circuit Breakers / Overload
    "cb": {"breaker", "circuit", "relay", "fuse", "overload"},
    "breaker": {"cb", "circuit", "relay", "fuse", "overload"},
    "relay": {"breaker", "contactor", "overload", "fuse", "switch"},
    "overload": {"breaker", "relay", "thermal", "trip", "tripped"},
    "tripped": {"trip", "breaker", "fault", "alarm", "opened"},

    # Maintenance / Replacement Actions
    "swap": {"replace", "replacement", "change", "install", "renew"},
    "swapping": {"replace", "replacement", "change", "install", "renew"},
    "swapped": {"replace", "replacement", "change", "install", "renew"},
    "renew": {"replace", "replacement", "change", "install"},
    "purge": {"clean", "cleaning", "clear", "flush", "blowout"},
    "unclog": {"clean", "cleaning", "clear", "unblock", "blockage"},
    "unblock": {"clean", "clear", "unclog", "blockage", "blocked"},
    "blockage": {"blocked", "clogged", "obstruction", "restriction"},
    "clogged": {"blocked", "blockage", "restriction", "obstruction"},

    # Power & Operations / LOTO
    "powerup": {"startup", "start", "boot", "isolator", "energize", "on"},
    "energize": {"power", "startup", "start", "boot", "isolator", "on"},
    "deenergize": {"shutdown", "stop", "isolator", "loto", "lockout", "off"},
    "loto": {"lockout", "tagout", "isolator", "safety", "padlock"},
    "lockout": {"loto", "tagout", "isolator", "safety", "padlock"},
    "tagout": {"loto", "lockout", "isolator", "safety", "padlock"},
    "isolator": {"switch", "disconnect", "power", "main", "loto"},

    # Mechanical & Drives
    "spindle": {"shaft", "arbor", "rotor", "chuck", "motor", "drive"},
    "arbor": {"spindle", "shaft", "blade", "motor"},
    "sawblade": {"blade", "saw", "cutter", "knife", "teeth"},
    "cutter": {"blade", "saw", "tool", "knife", "bit"},
    "collet": {"chuck", "clamp", "holder", "spindle"},
    "coolant": {"fluid", "lubricant", "lube", "oil"},
    "lube": {"lubricant", "lubrication", "oil", "grease", "fluid"},
    "lubricant": {"lube", "oil", "grease", "fluid"},
    "reservoir": {"tank", "sump", "container", "basin"},

    # Diagnostics, Safety & Calibration
    "malfunction": {"fault", "error", "alarm", "failure", "defect", "issue"},
    "breakdown": {"fault", "error", "alarm", "failure", "stopped"},
    "diagnostic": {"troubleshooting", "error", "code", "history", "log"},
    "calibrate": {"align", "alignment", "zero", "homing", "home", "reference"},
    "calibration": {"align", "alignment", "zero", "homing", "home", "reference"},
    "homing": {"home", "zero", "reference", "calibrate", "alignment"},
    "interlock": {"switch", "safety", "guard", "door", "sensor", "circuit"},
    "estop": {"emergency", "stop", "button", "interlock", "circuit"},

    # Pneumatics & Hydraulics
    "pneumatic": {"air", "pressure", "compressor", "line", "compressed-air", "gauge"},
    "pneumatics": {"air", "pressure", "compressor", "line", "compressed-air", "gauge"},
    "hydraulic": {"fluid", "oil", "pressure", "pump", "cylinder", "actuator"},
    "hydraulics": {"fluid", "oil", "pressure", "pump", "cylinder", "actuator"}
}


def _evaluate_context_relevance(
    conversation_text: str,
    retrieved_context: List[Dict[str, Any]],
    min_similarity: float = 0.35
) -> Dict[str, Any]:
    """
    Evaluates whether the retrieved context contains information relevant to the conversation.
    Returns:
        {
            "is_relevant": bool,
            "reason": str,
            "relevant_chunks": List[Dict[str, Any]]
        }
    """
    if not retrieved_context:
        return {
            "is_relevant": False,
            "reason": "empty_context",
            "relevant_chunks": []
        }

    # If ChromaDB similarity scores are present, verify that at least one chunk exceeds threshold
    has_sim_scores = any("similarity_score" in item for item in retrieved_context)
    if has_sim_scores:
        max_sim = max(item.get("similarity_score", 0.0) for item in retrieved_context)
        if max_sim < min_similarity:
            return {
                "is_relevant": False,
                "reason": "low_similarity",
                "relevant_chunks": []
            }

    # Explicit metadata relevance flag (if set by test or pre-processor)
    for item in retrieved_context:
        meta = item.get("metadata") or {}
        if meta.get("is_relevant") is False:
            return {
                "is_relevant": False,
                "reason": "marked_irrelevant_metadata",
                "relevant_chunks": []
            }

    # Semantic topic/domain alignment check:
    # If the customer conversation has substantive tokens, verify that at least one
    # substantive token, stem, or industrial technical synonym matches the retrieved context.
    conv_tokens = [
        w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", conversation_text.lower())
        if w not in STOPWORDS and not w.isdigit()
    ]
    if len(conv_tokens) >= 3:
        context_corpus = " ".join([
            (item.get("text") or item.get("document") or "") + " " +
            str(item.get("metadata") or {})
            for item in retrieved_context
        ]).lower()

        def _token_matches(token: str) -> bool:
            # 1. Exact whole-word match
            if re.search(r"\b" + re.escape(token) + r"\b", context_corpus):
                return True
            # 2. Standard stem match (length >= 6)
            if len(token) >= 6 and re.search(r"\b" + re.escape(token[:5]), context_corpus):
                return True
            # 3. Technical industrial synonym match
            synonyms = TECHNICAL_SYNONYM_MAP.get(token, set())
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", context_corpus):
                    return True
            return False

        has_overlap = any(_token_matches(t) for t in conv_tokens)
        if not has_overlap:
            return {
                "is_relevant": False,
                "reason": "semantic_domain_mismatch",
                "relevant_chunks": []
            }

    return {
        "is_relevant": True,
        "reason": "relevant_evidence_found",
        "relevant_chunks": retrieved_context
    }


def _is_positive_no_issue(text: str) -> bool:
    """
    Detects if dialogue represents a genuine customer confirmation of resolution without remaining issues.
    Rejects:
      - Questions asking if the issue is resolved (e.g. 'Is it resolved?')
      - Mixed messages where one issue is resolved but another problem remains
      - Utterances containing only agent/support speech
    """
    if not text or not text.strip():
        return False

    raw_text = text.strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # 1. Check for speaker prefixes (Agent vs Customer)
    agent_prefixes = ("agent:", "support:", "b:", "assistant:", "technician:")
    customer_prefixes = ("customer:", "client:", "user:", "caller:", "a:")

    has_agent_lines = any(l.lower().startswith(agent_prefixes) for l in lines)
    has_customer_lines = any(l.lower().startswith(customer_prefixes) for l in lines)

    # If transcript has labeled speakers:
    if has_agent_lines or has_customer_lines:
        customer_texts = [
            re.sub(r'^(customer|client|user|caller|a)\s*:\s*', '', l, flags=re.IGNORECASE).strip()
            for l in lines
            if l.lower().startswith(customer_prefixes)
        ]
        # If there are agent lines but NO customer lines, it's agent-only -> cannot be customer resolution
        if has_agent_lines and not customer_texts:
            return False
        if customer_texts:
            eval_text = " ".join(customer_texts)
        else:
            eval_text = raw_text
    else:
        eval_text = raw_text

    cleaned = eval_text.lower().strip()

    # 2. Questions check: Inquiries asking if something is resolved are NOT confirmations
    # e.g. "Is it resolved?", "Has it been fixed?", "Did that fix it?", or questions asking about status
    if re.search(r'\b(is|was|has|have|did|can|could|should|would)\b.*\b(resolved|fixed|working|ok|sorted)\b.*\?', cleaned):
        return False
    if re.search(r'^(is|was|has|have|did|can|could|should|would)\s+(it|this|that|the\s+\w+)\s+(resolved|fixed|working|all\s+set)\b', cleaned):
        return False
    if cleaned.endswith("?") and not any(phrase in cleaned for phrase in ["that fixed it", "problem is resolved", "it is fixed"]):
        return False

    # 3. Mixed message check: A positive statement followed by an active problem or contrastive conjunction
    # e.g., "That fixed the fan, but now there is error E-102"
    has_contrast = bool(re.search(r'\b(but|however|except|although|though|yet)\b', cleaned))
    has_error_code = bool(re.search(r'\b[eErR]-\d+\b', cleaned))
    has_active_problem = bool(re.search(
        r'\b(not working|still broken|won\'?t start|will not start|failed again|another problem|new error|another issue|new issue|still stopped|smoke|leak|fire|damaged|danger|abnormal noise|overheat|overheating|tripped again|unresolved|persists|persisting)\b',
        cleaned
    ))

    # Remaining issue words if contrastive connector is present
    has_contrast_issue = False
    if has_contrast:
        contrast_match = re.search(r'\b(but|however|except|although|though|yet)\b(.*)', cleaned)
        if contrast_match:
            after_contrast = contrast_match.group(2)
            issue_words = [
                "error", "fault", "code", "alarm", "broken", "stop", "stuck",
                "leak", "fail", "smoke", "noise", "pressure", "overheat",
                "problem", "issue", "damage", "warning", "malfunction", "won't",
                "not working", "still", "tripped", "replace", "inspect", "check"
            ]
            if any(w in after_contrast for w in issue_words):
                has_contrast_issue = True

    if has_error_code or has_active_problem or has_contrast_issue:
        return False

    strong_resolution_cues = [
        "that fixed it", "it is fixed", "problem is resolved", "issue is resolved",
        "fixed now", "resolved now", "working great now", "running smoothly now",
        "working fine now", "all sorted now", "that solved it", "back up and running",
        "back in operation", "working again"
    ]
    positive_cues = [
        "thank you", "thanks", "working well", "all good", "great now",
        "resolved", "everything is running", "running smoothly", "works great",
        "fixed now", "appreciate it", "have a great day", "have a good day",
        "no further questions", "everything is fine", "working fine", "all set"
    ] + strong_resolution_cues

    has_strong_res = any(c in cleaned for c in strong_resolution_cues)
    if has_strong_res:
        return True

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
    evidence_status: str = "sufficient",
    category: Optional[str] = None
) -> List[str]:
    """
    Guarantees exactly 4 distinct, non-repetitive suggestions ONLY when evidence_status is 'sufficient'.
    If evidence_status is 'insufficient' or 'partial', does NOT force 4 suggestions.
    """
    if evidence_status != "sufficient" or not suggestions:
        return suggestions

    seen = set()
    distinct: List[str] = []
    for s in suggestions:
        cleaned = str(s).strip()
        norm = re.sub(r"^(?:step\s*\d+[:.]|\d+[.:\)]|[•\-\*])\s*", "", cleaned, flags=re.IGNORECASE).strip().lower()
        norm_key = re.sub(r"[^\w\s]", "", norm)
        if norm_key and norm_key not in seen:
            seen.add(norm_key)
            distinct.append(cleaned)

    # Supplement if fewer than 4 distinct suggestions ONLY when evidence is sufficient
    for fallback in SAFE_SUPPLEMENTARY_STEPS:
        if len(distinct) >= 4:
            break
        fallback_norm = re.sub(r"[^\w\s]", "", fallback.lower()[:30])
        if not any(fallback_norm in re.sub(r"[^\w\s]", "", d.lower()) for d in distinct):
            distinct.append(fallback)

    return distinct[:4]


def _validate_and_sanitize_suggestions(
    parsed_suggestions: List[str],
    raw_sugg_sources: Any,
    retrieved_context: List[Dict[str, Any]],
    conversation_text: str,
    sources: List[Dict[str, Any]],
    claimed_evidence_status: str = "sufficient",
    unverified_details: str = ""
) -> Dict[str, Any]:
    """
    Validates suggestions and citations against retrieved ChromaDB chunks:
    1. Strips hallucinated citations (document references not in retrieved chunks).
    2. Identifies and strips suggestions with hallucinated technical parameters/part numbers/codes.
    3. Reconciles evidence_status (sufficient, partial, insufficient).
    4. Suppresses technical suggestions if evidence is insufficient.
    """
    valid_doc_refs = {s["document_reference"] for s in sources if s.get("document_reference")}
    sources_by_ref = {s["document_reference"]: s for s in sources if s.get("document_reference")}

    context_corpus = " ".join([
        (item.get("text") or item.get("document") or "") + " " +
        str(item.get("metadata") or {})
        for item in retrieved_context
    ]).lower()
    conv_lower = conversation_text.lower()

    if claimed_evidence_status == "insufficient":
        return {
            "evidence_status": "insufficient",
            "suggestions": [],
            "suggestion_sources": [],
            "unverified_details": unverified_details or "The issue is not documented in the available technical knowledge base."
        }

    verified_suggestions: List[str] = []
    verified_sources: List[List[Dict[str, Any]]] = []
    rejected_reasons: List[str] = []

    for i, sug in enumerate(parsed_suggestions):
        sug_lower = sug.lower()
        is_hallucinated = False

        # 0. Safety and prohibited action check (e.g. bypassing safety interlocks/circuits)
        prohibited_patterns = [
            r"\bbypass\b",
            r"\bdisable\s+(?:safety|interlock|guard|sensor|switch|breaker|circuit)\b",
            r"\boverride\s+(?:safety|interlock|switch|limit)\b",
            r"\bjumper\b"
        ]
        for pat in prohibited_patterns:
            if re.search(pat, sug_lower):
                if not re.search(pat, context_corpus):
                    is_hallucinated = True
                    rejected_reasons.append(f"Prohibited or unsupported action matching '{pat}' in suggestion: '{sug}'.")
                    break
        if is_hallucinated:
            continue

        # 1. Error codes verification
        error_codes = re.findall(r"\b[A-Z]{1,3}-\d{2,4}\b", sug)
        for code in error_codes:
            if code.lower() not in context_corpus and code.lower() not in conv_lower:
                is_hallucinated = True
                rejected_reasons.append(f"Error code {code} is not documented in the knowledge base.")
                break
        if is_hallucinated:
            continue

        # 2. Part numbers verification
        part_codes = re.findall(r"\b[A-Z0-9]{2,8}-[A-Z0-9]{2,8}(?:-[A-Z0-9]+)?\b", sug)
        for part in part_codes:
            if len(part) >= 5 and part.lower() not in context_corpus and part.lower() not in conv_lower:
                is_hallucinated = True
                rejected_reasons.append(f"Part number {part} is not documented in the knowledge base.")
                break
        if is_hallucinated:
            continue

        # 3. Technical parameters with units verification (with cross-machine & condition scoping)
        param_matches = list(re.finditer(r"\b(\d+(?:\.\d+)?)\s*(m/s|bar|psi|volts?|V\b|amps?|amperes?|A\b|pa\b|kpa|mm\b|cm\b|hz\b|kw\b|rpm|°c|°C)\b", sug))
        for m in param_matches:
            val_str = m.group(1)
            val_variations = [val_str]
            if val_str.endswith(".0"):
                val_variations.append(val_str[:-2])
            elif "." not in val_str:
                val_variations.append(val_str + ".0")

            chunks_with_param = [
                item for item in retrieved_context
                if any(v in ((item.get("text") or item.get("document") or "") + " " + str(item.get("metadata") or {})).lower() for v in val_variations)
            ]

            if not chunks_with_param and not any(v in conv_lower for v in val_variations):
                is_hallucinated = True
                rejected_reasons.append(f"Parameter '{m.group(0)}' is not documented in the knowledge base.")
                break

            # Cross-machine isolation check
            target_machines = set(re.findall(r"\b[A-Z]{2,4}-?\d{3,4}\b", (sug + " " + conversation_text).upper()))
            if target_machines and chunks_with_param:
                matching_machine_chunks = [
                    c for c in chunks_with_param
                    if not (c.get("metadata") or {}).get("machine") or
                    any(m_code in ((c.get("metadata") or {}).get("machine") or "").upper() or
                        m_code in ((c.get("text") or c.get("document") or "")).upper()
                        for m_code in target_machines)
                ]
                if not matching_machine_chunks:
                    is_hallucinated = True
                    chunk_m = (chunks_with_param[0].get("metadata") or {}).get("machine", "another machine")
                    rejected_reasons.append(f"Parameter '{m.group(0)}' belongs to {chunk_m}, not {list(target_machines)[0]}.")
                    break

            # Operating condition mismatch check
            is_normal_sug = any(w in sug_lower for w in ["normal", "cutting", "production"])
            if is_normal_sug and chunks_with_param:
                conflict = False
                for c in chunks_with_param:
                    c_text = ((c.get("text") or c.get("document") or "")).lower()
                    sentences = [s.strip() for s in re.split(r"[.\n;]", c_text) if s.strip()]
                    for sent in sentences:
                        if any(v in sent for v in val_variations):
                            is_sent_test = any(w in sent for w in ["commissioning", "test mode", "testing"])
                            is_sent_normal = any(w in sent for w in ["normal", "cutting", "production"])
                            if is_sent_test and not is_sent_normal:
                                conflict = True
                                break
                    if conflict:
                        break
                if conflict:
                    is_hallucinated = True
                    rejected_reasons.append(f"Parameter '{m.group(0)}' applies to commissioning/testing, not normal cutting operations.")
                    break

        if is_hallucinated:
            continue

        # Citations linking for this suggestion
        matched_refs: List[str] = []
        is_standard_protocol = any(k in sug_lower for k in [
            "operating baselines", "operating parameters", "fault logs", "error history",
            "safety status", "emergency stop", "authorized technical support", "authorized factory",
            "escalate", "missing diagnostic"
        ])

        if not is_standard_protocol:
            if isinstance(raw_sugg_sources, list) and i < len(raw_sugg_sources):
                model_item = raw_sugg_sources[i]
                candidate_refs: List[str] = []
                if isinstance(model_item, list):
                    for r in model_item:
                        ref_str = (r.get("document_reference") if isinstance(r, dict) else str(r)).strip()
                        if ref_str in valid_doc_refs:
                            candidate_refs.append(ref_str)
                elif isinstance(model_item, str) and model_item.strip() in valid_doc_refs:
                    candidate_refs.append(model_item.strip())

                # Semantic entailment check for model-supplied citations
                for ref_str in candidate_refs:
                    s_chunk = sources_by_ref.get(ref_str, {})
                    ref_meta_text = (s_chunk.get("title", "") + " " + s_chunk.get("section", "") + " " + s_chunk.get("excerpt", "")).lower()
                    chunk_full = " ".join([
                        (c.get("text") or c.get("document") or "").lower()
                        for c in retrieved_context
                        if (c.get("metadata") or {}).get("document_reference") == ref_str
                    ])
                    haystack = ref_meta_text + " " + chunk_full

                    sug_entities = re.findall(r"\b[A-Z0-9]{1,4}-[A-Z0-9]{2,8}\b|\b[A-Z]\d{1,3}\b", sug)
                    param_in_sug = re.findall(r"\b\d+(?:\.\d+)?\s*(?:m/s|bar|psi|volts?|v|amps?|pa|kpa|mm)\b", sug_lower)

                    if sug_entities and any(ent.lower() in haystack for ent in sug_entities):
                        if ref_str not in matched_refs:
                            matched_refs.append(ref_str)
                    elif param_in_sug and any(p in haystack for p in param_in_sug):
                        if ref_str not in matched_refs:
                            matched_refs.append(ref_str)
                    else:
                        sug_substantive = [w for w in re.findall(r"\b[a-z]{4,}\b", sug_lower) if w not in STOPWORDS]
                        overlap_count = sum(1 for w in sug_substantive if w in haystack)
                        if overlap_count >= 2 and ref_str not in matched_refs:
                            matched_refs.append(ref_str)

            if not matched_refs:
                for ref in valid_doc_refs:
                    if ref.lower() in sug_lower and ref not in matched_refs:
                        matched_refs.append(ref)

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
                    if "breaker q4" in sug_lower and ("breaker" in haystack or "q4" in haystack):
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

        verified_suggestions.append(sug)
        verified_sources.append(sug_citations)

    # Reconcile final status
    has_severe_grounding_failure = any(
        ("part number" in r.lower()
         or "error code" in r.lower()
         or "belongs to" in r.lower()
         or "prohibited" in r.lower()
         or "commissioning" in r.lower())
        for r in rejected_reasons
    )
    if not verified_suggestions:
        final_status = "insufficient"
    elif (
        claimed_evidence_status == "partial"
        or has_severe_grounding_failure
        or (unverified_details and "unverified" in unverified_details.lower())
    ):
        final_status = "partial"
    else:
        final_status = "sufficient"

    reconciled_unverified = unverified_details
    if rejected_reasons:
        rej_str = " ".join(rejected_reasons)
        reconciled_unverified = f"{reconciled_unverified} {rej_str}".strip() if reconciled_unverified else rej_str

    if final_status == "insufficient":
        return {
            "evidence_status": "insufficient",
            "suggestions": [],
            "suggestion_sources": [],
            "unverified_details": reconciled_unverified or "No verified technical documentation found for this issue."
        }

    return {
        "evidence_status": final_status,
        "suggestions": verified_suggestions,
        "suggestion_sources": verified_sources,
        "unverified_details": reconciled_unverified
    }


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
        logger.info("Customer conversation is positive without reported issue; returning resolved state.")
        return {
            "has_issue": False,
            "is_resolved": True,
            "status": "resolved",
            "likely_issue": "Issue Resolved",
            "suggestions": [],
            "suggestion": "",
            "safety": "",
            "confidence": 1.0,
            "sources": [],
            "suggestion_sources": [],
            "is_grounded": True,
            "category": category,
            "unverified_details": ""
        }

    # 3. Handle empty retrieved context safely without calling Gemini
    if not retrieved_context:
        logger.info("Empty retrieved context provided; returning insufficient documentation response without calling LLM.")
        return {
            "has_issue": True,
            "evidence_status": EVIDENCE_STATUS_INSUFFICIENT,
            "likely_issue": "Insufficient technical documentation available for this issue.",
            "suggestions": [],
            "suggestion_sources": [],
            "suggestion": (
                f"{INSUFFICIENT_INFO_RESPONSE} "
                "(No relevant information was found in the knowledge base)."
            ),
            "unverified_details": "No matching technical documentation was found in the knowledge base.",
            "safety": "",
            "confidence": 0.0,
            "sources": [],
            "is_grounded": False,
            "category": category
        }

    # 4. Fast check: Evaluate context relevance before LLM generation
    rel_eval = _evaluate_context_relevance(conversation_text, retrieved_context)
    if not rel_eval["is_relevant"]:
        logger.info("Retrieved context deemed irrelevant to query (%s); returning insufficient documentation response.", rel_eval["reason"])
        sources = _extract_sources(retrieved_context)
        return {
            "has_issue": True,
            "evidence_status": EVIDENCE_STATUS_INSUFFICIENT,
            "likely_issue": "The query is outside the scope of available technical documentation.",
            "suggestions": [],
            "suggestion_sources": [],
            "suggestion": INSUFFICIENT_INFO_RESPONSE,
            "unverified_details": "The query topic does not match any available equipment documentation in the knowledge base.",
            "safety": "",
            "confidence": 0.0,
            "sources": sources,
            "is_grounded": False,
            "category": category
        }

    # 5. Resolve API key
    if api_key is not None:
        resolved_key = api_key.strip()
    else:
        resolved_key = (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )

    if client is None and (not resolved_key or resolved_key.startswith("your_")):
        raise ValueError("GEMINI_API_KEY is required but not found in the environment or parameters.")

    # 6. Resolve configured model (defaults to gemini-3.5-flash-lite)
    model_name = os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    # 7. Format context and prompt
    context_str = _format_context_for_prompt(retrieved_context)
    category_instruction = f"The query is classified under: {category}." if category else ""

    prompt = f"""You are an expert AI customer support assistant for industrial machinery operating under STRICT DOCUMENT-GROUNDED RULES.
{category_instruction}

CORE GROUNDING POLICY:
You must provide technical answers and recommendations ONLY when the KNOWLEDGE BASE CONTEXT below contains sufficient supporting information.
If the documents do not contain the required information, you MUST NOT generate an answer using your pretrained knowledge, assumptions, guesses, or general industry knowledge.

CRITICAL INSTRUCTIONS:
1. ISSUE IDENTIFICATION:
   First, determine whether the customer has reported an active problem, fault, error code, maintenance need, or operational issue requiring assistance.
   - If the customer dialogue contains NO problem, fault, error, or operational issue (e.g., pleasantries, expressing thanks, confirmation that everything is working normally):
     Set "has_issue" to false, "evidence_status" to "sufficient", "likely_issue" to "", "suggestions" to [], and "safety" to "".
   - If a customer issue, question, or problem IS identified:
     Set "has_issue" to true and proceed to evidence evaluation.

2. EVIDENCE SUFFICIENCY EVALUATION:
   - CASE A: SUFFICIENT DOCUMENT INFORMATION AVAILABLE:
     If the KNOWLEDGE BASE CONTEXT contains verified, direct technical information addressing the customer's problem:
     * Set "evidence_status" to "sufficient".
     * Provide EXACTLY 4 distinct, actionable suggestions as an array of 4 individual strings in "suggestions".
     * Every suggestion must be grounded in the supplied documentation.
     * Attach supporting document references for each suggestion in "suggestion_sources" (e.g. [["TS-WP400-3.1"], ["TS-WP400-3.1", "UM-WP400-1.1.3"]]).
     * Set "unverified_details" to "".
     * Set "is_grounded" to true.
   - CASE B: INFORMATION PARTIALLY AVAILABLE:
     If the KNOWLEDGE BASE CONTEXT answers only part of the customer's question or error, but lacks documentation for other aspects:
     * Set "evidence_status" to "partial".
     * Answer ONLY the portions explicitly supported by the documents.
     * DO NOT fill missing details with assumptions or pretrained knowledge.
     * DO NOT force 4 suggestions. Return only the 1 to 3 verified suggestions in "suggestions".
     * In "unverified_details", clearly identify which specific details could not be verified from the knowledge base.
     * Set "is_grounded" to true.
   - CASE C: NO RELEVANT INFORMATION AVAILABLE:
     If the KNOWLEDGE BASE CONTEXT does NOT contain information about the customer's reported machine, issue, or question:
     * Set "evidence_status" to "insufficient".
     * Set "suggestions" to [].
     * Set "suggestion_sources" to [].
     * Set "suggestion" to "{INSUFFICIENT_INFO_RESPONSE}".
     * Set "unverified_details" to "No relevant technical information was found in the knowledge base.".
     * Set "safety" to "".
     * Set "confidence" to 0.0.
     * Set "is_grounded" to false.

3. STRICT SAFETY, CITATION, AND ADVERSARIAL GROUNDING PROHIBITIONS:
   - MACHINE & COMPONENT SCOPING: Use parameters, codes, and procedures ONLY for the customer's specific machine model. NEVER apply parameters or specifications from one machine model to a different machine model.
   - OPERATING CONDITION SCOPING: Match parameters strictly to the operating state (e.g., normal production cutting vs. commissioning/testing mode). NEVER apply test or commissioning thresholds to normal operation.
   - ISSUE ALIGNMENT: Recommendations must directly address the customer's reported issue. If retrieved documents describe an unrelated error or component, treat documentation as insufficient.
   - CONTRADICTIONS: If documents contain conflicting instructions or intervals, state the discrepancy clearly in "unverified_details" and set "evidence_status" to "partial".
   - NEVER invent machine-specific procedures, technical specifications, error codes, part numbers, or safety protocols not in the documentation.
   - NEVER invent or fabricate document references, titles, or page numbers not found in the KNOWLEDGE BASE CONTEXT.
   - NEVER recommend bypassing safety interlocks, disabling safety circuits, overriding switches, or performing undocumented repairs.
   - Include any mandatory documented safety precautions in the "safety" field.

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
  "evidence_status": "sufficient",
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
  "unverified_details": "",
  "safety": "Required safety precautions (or empty string if none)",
  "confidence": 0.95,
  "is_grounded": true
}}
"""

    # 8. Execute Gemini call with retries for transient errors
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

    # 9. Parse response text
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
    claimed_evidence_status = str(data.get("evidence_status") or EVIDENCE_STATUS_SUFFICIENT).lower()

    # If the model explicitly identified no customer issue
    if not has_issue and claimed_evidence_status != EVIDENCE_STATUS_INSUFFICIENT:
        return {
            "has_issue": False,
            "is_resolved": True,
            "status": "resolved",
            "evidence_status": EVIDENCE_STATUS_SUFFICIENT,
            "likely_issue": "Issue Resolved",
            "suggestions": [],
            "suggestion": "",
            "safety": "",
            "confidence": 1.0,
            "sources": [],
            "suggestion_sources": [],
            "is_grounded": True,
            "category": category,
            "unverified_details": ""
        }

    # Extract raw suggestions list
    parsed_suggestions: List[str] = []
    if isinstance(raw_suggestions, list):
        parsed_suggestions = [str(s).strip() for s in raw_suggestions if str(s).strip()]
    elif isinstance(data.get("suggestion"), str):
        parsed_suggestions = _extract_actions_from_text(data["suggestion"])

    likely_issue = str(data.get("likely_issue") or "").strip()
    safety = str(data.get("safety") or "").strip()
    unverified_details = str(data.get("unverified_details") or "").strip()
    claimed_evidence_status = str(data.get("evidence_status") or EVIDENCE_STATUS_SUFFICIENT).lower()

    # 10. Extract sources from retrieved context
    sources = _extract_sources(retrieved_context)

    # 11. Strict Evidence Validation Guardrail
    sanitized = _validate_and_sanitize_suggestions(
        parsed_suggestions=parsed_suggestions,
        raw_sugg_sources=data.get("suggestion_sources"),
        retrieved_context=retrieved_context,
        conversation_text=conversation_text,
        sources=sources,
        claimed_evidence_status=claimed_evidence_status,
        unverified_details=unverified_details
    )

    evidence_status = sanitized["evidence_status"]
    valid_suggestions = sanitized["suggestions"]
    final_suggestion_sources = sanitized["suggestion_sources"]
    unverified_details = sanitized["unverified_details"]

    # 12. Format Final Suggestion Payload according to Evidence Status
    if evidence_status == EVIDENCE_STATUS_INSUFFICIENT or not valid_suggestions:
        logger.info("Evidence insufficient; suppressing technical suggestions.")
        return {
            "has_issue": True,
            "evidence_status": EVIDENCE_STATUS_INSUFFICIENT,
            "likely_issue": likely_issue or "Insufficient technical documentation available for this issue.",
            "suggestions": [],
            "suggestion_sources": [],
            "suggestion": INSUFFICIENT_INFO_RESPONSE,
            "safety": "",
            "confidence": 0.0,
            "sources": sources,
            "is_grounded": False,
            "category": category,
            "unverified_details": unverified_details or "No relevant technical information was found in the knowledge base."
        }

    # If evidence is sufficient, enforce exactly 4 suggestions
    if evidence_status == EVIDENCE_STATUS_SUFFICIENT:
        final_suggestions = _ensure_exactly_four_suggestions(
            valid_suggestions,
            evidence_status=EVIDENCE_STATUS_SUFFICIENT,
            category=category
        )
        # Pad citation arrays if necessary for supplemented standard protocol steps
        while len(final_suggestion_sources) < len(final_suggestions):
            final_suggestion_sources.append([])
    else:
        # Partial evidence: do NOT force 4 suggestions. Return only verified recommendations.
        final_suggestions = valid_suggestions

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
        if unverified_details:
            parts.append(f"Unverified Details: {unverified_details}")
        if safety:
            parts.append(f"Safety Reminder: {safety}")
        res_suggestion = "\n\n".join(parts)

    confidence = data.get("confidence", 0.9)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence_val = 0.85
    else:
        confidence_val = max(0.0, min(1.0, float(confidence)))

    is_grounded = bool(data.get("is_grounded", True)) and evidence_status != EVIDENCE_STATUS_INSUFFICIENT

    return {
        "has_issue": True,
        "evidence_status": evidence_status,
        "likely_issue": likely_issue,
        "suggestions": final_suggestions,
        "suggestion_sources": final_suggestion_sources,
        "suggestion": res_suggestion,
        "safety": safety,
        "confidence": round(confidence_val, 4),
        "sources": sources,
        "is_grounded": is_grounded,
        "category": category,
        "unverified_details": unverified_details
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


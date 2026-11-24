"""
tests/test_adversarial_grounding.py

Adversarial audit tests for strict document grounding in AI-Powered Customer Support Assistant.
Tests 9 specific adversarial vectors:
1. Cross-machine / component contamination (value in doc belongs to a different machine/component).
2. Error code present in doc, but does not explain the reported problem.
3. Correct-looking recommendation but attached to an unrelated valid citation.
4. Numerical value in context belongs to a different operating condition.
5. Contradictory instructions across retrieved documents.
6. Multi-step recommendation where only one part is documented.
7. Question outside knowledge base receives high similarity score.
8. Partially supported answer containing an unsupported technical claim.
9. Generic fallback recommendation generated despite insufficient evidence.
"""

import unittest
from unittest.mock import MagicMock, patch
import json

from src.llm_suggestions import (
    generate_support_suggestion,
    _validate_and_sanitize_suggestions,
    _evaluate_context_relevance,
    _ensure_exactly_four_suggestions,
    EVIDENCE_STATUS_SUFFICIENT,
    EVIDENCE_STATUS_PARTIAL,
    EVIDENCE_STATUS_INSUFFICIENT,
    INSUFFICIENT_INFO_RESPONSE
)


class TestAdversarialDocumentGrounding(unittest.TestCase):
    """Adversarial stress-testing of document grounding guardrails."""

    def test_case_1_cross_machine_parameter_contamination(self):
        """
        Case 1: Retrieved documents contain technical parameters, but they belong to a DIFFERENT machine.
        The customer asks about WP-400.
        Doc 1 (WP-400) specifies 20 m/s air velocity.
        Doc 2 (MC-250) specifies 28 m/s air velocity and 160 bar hydraulic pressure.
        If a suggestion applies 28 m/s or 160 bar to the WP-400 or cites the WP-400 doc for it,
        it must be rejected as ungrounded cross-machine contamination.
        """
        retrieved_context = [
            {
                "document": "WP-400 Industrial Saw: Extraction system requires minimum air velocity of 20 m/s at extraction port.",
                "metadata": {
                    "source_file": "manual_wp400.pdf",
                    "document_reference": "UM-WP400-1.1.3",
                    "page": 2,
                    "machine": "WP-400",
                    "section": "Dust Extraction"
                }
            },
            {
                "document": "MC-250 Milling Center: High-volume coolant pump operates at 28 m/s fluid velocity and 160 bar hydraulic pressure.",
                "metadata": {
                    "source_file": "manual_mc250.pdf",
                    "document_reference": "UM-MC250-2.0",
                    "page": 14,
                    "machine": "MC-250",
                    "section": "Coolant & Hydraulics"
                }
            }
        ]
        conversation_text = "Customer: What air velocity and hydraulic pressure should I set for my WP-400 saw?"

        # Mock LLM output that contaminates WP-400 with MC-250's 28 m/s and 160 bar
        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "WP-400 operating parameters",
            "suggestions": [
                "Set WP-400 extraction air velocity to 28 m/s.", # Hallucination: 28 m/s is for MC-250!
                "Maintain hydraulic system pressure at 160 bar on the WP-400.", # Hallucination: 160 bar is for MC-250!
                "Verify WP-400 extraction port has at least 20 m/s air velocity.", # Genuine WP-400 parameter
                "Inspect dust extraction ducting for blockages."
            ],
            "suggestion_sources": [
                ["UM-WP400-1.1.3"], # Falsely cited UM-WP400-1.1.3 for 28 m/s!
                ["UM-WP400-1.1.3"], # Falsely cited UM-WP400-1.1.3 for 160 bar!
                ["UM-WP400-1.1.3"], # Valid
                ["UM-WP400-1.1.3"]
            ],
            "unverified_details": "",
            "safety": "Wear eye protection.",
            "confidence": 0.95,
            "is_grounded": True
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # The suggestions with 28 m/s and 160 bar attributed to WP-400 must NOT be accepted as grounded!
        for s in result["suggestions"]:
            self.assertNotIn("28 m/s", s, "Cross-machine contaminated parameter 28 m/s was accepted for WP-400!")
            self.assertNotIn("160 bar", s, "Cross-machine contaminated parameter 160 bar was accepted for WP-400!")

        # Since 2 suggestions were ungrounded hallucinations, evidence_status must NOT be "sufficient"
        self.assertIn(result["evidence_status"], [EVIDENCE_STATUS_PARTIAL, EVIDENCE_STATUS_INSUFFICIENT])
        self.assertTrue(len(result["unverified_details"]) > 0)

    def test_case_2_error_code_without_problem_explanation(self):
        """
        Case 2: A retrieved document contains an error code (E-0412), but does not explain
        or address the customer's reported mechanical problem (conveyor belt snapped).
        The model must not hallucinate that E-0412 solves the conveyor belt snap.
        """
        retrieved_context = [
            {
                "document": "Error Code E-0412: Thermal overload on spindle drive motor. Reset breaker Q4 on cabinet B.",
                "metadata": {
                    "source_file": "troubleshooting.pdf",
                    "document_reference": "TS-WP400-3.1",
                    "page": 6,
                    "section": "Spindle Thermal Overload"
                }
            }
        ]
        conversation_text = "Customer: The conveyor belt snapped in half and rubber debris is scattered everywhere."

        # Model tries to answer with E-0412 spindle thermal overload
        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "Conveyor belt snapped",
            "suggestions": [
                "Reset breaker Q4 to clear error E-0412 on the spindle motor.",
                "Inspect spindle motor thermal overload relay.",
                "Clear error E-0412 from the control panel.",
                "Restart the machine cycle."
            ],
            "suggestion_sources": [
                ["TS-WP400-3.1"],
                ["TS-WP400-3.1"],
                ["TS-WP400-3.1"],
                []
            ],
            "unverified_details": "",
            "safety": "",
            "confidence": 0.9,
            "is_grounded": True
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # The snapped conveyor belt is completely unaddressed by spindle error E-0412.
        # Evidence status must reflect insufficient evidence for the actual customer problem.
        self.assertEqual(result["evidence_status"], EVIDENCE_STATUS_INSUFFICIENT)
        self.assertEqual(result["suggestions"], [])

    def test_case_3_unrelated_valid_citation_misattribution(self):
        """
        Case 3: Gemini generates a suggestion about fan breaker Q4, but attaches an unrelated
        valid citation UM-WP400-1.1.3 (which is about dust extraction velocity, not breaker Q4).
        The validator must NOT accept UM-WP400-1.1.3 for breaker Q4.
        """
        retrieved_context = [
            {
                "document": "Dust extraction hood requires 20 m/s air velocity.",
                "metadata": {
                    "source_file": "user_manual.pdf",
                    "document_reference": "UM-WP400-1.1.3",
                    "page": 2,
                    "section": "Extraction Velocity"
                }
            },
            {
                "document": "Check fan circuit breaker Q4 located on electrical cabinet B.",
                "metadata": {
                    "source_file": "troubleshooting.pdf",
                    "document_reference": "TS-WP400-3.1",
                    "page": 6,
                    "section": "Electrical Cabinets"
                }
            }
        ]
        conversation_text = "Customer: The extraction fan tripped."

        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "Fan circuit breaker tripped",
            "suggestions": [
                "Inspect fan circuit breaker Q4 on control panel B.",
                "Verify extraction air velocity is at least 20 m/s.",
                "Inspect dust ducting for blockages.",
                "Check extraction motor terminal box."
            ],
            # Adversarial misattribution: Suggestion 1 (Breaker Q4) is misattributed to UM-WP400-1.1.3!
            "suggestion_sources": [
                ["UM-WP400-1.1.3"], # Misattribution! UM-WP400-1.1.3 does NOT mention breaker Q4!
                ["UM-WP400-1.1.3"],
                ["UM-WP400-1.1.3"],
                ["TS-WP400-3.1"]
            ],
            "unverified_details": "",
            "safety": "",
            "confidence": 0.95,
            "is_grounded": True
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # Check Suggestion 1 citations: UM-WP400-1.1.3 must NOT be attached to Breaker Q4 suggestion!
        sugg_1_sources = [c["document_reference"] for c in result["suggestion_sources"][0]]
        self.assertNotIn("UM-WP400-1.1.3", sugg_1_sources,
                         "Unrelated citation UM-WP400-1.1.3 was falsely attached to breaker Q4 suggestion!")

    def test_case_4_condition_mismatch_parameter(self):
        """
        Case 4: A numerical value exists in context, but for a DIFFERENT operating condition
        (e.g. 35 m/s during commissioning test vs 20 m/s during normal cutting operation).
        Customer asks about normal cutting operation.
        Suggestion claiming 35 m/s for normal operation must be rejected/flagged.
        """
        retrieved_context = [
            {
                "document": "During initial commissioning and test mode, test air velocity at 35 m/s. During normal cutting operation, maintain air velocity at minimum 20 m/s.",
                "metadata": {
                    "source_file": "dust_guide.pdf",
                    "document_reference": "UM-WP400-1.1.3",
                    "page": 2,
                    "section": "Dust Extraction"
                }
            }
        ]
        conversation_text = "Customer: What air velocity is required during normal cutting operations?"

        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "Operating air velocity requirement",
            "suggestions": [
                "Set air velocity to 35 m/s during normal cutting operations.", # Wrong condition! 35 m/s is for commissioning test mode!
                "Verify dust extraction hood is securely clamped.",
                "Inspect ducting joints for leaks.",
                "Confirm extraction fan is spinning in correct direction."
            ],
            "suggestion_sources": [
                ["UM-WP400-1.1.3"],
                ["UM-WP400-1.1.3"],
                ["UM-WP400-1.1.3"],
                []
            ],
            "unverified_details": "",
            "safety": "",
            "confidence": 0.9,
            "is_grounded": True
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # The suggestion with 35 m/s under normal cutting operations must NOT be certified as sufficient/grounded
        for s in result["suggestions"]:
            if "normal" in s.lower() and "cutting" in s.lower():
                self.assertNotIn("35 m/s", s, "35 m/s commissioning parameter was applied to normal cutting operations!")

    def test_case_5_contradictory_documentation_instructions(self):
        """
        Case 5: Retrieved documents contain contradictory instructions
        (Doc A says replace filter every 3 months; Doc B says replace filter every 6 months).
        System must not arbitrarily pick one without noting the discrepancy or must indicate partial/unverified.
        """
        retrieved_context = [
            {
                "document": "Filter cartridge must be replaced every 3 months under heavy dust load.",
                "metadata": {
                    "source_file": "quick_service.pdf",
                    "document_reference": "QS-WP400-1.0",
                    "page": 4,
                    "section": "Quick Service"
                }
            },
            {
                "document": "Filter cartridge replacement interval is every 6 months under standard factory conditions.",
                "metadata": {
                    "source_file": "maintenance_proc.pdf",
                    "document_reference": "MP-ALL-2.3",
                    "page": 8,
                    "section": "Preventive Maintenance"
                }
            }
        ]
        conversation_text = "Customer: When should I replace the dust filter cartridge?"

        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "partial",
            "likely_issue": "Filter cartridge replacement interval",
            "suggestions": [
                "Replace filter cartridge every 6 months under standard conditions, or every 3 months under heavy dust load.",
                "Inspect filter differential pressure indicator monthly."
            ],
            "suggestion_sources": [
                ["MP-ALL-2.3", "QS-WP400-1.0"],
                ["MP-ALL-2.3"]
            ],
            "unverified_details": "Documentation specifies both 3-month (heavy load) and 6-month (standard) intervals.",
            "safety": "Wear respiratory mask when changing filter.",
            "confidence": 0.85,
            "is_grounded": True
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # In partial evidence status, it must NOT force exactly 4 suggestions
        self.assertEqual(result["evidence_status"], EVIDENCE_STATUS_PARTIAL)
        self.assertEqual(len(result["suggestions"]), 2)
        self.assertTrue("3 months" in result["unverified_details"] or "heavy" in result["suggestions"][0])

    def test_case_6_partially_documented_multi_step_claim(self):
        """
        Case 6: Multi-step recommendation where one step is documented, but another step
        recommends an undocumented and dangerous action (e.g., bypass interlock switch).
        The unsupported step must be rejected, and evidence_status must be partial.
        """
        retrieved_context = [
            {
                "document": "Clean the extraction dust filter every 6 months using dry compressed air.",
                "metadata": {
                    "source_file": "maintenance.pdf",
                    "document_reference": "MP-ALL-2.3",
                    "page": 8,
                    "section": "Filter Cleaning"
                }
            }
        ]
        conversation_text = "Customer: The filter differential pressure alarm is active."

        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "sufficient", # Model falsely claimed sufficient!
            "likely_issue": "Filter differential pressure alarm",
            "suggestions": [
                "Clean the extraction dust filter using dry compressed air.", # Documented
                "Bypass the secondary safety door interlock switch S2.", # Dangerous undocumented action!
                "Inspect pneumatic line pressure.",
                "Contact technical support if alarm persists."
            ],
            "suggestion_sources": [
                ["MP-ALL-2.3"],
                [],
                [],
                []
            ],
            "unverified_details": "",
            "safety": "",
            "confidence": 0.9,
            "is_grounded": True
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # "Bypass the secondary safety door interlock" is dangerous and undocumented -> MUST be stripped!
        for s in result["suggestions"]:
            self.assertNotIn("bypass", s.lower(), "Dangerous undocumented bypass action was not stripped!")

        # Since an undocumented recommendation was rejected, evidence_status cannot be "sufficient"
        self.assertIn(result["evidence_status"], [EVIDENCE_STATUS_PARTIAL, EVIDENCE_STATUS_INSUFFICIENT])

    def test_case_7_out_of_domain_query_with_high_similarity(self):
        """
        Case 7: The customer asks a question outside the technical domain (e.g. recipe or bluetooth headphones),
        but the retrieval engine returns a machine chunk with artificial high similarity score (0.45).
        Relevance evaluation must detect zero entity/topic match and declare insufficient evidence.
        """
        retrieved_context = [
            {
                "document": "WP-400 table saw operating instructions: verify blade tilt angle and fence alignment.",
                "similarity_score": 0.45, # High similarity score artifact
                "metadata": {
                    "source_file": "wp400_manual.pdf",
                    "document_reference": "UM-WP400-1.1",
                    "page": 1,
                    "section": "Saw Alignment"
                }
            }
        ]
        conversation_text = "Customer: Can you give me the best recipe for baking chocolate chip cookies?"

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "has_issue": True,
            "evidence_status": "insufficient",
            "likely_issue": "",
            "suggestions": [],
            "suggestion_sources": [],
            "unverified_details": "No documentation on cookies.",
            "safety": "",
            "confidence": 0.0,
            "is_grounded": False
        })
        mock_client.models.generate_content.return_value = mock_resp

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # Must be recognized as completely irrelevant to baking cookies
        self.assertEqual(result["evidence_status"], EVIDENCE_STATUS_INSUFFICIENT)
        self.assertEqual(result["suggestions"], [])
        self.assertFalse(result["is_grounded"])

    def test_case_8_partial_answer_with_unsupported_technical_claim(self):
        """
        Case 8: A suggestion contains a valid general instruction but invents an unsupported
        part number (FC-99999) or parameter (12.5 kPa).
        The validator must reject the suggestion and record the unverified parameter.
        """
        retrieved_context = [
            {
                "document": "Replace filter cartridge when differential pressure exceeds 2.5 kPa.",
                "metadata": {
                    "source_file": "filter_guide.pdf",
                    "document_reference": "UM-WP400-1.1.3",
                    "page": 4,
                    "section": "Filter Differential Pressure"
                }
            }
        ]
        conversation_text = "Customer: What filter cartridge part number should I order and what is the differential limit?"

        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "sufficient",
            "likely_issue": "Filter cartridge replacement",
            "suggestions": [
                "Order replacement filter cartridge part FC-99999 immediately.", # Invented part FC-99999!
                "Replace filter cartridge when differential pressure exceeds 2.5 kPa.", # Grounded
                "Check filter differential pressure gauge on panel C.",
                "Verify air extraction velocity is 20 m/s." # Invented 20 m/s not in this doc!
            ],
            "suggestion_sources": [
                ["UM-WP400-1.1.3"],
                ["UM-WP400-1.1.3"],
                [],
                []
            ],
            "unverified_details": "",
            "safety": "",
            "confidence": 0.9,
            "is_grounded": True
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # FC-99999 must not appear in any suggestion
        for s in result["suggestions"]:
            self.assertNotIn("FC-99999", s)

        # Evidence status must be partial or insufficient, and unverified_details must mention FC-99999
        self.assertIn(result["evidence_status"], [EVIDENCE_STATUS_PARTIAL, EVIDENCE_STATUS_INSUFFICIENT])
        self.assertIn("FC-99999", result["unverified_details"])

    def test_case_9_generic_fallback_on_insufficient_evidence(self):
        """
        Case 9: When retrieved evidence is insufficient (e.g. laser diode failure on wood saw),
        the system must NEVER pad or return generic fallback suggestions (SAFE_SUPPLEMENTARY_STEPS).
        Suggestions array must be strictly empty ([]).
        """
        retrieved_context = [
            {
                "document": "Dust extraction hood requires 20 m/s air velocity.",
                "metadata": {
                    "source_file": "extraction.pdf",
                    "document_reference": "UM-WP400-1.1.3",
                    "page": 2,
                    "section": "Extraction Velocity"
                }
            }
        ]
        conversation_text = "Customer: The laser diode frequency calibration failed with error LD-900."

        mock_response_payload = {
            "has_issue": True,
            "evidence_status": "insufficient",
            "likely_issue": "Laser diode calibration failure LD-900",
            "suggestions": [],
            "suggestion_sources": [],
            "unverified_details": "No documentation on laser diode calibration or LD-900.",
            "safety": "",
            "confidence": 0.0,
            "is_grounded": False
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response_payload)
        )

        result = generate_support_suggestion(
            conversation_text=conversation_text,
            retrieved_context=retrieved_context,
            client=mock_client
        )

        # Must have empty suggestions, no fallbacks
        self.assertEqual(result["evidence_status"], EVIDENCE_STATUS_INSUFFICIENT)
        self.assertEqual(result["suggestions"], [])
        self.assertEqual(result["suggestion_sources"], [])
        self.assertFalse(result["is_grounded"])
        self.assertIn("couldn't find sufficient information", result["suggestion"].lower())


if __name__ == "__main__":
    unittest.main()

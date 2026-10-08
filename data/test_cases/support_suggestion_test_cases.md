# Support Suggestion Test Cases (RAG + Gemini)

Synthetic test data only. Cases are grounded in two fictional knowledge-base documents: the **Vantor Industrial** Sample Technical Documentation Pack (PDF) and the **Kestrel Machinery** Machine Support Knowledge Base (DOCX).

Distribution: 8 grounded, 2 cross-document, 2 no-context (12 total).

| ID | Machine | Customer Query | Expected Category | Expected Grounded | Expected Source Ref(s) | Topic |
|---|---|---|---|---|---|---|
| SS-01 | WP-400 (Vantor) | Our WP-400 panel saw just threw an E-206 and the blade was shaking badly right before it stopped. What should we do before restarting it? | Technical Troubleshooting | Yes | TS-WP400-3.1, RP-WP400-4.1 | WP-400 alarm E-206 (blade vibration/imbalance): stop, inspect blade and flanges, replace the blade, retighten the arbor nut |
| SS-02 | GC-120 (Vantor) | We're about to run a batch of 6 mm glass on our GC-120. What scoring pressure should we set, and how long can a sheet sit after scoring before we break it? | Machine Operation Issues | Yes | UM-GC120-1.4.1, UM-GC120-1.4.2 | GC-120 scoring pressure for 6 mm glass and the time limit before break-out |
| SS-03 | SC-900 (Vantor) | The water filter on our stone saw looks filthy. Which filter do we order for the SC-900, and how often should it be cleaned or replaced? | Maintenance & Parts | Yes | MP-ALL-2.3, MP-ALL-2.1 | SC-900 water filter element SC9-WF-02: part number, cleaning and replacement interval |
| SS-04 | MC-250 (Vantor) | The plasma on our MC-250 keeps going out in the middle of a cut. What should we look at? | Technical Troubleshooting | Yes | TS-MC250-3.3, FAQ-ALL-5 | MC-250 arc extinguishing mid-cut: ground clamp, rust or paint on the plate, cutting speed |
| SS-05 | WR-700 (Kestrel) | Our WR-700 won't start a job and the screen shows R-101. The table doesn't seem to be gripping the sheet properly. | Technical Troubleshooting | Yes | TS-WR700-3.1, OG-WR700-1.1.3 | WR-700 alarm R-101 (table vacuum low): cover unused zones, check gasket, replace vacuum pump filter |
| SS-06 | SJ-650 (Kestrel) | We keep cracking marble right at the start of the cut on our SJ-650. How should we be starting the cut? | Machine Operation Issues | Yes | OG-SJ650-1.2.2, TS-SJ650-3.2, FAQ-ALL-5 | SJ-650 low-pressure pierce to avoid cracking stone at the start of a cut |
| SS-07 | MF-180 (Kestrel) | Our MF-180 is showing L-05 and it won't pierce the plate properly. What's wrong and how do we fix it? | Technical Troubleshooting | Yes | TS-MF180-3.3, RP-MF180-4.3 | MF-180 alarm L-05 (protective window contaminated) and the window replacement procedure |
| SS-08 | GL-300 (Kestrel) | How long should a scoring wheel last on our GL-300, and which one do I order when it's worn? | Maintenance & Parts | Yes | MP-ALL-2.3, MP-ALL-2.1, FAQ-ALL-5 | GL-300 carbide scoring wheel GL3-SW-8: part number and expected life |
| SS-09 | GC-120 / GL-300 | We run a GC-120 and a GL-300 in the same shop and both are throwing low-vacuum alarms today. What should we check on each machine? | Technical Troubleshooting | Yes | TS-GC120-3.4, TS-GL300-3.4 | Low vacuum alarms on two glass machines: G-03 (GC-120, Vantor) and H-07 (GL-300, Kestrel) |
| SS-10 | MC-250 / MF-180 | Our new operator is learning on the MC-250 plasma table and the MF-180 laser. What needs to be checked on each one before the first cut of the day? | Machine Operation Issues | Yes | UM-MC250-1.3.1, OG-MF180-1.3.1 | Startup checks for the MC-250 plasma table (Vantor) and the MF-180 fibre laser (Kestrel) |
| SS-11 | WR-700 (Kestrel) | What's the part number for the spindle bearing on our WR-700, and how often should it be changed? | Maintenance & Parts | No | none (insufficient documentation) | WR-700 spindle bearing part number and interval (not documented); a transparent 'documentation does not provide sufficient information' response is expected |
| SS-12 | Press brake (not covered) | Our press brake keeps losing hydraulic pressure halfway through a bend. What should we check? | Technical Troubleshooting | No | none (insufficient documentation) | Press brake hydraulic pressure loss (equipment and hydraulic systems are not documented); a transparent 'documentation does not provide sufficient information' response is expected |

## Notes for the tester

- **Doc refs are not unique across the two documents.** `MP-ALL-2.1`, `MP-ALL-2.2`, `MP-ALL-2.3`, `MP-ALL-2.4` and `FAQ-ALL-5` exist in both. Score source references as (document, ref) pairs, and make sure chunk IDs in the vector store include the document name, otherwise one document's chunks can overwrite the other's.
- The machine column shows the document in brackets for single-document cases.
- Refs listed after the first one are supporting sections that also contain the answer; a correct system should cite at least the primary ref.

## Expected System Behavior

**A grounded response should:**
- Use only facts from the retrieved chunks (alarm meanings, causes, actions, values, part numbers, intervals).
- Give 2-3 clear, actionable suggestions in a sensible order.
- Cite the correct source reference(s), and the correct document where refs overlap.
- Keep machine-specific facts with the right machine and not mix numbers between similar machines.

**A no-context response should:**
- State plainly that the available documentation does not provide sufficient information.
- Not invent causes, steps, part numbers or values, and not present a related-looking entry from another machine or system as the answer.
- Cite no source reference.
- Optionally suggest escalating to a service engineer or checking the manufacturer's documentation.

**What counts as hallucination:**
- Any alarm code, part number, value, step, interval or procedure that is not in the retrieved text.
- A real fact attached to the wrong machine (for example a Vantor part number given for a Kestrel machine).
- A citation to a source that does not contain the claim, or to a source that does not exist.
- Answering a no-context question with a confident but unsupported answer.

**Why source references matter:**
- They let the support agent verify a suggestion before relaying it to a customer.
- They make retrieval testable: you can check that the right chunk was retrieved and cited.
- They expose hallucination: a claim with no matching source is unsupported.

# Support Suggestion Dataset Evaluation Results

- **Execution Timestamp:** 2026-10-07T05:58:12Z
- **Active LLM Generation Model:** `gemini-3.5-flash-lite`
- **Embedding Model:** `gemini-embedding-2` (768 dimensions)

## Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Total Cases** | 12 |
| **Passed** | **12** |
| **Failed** | 0 |
| **Single-Document Grounded Passed** | 8/8 |
| **Cross-Document Grounded Passed** | 2/2 |
| **No-Context Safety Passed** | 2/2 |
| **Grounding Accuracy** | 100.0% |
| **Source Reference Match Rate** | 100.0% |
| **No-Context Safety Rate** | 100.0% |
| **Average Gemini Generation Latency** | 2649.49 ms |
| **Average Total Pipeline Latency** | 3404.79 ms |
| **429 Rate Limit Errors** | 0 |

## Detailed Case Results

### SS-01: WP-400 — ✅ PASS
- **Query:** *"Our WP-400 panel saw just threw an E-206 and the blade was shaking badly right before it stopped. What should we do before restarting it?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['RP-WP400-4.1', 'TS-WP400-3.1']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'RP-WP400-4.1', 'TS-WP400-3.1', 'UM-WP400-1.1.1', 'UM-WP400-1.1.3']`
- **Returned Sources:** `['FAQ-ALL-5', 'RP-WP400-4.1', 'TS-WP400-3.1', 'UM-WP400-1.1.1', 'UM-WP400-1.1.3']`
- **Confidence:** `1.0`
- **Latency:** Generation: `7781.13 ms` | Total: `8730.7 ms`
- **Generated Suggestion:**
  > Safety Warning: Switch off and lock out the machine, and wait 2 minutes until the spindle has fully stopped before performing any inspection or maintenance. Never bypass safety interlocks.

Likely cause for error E-206 (Blade vibration / imbalance above 4.5 mm/s): Damaged blade, dirty flange, or loose arbor nut.

Resolution steps:
1. Stop at once and ensure the machine is switched off, locked out, and the spindle has stopped.
2. Open the blade guard and insert the spindle lock pin.
3. Inspect the blade and flanges for damage or debris.
4. If replacing the blade (WP4-BL-300), remove the arbor nut (note: it has a left-hand thread), remove the outer flange and old blade, and clean both flanges and the arbor.
5. Fit the new blade with the rotation arrow matching the spindle direction.
6. Refit the outer flange and tighten the arbor nut to 60 Nm.
7. Remove the lock pin, close the guard, remove the lock-out, run at 1000 rpm for 30 seconds, then at full speed.
8. Reset the blade-hours counter. If E-206 returns, re-inspect the flanges for damage or debris.

### SS-02: GC-120 — ✅ PASS
- **Query:** *"We're about to run a batch of 6 mm glass on our GC-120. What scoring pressure should we set, and how long can a sheet sit after scoring before we break it?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['UM-GC120-1.4.1', 'UM-GC120-1.4.2']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'OG-GL300-1.4.1', 'TS-GC120-3.4', 'UM-GC120-1.4.1', 'UM-GC120-1.4.2']`
- **Returned Sources:** `['FAQ-ALL-5', 'OG-GL300-1.4.1', 'TS-GC120-3.4', 'UM-GC120-1.4.1', 'UM-GC120-1.4.2']`
- **Confidence:** `1.0`
- **Latency:** Generation: `3596.51 ms` | Total: `4187.35 ms`
- **Generated Suggestion:**
  > For the GC-120 running 6 mm glass, set the scoring pressure to 3.0 bar. Break the sheet within 30 seconds of scoring, as the score line loses effect as the glass relaxes.

### SS-03: SC-900 — ✅ PASS
- **Query:** *"The water filter on our stone saw looks filthy. Which filter do we order for the SC-900, and how often should it be cleaned or replaced?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['MP-ALL-2.1', 'MP-ALL-2.3']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'MP-ALL-2.3', 'RP-SC900-4.2', 'TS-SC900-3.2']`
- **Returned Sources:** `['FAQ-ALL-5', 'MP-ALL-2.3', 'RP-SC900-4.2', 'TS-SC900-3.2']`
- **Confidence:** `1.0`
- **Latency:** Generation: `2453.41 ms` | Total: `3074.26 ms`
- **Generated Suggestion:**
  > For the SC-900 stone saw, order part number SC9-WF-02 (Water filter element). According to the documentation, clean the SC9-WF-02 weekly and replace it upon alarm F-01 or monthly.

### SS-04: MC-250 — ✅ PASS
- **Query:** *"The plasma on our MC-250 keeps going out in the middle of a cut. What should we look at?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['FAQ-ALL-5', 'TS-MC250-3.3']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'RP-MC250-4.3', 'TS-MC250-3.3', 'UM-MC250-1.3.1']`
- **Returned Sources:** `['FAQ-ALL-5', 'RP-MC250-4.3', 'TS-MC250-3.3', 'UM-MC250-1.3.1']`
- **Confidence:** `0.95`
- **Latency:** Generation: `2660.82 ms` | Total: `3335.08 ms`
- **Generated Suggestion:**
  > For maintenance, blade replacement, consumable change, or mechanical troubleshooting, always ensure safety procedures are followed (switch off, lock out the isolator, and wait for parts to stop). 

Likely causes for the plasma arc going out mid-cut include a poor ground connection, rust or paint on the plate, or the cut speed being set too high.

Resolution steps:
1. Check the ground clamp and clamp it directly to clean bare metal on the plate, ensuring there is no rust or paint preventing a stable arc.
2. Reduce the cut speed.
3. If replacing consumables during troubleshooting, switch off the power source and disconnect it from the supply, and shut off the air before servicing.

### SS-05: WR-700 — ✅ PASS
- **Query:** *"Our WR-700 won't start a job and the screen shows R-101. The table doesn't seem to be gripping the sheet properly."*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['OG-WR700-1.1.3', 'TS-WR700-3.1']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'OG-WR700-1.1.1', 'OG-WR700-1.1.3', 'RP-WR700-4.5', 'TS-WR700-3.1']`
- **Returned Sources:** `['FAQ-ALL-5', 'OG-WR700-1.1.1', 'OG-WR700-1.1.3', 'RP-WR700-4.5', 'TS-WR700-3.1']`
- **Confidence:** `1.0`
- **Latency:** Generation: `2007.7 ms` | Total: `2703.09 ms`
- **Generated Suggestion:**
  > For any maintenance or troubleshooting, ensure you follow required safety procedures, such as switching off and locking out the isolator before performing checks. 

Alarm R-101 indicates table vacuum is low (above -0.55 bar). The likely causes include open zones, a worn gasket, a dirty pump filter, or small parts. 

Resolution steps:
1. Cover every unused vacuum zone with a blanking sheet.
2. Check and replace the gasket if worn.
3. Replace the pump filter (part WR7-VF-20).
4. If cutting parts smaller than 150 mm x 150 mm, use a gasket or clamps in addition to the vacuum, as parts below this size cannot rely on vacuum alone. 

Note: The controller blocks job start when vacuum is above -0.55 bar. Never reduce the vacuum threshold to start a job. The alarm clears automatically below -0.60 bar.

### SS-06: SJ-650 — ✅ PASS
- **Query:** *"We keep cracking marble right at the start of the cut on our SJ-650. How should we be starting the cut?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['FAQ-ALL-5', 'OG-SJ650-1.2.2', 'TS-SJ650-3.2']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'OG-SJ650-1.2.1', 'OG-SJ650-1.2.2', 'UM-SC900-1.2.1', 'UM-SC900-1.2.2']`
- **Returned Sources:** `['FAQ-ALL-5', 'OG-SJ650-1.2.1', 'OG-SJ650-1.2.2', 'UM-SC900-1.2.1', 'UM-SC900-1.2.2']`
- **Confidence:** `1.0`
- **Latency:** Generation: `1423.57 ms` | Total: `2251.99 ms`
- **Generated Suggestion:**
  > To prevent cracking your marble when the cut starts on the SJ-650, do not pierce at full pressure. Instead, pierce at a low pressure of 1500 bar for 3 seconds, and then ramp to full pressure. (Reference: FAQ-ALL-5 / Section 1.2.2).

### SS-07: MF-180 — ✅ PASS
- **Query:** *"Our MF-180 is showing L-05 and it won't pierce the plate properly. What's wrong and how do we fix it?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['RP-MF180-4.3', 'TS-MF180-3.3']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'TS-MC250-3.3', 'TS-MF180-3.3']`
- **Returned Sources:** `['FAQ-ALL-5', 'TS-MC250-3.3', 'TS-MF180-3.3']`
- **Confidence:** `1.0`
- **Latency:** Generation: `2432.97 ms` | Total: `3174.16 ms`
- **Generated Suggestion:**
  > Safety reminder: Before performing any maintenance or replacement, ensure you switch off and lock out the isolator, and wait for parts to stop.

Likely cause: Alarm code L-05 indicates the protective window is contaminated (spatter on the window, burnt-on dirt), which also causes the issue of the laser not piercing the plate properly.

Resolution steps:
1. Replace the protective window using part number MF1-PW-27 (refer to Section 4.3).
2. If the laser still does not pierce the plate after replacing the window, recalibrate the focus position and clean any rusty or oily surface on the plate.

### SS-08: GL-300 — ✅ PASS
- **Query:** *"How long should a scoring wheel last on our GL-300, and which one do I order when it's worn?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['FAQ-ALL-5', 'MP-ALL-2.1', 'MP-ALL-2.3']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'RP-GC120-4.4', 'UM-GC120-1.4.2']`
- **Returned Sources:** `['FAQ-ALL-5', 'RP-GC120-4.4', 'UM-GC120-1.4.2']`
- **Confidence:** `1.0`
- **Latency:** Generation: `1224.79 ms` | Total: `2250.2 ms`
- **Generated Suggestion:**
  > On the GL-300, a scoring wheel lasts about 8,000 m of scoring, or until the edge quality drops. When it is worn, order part number GL3-SW-8.

### SS-09: GC-120 / GL-300 — ✅ PASS
- **Query:** *"We run a GC-120 and a GL-300 in the same shop and both are throwing low-vacuum alarms today. What should we check on each machine?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['TS-GC120-3.4', 'TS-GL300-3.4']`
- **Retrieved Sources:** `['OG-GL300-1.4.1', 'OG-WR700-1.1.3', 'TS-GC120-3.4', 'TS-GL300-3.4', 'TS-MF180-3.3']`
- **Returned Sources:** `['OG-GL300-1.4.1', 'OG-WR700-1.1.3', 'TS-GC120-3.4', 'TS-GL300-3.4', 'TS-MF180-3.3']`
- **Confidence:** `1.0`
- **Latency:** Generation: `1737.09 ms` | Total: `2458.85 ms`
- **Generated Suggestion:**
  > SAFETY: Before any maintenance or consumable change, switch the machine off, lock out the main isolator per your site lock-out/tag-out procedure, and wait until all moving parts have stopped.

For the GC-120 (Alarm G-03: Vacuum pressure low):
1. Likely cause: Worn seal strip, sheet not fully seated, or pump filter dirty.
2. Action: Reseat the sheet, clean the pump filter, and replace part GC1-VB-02 if needed.

For the GL-300 (Alarm H-07: Vacuum low, above -0.5 bar):
1. Likely cause: Sheet not seated, dirty filter, or worn seal.
2. Action: Reseat the sheet, clean the pump filter, and check the seal.

### SS-10: MC-250 / MF-180 — ✅ PASS
- **Query:** *"Our new operator is learning on the MC-250 plasma table and the MF-180 laser. What needs to be checked on each one before the first cut of the day?"*
- **Expected Grounded:** `True` | **Actual Grounded:** `True`
- **Expected Sources:** `['OG-MF180-1.3.1', 'UM-MC250-1.3.1']`
- **Retrieved Sources:** `['FAQ-ALL-5', 'OG-MF180-1.3.1', 'TS-MC250-3.3', 'UM-MC250-1.3.1', 'UM-MC250-1.3.2']`
- **Returned Sources:** `['FAQ-ALL-5', 'OG-MF180-1.3.1', 'TS-MC250-3.3', 'UM-MC250-1.3.1', 'UM-MC250-1.3.2']`
- **Confidence:** `1.0`
- **Latency:** Generation: `2035.19 ms` | Total: `2753.07 ms`
- **Generated Suggestion:**
  > Before starting your first cut of the day, please ensure safety procedures are followed and check the following for each machine:

**MC-250 CNC Plasma Cutting Table:**
1. Ensure the plasma supply has clean, dry compressed air at 6.2 bar (90 psi) and drain the filter bowl daily.
2. Attach the work (ground) clamp directly to clean bare metal on the plate (paint and rust prevent a stable arc).
3. Switch on the power source and wait for the READY lamp.
4. Home the torch height controller (THC).

**MF-180 Fibre Laser Cutting Machine:**
1. Check the chiller: coolant temperature must be 22 °C and the level above MIN (the laser will not start if the coolant is above 28 °C).
2. Open the assist-gas valve (oxygen for mild steel, nitrogen for stainless steel and aluminium; minimum supply pressure 16 bar for nitrogen).
3. Switch on the machine and wait for the READY lamp on the laser source (about 90 seconds).
4. Run nozzle centring and the capacitive height-sensor calibration.
5. Load the sheet on the pallet and make a test cut on an offcut before production.

### SS-11: WR-700 — ✅ PASS
- **Query:** *"What's the part number for the spindle bearing on our WR-700, and how often should it be changed?"*
- **Expected Grounded:** `False` | **Actual Grounded:** `False`
- **Expected Sources:** `[]`
- **Retrieved Sources:** `['FAQ-ALL-5', 'MP-ALL-2.1', 'MP-ALL-2.3', 'RP-WR700-4.1']`
- **Returned Sources:** `['FAQ-ALL-5', 'MP-ALL-2.1', 'MP-ALL-2.3', 'RP-WR700-4.1']`
- **Confidence:** `1.0`
- **Latency:** Generation: `1491.6 ms` | Total: `2220.05 ms`
- **Generated Suggestion:**
  > The available documentation does not provide sufficient information regarding the part number or replacement interval for the spindle bearing on the WR-700.

### SS-12: Press brake (not covered) — ✅ PASS
- **Query:** *"Our press brake keeps losing hydraulic pressure halfway through a bend. What should we check?"*
- **Expected Grounded:** `False` | **Actual Grounded:** `False`
- **Expected Sources:** `[]`
- **Retrieved Sources:** `['FAQ-ALL-5', 'TS-MC250-3.3', 'UM-GC120-1.4.2', 'UM-MC250-1.3.1', 'UM-WP400-1.1.1']`
- **Returned Sources:** `['FAQ-ALL-5', 'TS-MC250-3.3', 'UM-GC120-1.4.2', 'UM-MC250-1.3.1', 'UM-WP400-1.1.1']`
- **Confidence:** `1.0`
- **Latency:** Generation: `2949.08 ms` | Total: `3718.67 ms`
- **Generated Suggestion:**
  > The available documentation does not provide sufficient information regarding press brakes or hydraulic pressure issues.

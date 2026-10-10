# Demonstration Script: AI-Powered Customer Support Assistant
## 3–5 Minute Live Walkthrough (Scenario: WP-400 Error E-102)

This demonstration script provides a step-by-step walkthrough of the AI-Powered Customer Support Assistant during a 3–5 minute presentation or evaluation. It uses the standard industrial troubleshooting scenario verified during testing (**WP-400 CNC Panel Saw — Error E-102 & Dust Extraction Interlock**).

> [!NOTE]
> For a comprehensive, step-by-step manual testing script designed for external evaluators testing a deployed or hosted application (including authentication flows, negative edge cases, and an evaluator results checklist), see [docs/EVALUATOR_TEST_SCRIPT.md](file:///Users/boomika/AI-Powered%20Customer%20Support%20Assistant/docs/EVALUATOR_TEST_SCRIPT.md).

---

## Demo Overview & Goal

**Scenario:** A factory operator calls technical support because their WP-400 CNC panel saw shut down with fault code `E-102` and the dust extraction fan is inactive.

**Pipeline Flow Demonstrated:**
```
Microphone Capture (sounddevice)
      ↓
AssemblyAI Streaming Transcription
      ↓
Semantic Role Classification (CUSTOMER vs. AGENT)
      ↓
Customer Sentiment Analysis (Negative / Agitated)
      ↓
Query Categorization (Technical Troubleshooting)
      ↓
ChromaDB Vector Retrieval (Gemini Embedding 2)
      ↓
Gemini 3.5 Flash-Lite Suggestion Synthesis
      ↓
Strict Grounding & Industrial Safety Warning
      ↓
Knowledge Base Source Provenance
      ↓
Clean Session Termination
```

---

## Pre-Demo Checklist (1 Minute Before Presentation)

1. **Verify Backend is Running:**
   ```bash
   uvicorn src.main:app --port 5000
   ```
2. **Open Dashboard:**
   Navigate to `http://localhost:5000/dashboard` in Google Chrome or another browser.
3. **Verify Subsystems Bar:**
   - Status: **Connected** (green beacon)
   - Microphone: **Ready**
   - AssemblyAI: **Ready**
   - Speaker Classification: **Ready**
   - Gemini 3.5 Flash-Lite: **Ready**
   - Knowledge Base (ChromaDB): **Indexed** (68 chunks)

---

## Demonstration Script

### STEP 1: Introduction & Starting the Session (0:00 – 0:45)

**Feature Demonstrated:** System Initialization, Zero Disk Audio Capture, WebSocket Connection.

**Presenter Action (What to Click):**
- Click the green **"Start Session"** button in the top-right header.

**Presenter Narration (What to Say):**
> *"Welcome everyone. Today we are demonstrating the AI-Powered Customer Support Assistant—a real-time AI copilot designed for industrial equipment support agents.*
>
> *I have just clicked 'Start Session'. Notice the header telemetry: the session timer is ticking, our WebSocket connection is live, and audio capture has initialized through sounddevice at 16 kHz mono. Crucially, audio frames stream entirely in-memory—zero voice audio is written to disk, ensuring strict compliance and privacy.*
>
> *Let's simulate a live support call from a machine operator on a factory floor."*

**Dashboard Result (What Should Appear):**
- Status button switches from green "Start Session" to disabled, while red **"Stop Session"** becomes active.
- Session timer starts (`00:00:01`, `00:00:02`...).
- Subsystem strip: Microphone dot turns green (**Active**), Stream Indicator shows **LISTENING**.

---

### STEP 2: Customer Problem Statement (0:45 – 1:30)

**Features Demonstrated:** Real-Time Transcription, Customer Role Classification, Sentiment Detection, Query Categorization.

**Presenter Action (What to Say into the Microphone):**
> *(Speak clearly into the microphone in a concerned, slightly frustrated tone):*
>
> **"Hello, our WP-400 panel saw just shut down in the middle of a production run. The control panel is displaying error E-102 and the dust extraction fan is completely off. We need to get this running immediately."**

**Presenter Narration (What to Point Out to the Audience):**
> *"Notice three things happening simultaneously on the dashboard:*
>
> *1. **Real-Time Streaming STT:** As I spoke, AssemblyAI streamed partial word tokens directly into the feed under the 'LISTENING...' indicator before finalizing the turn.*
> *2. **Semantic Role Classification:** The system did not use hardcoded speaker assumptions. Gemini 3.5 Flash-Lite analyzed the linguistic discourse and accurately assigned the badge **CUSTOMER** with high confidence.*
> *3. **Emotional Tone & Categorization:** The Customer Sentiment panel immediately flagged **Negative** (or **Agitated**), noting the customer's machine breakdown and downtime. Simultaneously, the Query Category was classified as **Technical Troubleshooting**."*

**Dashboard Result (What Should Appear):**
- **Live Conversation Feed:** Displays the utterance with an amber **CUSTOMER** badge, confidence (e.g. `95%`), and timestamp.
- **Customer Sentiment Card:** Displays **Negative** (or **Agitated**) with a red/amber pill and a rationale explaining the operator's production stoppage.
- **Query Category Card:** Displays **Technical Troubleshooting** with high confidence.

---

### STEP 3: Automated RAG Retrieval & Grounded AI Suggestions (1:30 – 2:30)

**Features Demonstrated:** ChromaDB Semantic Retrieval (Gemini Embedding 2), Grounded Support Suggestion Synthesis, Industrial Safety Warnings.

**Presenter Narration (What to Say):**
> *"Because this turn was verified as a CUSTOMER problem, the backend automatically triggered our parallel RAG engine.*
>
> *In the background, Gemini Embedding 2 generated a 768-dimensional query vector and searched our local ChromaDB collection of sample technical manuals in under 800 milliseconds. Then, Gemini 3.5 Flash-Lite synthesized this comprehensive support card on the right:*
>
> *Look at the **AI Support Suggestions** card:*
> - *The **Likely Issue** accurately identifies a differential pressure fault across the extraction cartridge or a tripped fan motor breaker.*
> - *Under **Recommended Actions**, the assistant provides clear step-by-step instructions:*
>   1. *Check the dust extraction fan circuit breaker / thermal overload.*
>   2. *Inspect the `WP4-FL-DE1` cartridge filter for blockages.*
>   3. *Clear the alarm and restart.*
> - *Notice the **Safety Banner** highlighted in amber at the bottom: It explicitly reminds the agent to tell the operator to turn off the main isolator and follow Lock-Out / Tag-Out (LOTO) procedures before opening any filter compartments.*
> - *And notice the green badge at the top: **Grounded in Documentation**. The assistant did not hallucinate an error code or procedure—it is strictly bound to the manual."*

**Dashboard Result (What Should Appear):**
- **AI Support Suggestions Card:**
  - Grounding Pill: **Grounded in Documentation** (green).
  - Likely Issue: Diagnostic explanation referencing error E-102 and dust extraction.
  - Recommended Actions: 2–3 structured action cards listing inspection steps for breaker and filter cartridge `WP4-FL-DE1`.
  - Safety Warning Box: Prominent warning reminding operator to lock out the isolator before servicing.

---

### STEP 4: Inspecting Knowledge Base Sources (2:30 – 3:15)

**Feature Demonstrated:** Provenance Attribution, Document Reference & Page Metadata.

**Presenter Action (What to Click / Point To):**
- Direct the audience's attention to the **Knowledge Base Sources** card beneath the suggestions.

**Presenter Narration (What to Say):**
> *"In a mission-critical industrial setting, an agent cannot simply trust an opaque AI output. They need verifiable provenance.*
>
> *Here in the **Knowledge Base Sources** panel, every single chunk retrieved from ChromaDB is attributed with complete metadata:*
> - *Source Document:* `Sample_Technical_Documentation_Pack_SYNTHETIC.pdf`
> - *Page Reference:* `Page 8`
> - *Document Reference Code:* `TS-WP400-3.1`
> - *Section:* `Section 3.1 Error Codes and Alarms`
>
> *The agent can cite these exact section numbers directly to the customer or open the manual knowing exactly where the procedure is located."*

**Dashboard Result (What Should Appear):**
- **Knowledge Base Sources:** Displays citation cards showing source document filename, page number, document reference `TS-WP400-3.1`, and section title.

---

### STEP 5: Agent Turn & Speaker Role Disambiguation (3:15 – 3:50)

**Feature Demonstrated:** Agent Role Classification, Non-Triggering of RAG on Agent Speech.

**Presenter Action (What to Say into the Microphone):**
> *(Speak calmly in an authoritative support agent tone):*
>
> **"Thank you for reporting that. Please do not open the dust cabinet yet. First, make sure the main power isolator is locked out, and check if the thermal breaker on the extraction unit has tripped."**

**Presenter Narration (What to Point Out):**
> *"Now observe the Live Conversation panel.*
>
> *The speech transcription finalized, and the classifier recognized the supportive, instructional phrasing and labeled this turn with a blue **AGENT** badge.*
>
> *Because this was an AGENT turn, the assistant logged the dialogue to maintain full conversation context, but deliberately did **not** re-run an unnecessary vector search or overwrite the customer's troubleshooting suggestions. The suggestion cards remain stable on the screen."*

**Dashboard Result (What Should Appear):**
- **Live Conversation Feed:** Appends the new turn with a blue **AGENT** badge.
- **AI Suggestions & Sources:** Remain steady on screen without unnecessary flickering or duplicate LLM calls.

---

### STEP 6: Stopping the Session & Conclusion (3:50 – 4:30)

**Features Demonstrated:** Clean Shutdown, Graceful Resource Release, Session Persistence.

**Presenter Action (What to Click):**
- Click the red **"Stop Session"** button in the header.

**Presenter Narration (What to Say):**
> *"To conclude the call, I click 'Stop Session'.*
>
> *The microphone hardware stream closes immediately, the AssemblyAI WebSocket disconnects cleanly, and the session timer halts.*
>
> *Notice that all transcript entries, sentiment analytics, and recommendation cards remain fully visible on the screen so the support agent can wrap up their call documentation and log CRM notes without losing any data.*
>
> *In summary, within less than four minutes, we demonstrated:*
> 1. *Zero-disk microphone capture and live streaming speech-to-text.*
> 2. *Semantic speaker-role classification separating Customer from Agent.*
> 3. *Customer sentiment and query categorization.*
> 4. *ChromaDB vector search using Gemini Embedding 2.*
> 5. *Strictly grounded troubleshooting recommendations with safety protocols and complete manual provenance.*
>
> *Thank you. I am happy to take any questions."*

**Dashboard Result (What Should Appear):**
- Subsystem indicators return to **Ready**.
- Session timer pauses.
- "Start Session" button re-enables; "Stop Session" button disables.
- Entire conversation feed, sentiment card, categorization badge, suggestions, and source citations remain intact on the dashboard.

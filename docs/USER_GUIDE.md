# Support Agent User Guide
## AI-Powered Customer Support Assistant

Welcome to the **AI-Powered Customer Support Assistant**. This guide is designed for technical support agents who assist customers troubleshooting industrial machinery. It explains how to launch and operate your assistant during live calls to quickly find verified procedural recommendations, safety guidelines, and manual citations.

---

## 1. Starting the Application

Before taking support calls, launch the assistant backend on your computer:

1. Open your terminal application.
2. Navigate to your assistant folder and activate the environment:
   ```bash
   source venv/bin/activate
   ```
   *(On Windows: `venv\Scripts\activate`)*
3. Start the application:
   ```bash
   uvicorn src.main:app --port 5000
   ```
4. The service will indicate that it is running on `http://127.0.0.1:5000`. Keep this terminal window open during your shift.

---

## 2. Opening the Dashboard

1. Open your web browser (Google Chrome, Microsoft Edge, Safari, or Mozilla Firefox).
2. Enter the following URL into your address bar:
   ```
   http://localhost:5000/dashboard
   ```
3. The dark-themed **AI Support Assistant** dashboard will load.
4. Check the **Status** pill in the top header. It will show a green dot and read **Connected**.
5. Check the status bar under the header to confirm all subsystems (Microphone, AssemblyAI, Speaker Classification, Gemini 3.5 Flash-Lite, and Knowledge Base) are in **Ready** or **Indexed** state.

---

## 3. Starting a Support Session

When you are ready to answer an incoming customer call:

1. Put on your headset with microphone.
2. In the top right corner of the dashboard, click the green **"Start Session"** button.
3. The button will switch to an active state, the session timer will begin counting (`00:00:01`), and the microphone indicator will show **Listening**.
4. The assistant is now listening to your conversation in real time. All audio is processed in volatile memory—**no audio files are saved to disk**.

---

## 4. Live Conversation Display

The left half of your screen displays the **Live Conversation** feed:

- **Listening Preview:** As either you or the customer speaks, interim transcribed words will stream in real time under a pulsing **"LISTENING..."** badge.
- **Auto-Scroll:** New dialogue entries automatically scroll into view. You can toggle the **Auto-scroll** switch in the panel header if you wish to freeze the view and read earlier statements.
- **Turn Timestamps:** Each turn displays an accurate timestamp so you can follow the conversation timeline.

---

## 5. CUSTOMER vs AGENT Turns

The assistant automatically analyzes the dialogue to distinguish between you and the customer:

- **CUSTOMER Turns (Amber Badge):**
  - Statements made by the customer explaining faults, machine models, symptoms, or questions.
  - Identified with a `CUSTOMER` badge and a confidence percentage (e.g., `95%`).
  - **Important:** Customer turns immediately trigger the assistant's intelligence engine to analyze the problem and retrieve technical recommendations.
- **AGENT Turns (Blue Badge):**
  - Statements spoken by you (the support agent) acknowledging the customer or giving instructions.
  - Labeled with an `AGENT` badge.
  - Agent turns are recorded on the transcript to maintain full dialogue context, but they do **not** trigger redundant documentation searches.
- **UNKNOWN Turns (Gray Badge):**
  - Brief greetings, background chatter, or unclear audio where the speaker role cannot be confirmed with high confidence (above 80%).

---

## 6. Sentiment Indicator

Located in the top-right panel under **Customer Sentiment**:

- Tracks the emotional state and stress level of the customer based on their latest statements.
- **Sentiment Labels:**
  - 🟢 **Positive:** The customer is pleased, expressing gratitude, or confirming success.
  - ⚪ **Neutral:** The customer is calm, matter-of-fact, and sharing information.
  - 🟡 **Negative:** The customer is reporting an equipment breakdown, delay, or frustration.
  - 🔴 **Agitated:** The customer is experiencing urgent downtime, expressing sharp anger, or requesting escalation.
- **Confidence & Rationale:** Beneath the badge, a brief note explains why this sentiment was detected (e.g., *"Customer reporting machine breakdown with stopped fan"*), helping you adjust your communication tone.

---

## 7. Query Category

Located next to the Sentiment card under **Query Category**:

- Automatically tags the inquiry into one of three standard support categories:
  1. ⚙️ **Machine Operation Issues:** Startup sequences, controls, homing, feed rates, spindle adjustments.
  2. 🔧 **Maintenance & Parts:** Filter replacements, lubrication intervals, consumable wear, spare part ordering.
  3. ⚠️ **Technical Troubleshooting:** Fault codes (e.g., E-102), electrical trips, motor stalls, sensor malfunctions.
- Use this category badge to log support tickets accurately in your ticketing software without manual tagging.

---

## 8. AI Support Suggestions

The large center card in the right column—**AI Support Suggestions**—is your primary troubleshooting assistant:

- **Likely Issue:** Summarizes the probable mechanical or electrical root cause (e.g., *"Differential pressure threshold exceeded or fan thermal trip"*).
- **Recommended Actions:** Delivers 2 to 3 numbered, actionable steps to guide the customer through troubleshooting.
- **Safety Warning Box:** Highlighted in amber/red at the bottom of the card, reminding you of mandatory safety protocols (such as turning off the main isolator, applying Lock-Out / Tag-Out (LOTO), or waiting for rotating blades to stop).

---

## 9. Knowledge Base Sources

Directly beneath the suggestions card, the **Knowledge Base Sources** section displays the exact documents used:

- Shows the specific manual name (e.g., `Sample_Technical_Documentation_Pack_SYNTHETIC.pdf`).
- Displays the **Page Number** (e.g., `Page 8`).
- Lists the official **Document Reference** (e.g., `TS-WP400-3.1`) and **Section Title** (e.g., `Section 3.1 Error Codes and Alarms`).
- You can quote these section references directly to the customer or consult the manual on your desk if deeper schematics are required.

---

## 10. Grounded Recommendations

What does **Grounded** mean?

- The assistant is engineered with a strict **grounding-only rule**.
- It is prohibited from guessing, hallucinating, or making up fault numbers, bolt torques, or wiring diagrams.
- Every instruction shown on your screen comes directly from verified technical documentation in the local database.
- The badge at the top right of the suggestions card will display a green **"Grounded in Documentation"** badge.

---

## 11. Insufficient Documentation Behavior

What happens if a customer asks about an undocumented machine or an unknown fault code?

- The assistant will **never fabricate an answer**.
- Instead, the suggestions card will display a clear warning banner:
  > **⚠️ Insufficient Technical Documentation**  
  > *The available documentation does not contain verified resolution steps for this specific issue. Escalate to senior technical engineering.*
- The grounding indicator will display **Ungrounded / Safe Fallback**.
- When you see this, inform the customer that their scenario requires specialized engineering consultation, and escalate the ticket according to your standard operating procedure.

---

## 12. Stopping a Session

When the customer call ends:

1. Click the red **"Stop Session"** button in the header.
2. Audio streaming and microphone capture will stop immediately.
3. The session timer will pause, and the microphone indicator will return to **Ready**.
4. The transcript and suggestions will remain on your screen so you can complete your call documentation or CRM ticket notes.
5. Clicking **"Start Session"** again will begin a fresh session with a cleared feed.

---

## 13. Basic Troubleshooting

| Issue | Likely Cause | What to Do |
| :--- | :--- | :--- |
| **Status shows "Disconnected"** | Backend server is not running or network interrupted. | Check that your terminal window running `uvicorn` is still active. If it stopped, restart it. Refresh your browser page. |
| **Microphone shows "Error" or "Not Capturing"** | Microphone permissions blocked in operating system. | Ensure your browser or terminal has permission to access the microphone in System Settings (Privacy & Security -> Microphone). |
| **Transcripts are delayed or not appearing** | AssemblyAI API key missing or internet connection down. | Verify your internet connection is active. Ensure `ASSEMBLYAI_API_KEY` is properly entered in your `.env` file. |
| **Suggestions take several seconds to appear** | LLM API queue latency or rate pacing. | The assistant normally delivers recommendations in approximately 3 seconds under normal network conditions. If the Google Gemini free-tier rate limit is reached, recommendations may pause briefly. |
| **Role says "UNKNOWN" frequently** | Soft speech, low microphone volume, or heavy background noise. | Move the headset microphone closer to your mouth and speak clearly. Ensure the customer's audio is clearly audible. |
| **Empty Knowledge Base / 0 Chunks** | Ingestion script has not been run. | Run `python scripts/ingest_documents.py` in your terminal to index the technical documentation. |

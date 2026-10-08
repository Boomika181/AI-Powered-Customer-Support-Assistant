# AI-Powered Customer Support Assistant

A real-time, local-first AI copilot designed for industrial machinery customer support agents. The system captures live microphone audio during support calls, streams real-time speech transcription via AssemblyAI, semantically classifies speaker roles (Customer vs. Agent) and emotional tone using Google Gemini 3.5 Flash-Lite, categorizes queries, retrieves authoritative procedures from local ChromaDB vector storage via Gemini Embedding 2, and synthesizes grounded, safety-compliant troubleshooting suggestions on an interactive web dashboard.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Key Features](#3-key-features)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Project Structure](#6-project-structure)
7. [Prerequisites](#7-prerequisites)
8. [Installation](#8-installation)
9. [Environment Variables](#9-environment-variables)
10. [API Key Setup](#10-api-key-setup)
11. [Knowledge Base / Document Ingestion](#11-knowledge-base--document-ingestion)
12. [Embedding Model](#12-embedding-model)
13. [Vector Database](#13-vector-database)
14. [Gemini Model](#14-gemini-model)
15. [Real-Time Transcription](#15-real-time-transcription)
16. [Semantic Customer/Agent Role Classification](#16-semantic-customeragent-role-classification)
17. [Sentiment Classification](#17-sentiment-classification)
18. [Query Categorization](#18-query-categorization)
19. [RAG Support Suggestions](#19-rag-support-suggestions)
20. [Grounding / Source References](#20-grounding--source-references)
21. [Running the Application](#21-running-the-application)
22. [Opening the Dashboard](#22-opening-the-dashboard)
23. [Running Tests](#23-running-tests)
24. [Expected Test Results](#24-expected-test-results)
25. [End-to-End Demo Flow](#25-end-to-end-demo-flow)
26. [Limitations](#26-limitations)
27. [Safety / Human Verification Notes](#27-safety--human-verification-notes)

---

## 1. Project Overview

The **AI-Powered Customer Support Assistant** is a local Proof of Concept (POC) engineered to support customer service and field support personnel troubleshooting complex industrial equipment. Operating alongside the agent during live phone or headset conversations, the assistant continuously processes speech input, isolates customer statements, performs real-time retrieval against technical documentation packs, and surfaces actionable solutions, safety warnings, and manual citations before the agent finishes typing.

---

## 2. Problem Statement

Industrial machinery support agents handle calls regarding mission-critical hardware—such as CNC panel saws, plasma cutters, bridge saws, and automated glass cutters. When a machine faults:
- **Costly Downtime:** Factory downtime costs hundreds of dollars per minute; agents must diagnose faults rapidly.
- **Dense Technical Literature:** Manuals and fault code listings span hundreds of pages across disparate PDF and DOCX files. Searching these manuals manually while conversing with an agitated machine operator causes prolonged resolution delays.
- **High Consequence of Error:** Misinterpreting a fault code or skipping lockout/tagout (LOTO) protocols risks severe operator injury and catastrophic machine damage.
- **Audio Diarization Challenges:** In single-microphone headset setups, speaker separation by acoustic hardware alone often drifts; software must accurately distinguish who is speaking (Customer vs. Agent) without hardcoded assumptions.

---

## 3. Key Features

- **Zero Disk Audio Capture:** Captures 16 kHz mono 16-bit PCM audio directly in memory via `sounddevice`, ensuring zero voice recordings are written to local disk.
- **Low-Latency Streaming Transcription:** Real-time speech-to-text via AssemblyAI Streaming API v3 over WebSocket with partial transcript preview and turn finalization hooks.
- **Semantic Role Classification:** Employs Gemini 3.5 Flash-Lite and conversational continuity tracking to label turns as `CUSTOMER`, `AGENT`, or `UNKNOWN` with a conservative confidence gate (>= 0.80).
- **Consolidated Customer Context Analysis:** Parallelized single-turn analysis that extracts customer emotional tone (`Positive`, `Neutral`, `Negative`, `Agitated`) and query category (`Machine Operation Issues`, `Maintenance & Parts`, `Technical Troubleshooting`).
- **Semantic RAG with Gemini Embedding 2:** Fixed 768-dimensional embeddings searching a persistent ChromaDB vector store across ingested PDF and DOCX technical manuals.
- **Strictly Grounded Support Suggestions:** Generates structured action cards, likely causes, and safety warnings strictly derived from retrieved documentation, completely refusing ungrounded hallucinations.
- **Interactive Local Dashboard:** Clean dark-mode UI built in vanilla HTML5, CSS3, and JavaScript, connected via WebSocket (`/ws/live`) to provide live updates, telemetry, citation cards, and session controls.

---

## 4. System Architecture

```
                       Support Agent Headset / Microphone
                                       │
                                       ▼
                     [sounddevice] 16 kHz Mono PCM Stream
                                       │
                                       ▼
              [AssemblyAI Streaming v3] WebSocket Speech-to-Text
                                       │
                    Partial Transcripts ├──► Live UI Typing Preview
                      Final Turn Events └──► Speech Break Trigger
                                       │
                                       ▼
                       [PipelineCoordinator (FastAPI)]
                                       │
             [Speaker Role Classifier] Gemini 3.5 Flash-Lite
                     ├── AGENT / UNKNOWN ──► Display Turn (No RAG Trigger)
                     └── CUSTOMER Turn ────► Trigger Downstream Pipeline
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼ (Thread 1)                            ▼ (Thread 2)
     [Customer Context Analyzer]               [RAG Vector Search]
        Gemini 3.5 Flash-Lite                 Gemini Embedding 2 (768d)
     ├── Sentiment Analysis (4 classes)                    │
     └── Query Categorization (3 classes)                  ▼
                   │                           [Local ChromaDB Store]
                   │                           Top-3 Relevant Chunks
                   └───────────────────┬───────────────────┘
                                       ▼
                        [Support Suggestion Generator]
                            Gemini 3.5 Flash-Lite
                                       │
                 ├── Actionable Troubleshooting Guidance
                 ├── Critical Industrial Safety Protocols
                 └── Verified Provenance & Manual Citations
                                       │
                                       ▼
                         [WebSocket Endpoint: /ws/live]
                                       │
                                       ▼
                   Agent Web Dashboard (http://localhost:5000/dashboard)
```

---

## 5. Technology Stack

| Component | Technology | Version / Specification | Rationale & Implementation Details |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python | 3.10 – 3.13 | Core backend, async orchestration, and math operations. |
| **Backend Framework** | FastAPI / Uvicorn | 0.115+ / 0.30+ | Asynchronous REST endpoints (`/health`, `/dashboard`) and WebSocket (`/ws/live`). |
| **Audio Capture** | `sounddevice` | 0.5+ | Captures 16 kHz mono 16-bit PCM (`pcm_s16le`) in-memory; zero disk storage. |
| **Live Transcription** | AssemblyAI Streaming | v3 WebSocket SDK | High-accuracy real-time speech streaming with end-of-turn event hooks. |
| **LLM Generation** | Google Gemini 3.5 Flash-Lite | `gemini-3.5-flash-lite` | Direct text generation via `google-genai` SDK for role, sentiment, category, and suggestions. |
| **Embedding Model** | Google Gemini Embedding 2 | `gemini-embedding-2` | Generates 768-dimensional document and query vectors (`output_dimensionality=768`). |
| **Vector Database** | ChromaDB | 0.5+ (Local) | Persistent local vector store (`./chroma_db`) using cosine distance similarity. |
| **Document Parsing** | PyMuPDF & python-docx | `fitz` 1.24+ / `docx` 1.1+ | Robust extraction of tables, sections, and page references from PDF and DOCX files. |
| **Frontend UI** | HTML5 / CSS3 / Vanilla JS | Modern ES6+ | Real-time reactive dashboard, dark mode, zero frontend frameworks. |

---

## 6. Project Structure

```
AI-Powered Customer Support Assistant/
├── .env.example                     # Environment template with required variable keys
├── .gitignore                       # Ignores secrets (.env), virtualenv, bytecode, and cache
├── requirements.txt                 # Pinned project dependencies
├── README.md                        # Master architectural and operational documentation
│
├── chroma_db/                       # Local persistent ChromaDB vector storage (generated)
│
├── data/
│   ├── raw/                         # Source sample technical documentation
│   │   ├── Sample_Technical_Documentation_Pack_SYNTHETIC.pdf
│   │   └── Kestrel_Machinery_Support_Knowledge_Base_SYNTHETIC.docx
│   ├── processed/                   # Cached extracted chunks with rich metadata
│   │   └── chunks.json
│   └── test_cases/                  # Curated evaluation datasets and results
│       ├── support_suggestion_test_cases.json
│       ├── support_suggestion_evaluation_results.json
│       └── support_suggestion_evaluation_results.md
│
├── docs/                            # Architectural specifications and user manuals
│   ├── architecture.md              # Detailed data flows and component responsibilities
│   ├── tech_stack.md                # Technology decisions and evaluation matrix
│   ├── model_evaluation.md          # Benchmark comparison of LLM candidates
│   ├── USER_GUIDE.md                # Support agent operational handbook
│   ├── DEMO_SCRIPT.md               # 3–5 minute step-by-step presentation script
│   ├── PHASE_1_COMPLETION_VERIFICATION.md # Phase 1 verification report
│   ├── SUGGESTION_LATENCY_AUDIT.md  # Latency breakdown and profiling audit
│   └── SUGGESTION_CLIENT_REUSE_IMPLEMENTATION.md # Persistent client optimization report
│
├── src/                             # Core application source code
│   ├── __init__.py
│   ├── audio_capture.py             # sounddevice in-memory microphone capture
│   ├── transcription.py             # AssemblyAI v3 streaming client and conversation buffer
│   ├── speaker_role_classifier.py   # Gemini 3.5 Flash-Lite semantic role classifier & continuity tracker
│   ├── customer_context_analyzer.py # Combined sentiment & category evaluator (Option B)
│   ├── sentiment_analysis.py        # Standalone customer sentiment module
│   ├── query_categorization.py      # Standalone query classification module
│   ├── document_ingestion.py        # PDF & DOCX parser, table normalizer, and chunker
│   ├── embeddings.py                # Gemini Embedding 2 integration (768d, no fallback)
│   ├── rag_search.py                # ChromaDB vector retrieval with similarity scores
│   ├── llm_suggestions.py           # Grounded suggestion generator with persistent client reuse
│   ├── pipeline_coordinator.py      # Master orchestrator bridging audio, STT, LLM, and WebSockets
│   └── main.py                      # FastAPI server, static mounts, and WebSocket endpoint
│
├── static/                          # Agent web dashboard frontend
│   ├── index.html                   # Semantic HTML5 console layout
│   ├── style.css                    # Modern industrial dark theme and glassmorphism styling
│   └── app.js                       # WebSocket management, event handling, and DOM rendering
│
├── scripts/                         # Operational utilities and live benchmark scripts
│   ├── ingest_documents.py          # Executes repeatable PDF & DOCX ingestion pipeline
│   ├── test_microphone.py           # Standalone microphone capture verification
│   ├── verify_chromadb.py           # ChromaDB collection inspection and test search
│   ├── verify_phase0.py             # Phase 0 validation runner
│   ├── run_phase1_e2e_verification.py # Full Phase 1 automated runner
│   ├── test_suggestions_e2e.py      # 5-scenario feature-level E2E pipeline verification
│   ├── test_support_suggestions_dataset.py # 12-case RAG evaluation benchmark
│   ├── verify_bounded_context_live.py # Live bounded context verification
│   ├── verify_parallel_downstream_live.py # Parallel execution live verification
│   └── verify_suggestion_client_reuse_live.py # Persistent client reuse test
│
└── tests/                           # Complete automated unit test regression suite (186 tests)
    ├── test_api_endpoints.py
    ├── test_customer_context_analyzer.py
    ├── test_document_ingestion.py
    ├── test_embeddings.py
    ├── test_llm_suggestions.py
    ├── test_phase0.py
    ├── test_pipeline_coordinator.py
    ├── test_query_categorization.py
    ├── test_rag_search.py
    ├── test_sentiment_analysis.py
    ├── test_speaker_role_classifier.py
    ├── test_support_suggestions_dataset.py
    └── test_transcription.py
```

---

## 7. Prerequisites

- **Operating System:** macOS (Apple Silicon or Intel), Linux (Ubuntu 20.04+), or Windows 10/11.
- **Python:** Python 3.10, 3.11, 3.12, or 3.13 installed.
- **Audio Hardware:** Operational microphone (built-in laptop mic or USB headset).
- **Network:** Outbound HTTPS/WSS access to AssemblyAI and Google Gemini APIs.

---

## 8. Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd "AI-Powered Customer Support Assistant"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   *(Windows: `venv\Scripts\activate`)*

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 9. Environment Variables

Configuration is loaded from `.env` in the project root. Inspect the provided template:

```ini
# AssemblyAI API Key for Speech-to-Text streaming
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Google Gemini API Key for Embeddings and LLM tasks
GEMINI_API_KEY=your_gemini_api_key_here

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=technical_documentation

# Model Configurations
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_MODEL=gemini-3.5-flash-lite
LLM_MODEL=gemini-3.5-flash-lite

# Audio Capture Configuration
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_CHUNK_SIZE=1024

# Speaker Diarization Defaults
SPEAKER_A_ROLE=Customer
SPEAKER_B_ROLE=Agent
SPEAKER_ROLE_CONFIDENCE_THRESHOLD=0.80
MAX_CUSTOMER_TURNS=4
```

---

## 10. API Key Setup

1. **Copy the example configuration:**
   ```bash
   cp .env.example .env
   ```
2. **Obtain API keys:**
   - **AssemblyAI:** Register at [assemblyai.com](https://www.assemblyai.com) and copy your API key from the dashboard.
   - **Google Gemini:** Generate an API key in [Google AI Studio](https://aistudio.google.com/).
3. **Insert the keys into `.env`:**
   ```ini
   ASSEMBLYAI_API_KEY=a1b2c3d4e5f6...
   GEMINI_API_KEY=AIzaSy...
   ```

---

## 11. Knowledge Base / Document Ingestion

The offline ingestion pipeline processes sample technical manuals into structured, searchable text representations and indexes them into ChromaDB:

```bash
python scripts/ingest_documents.py
```

### Ingestion Details
- **Source Documents:**
  - `data/raw/Sample_Technical_Documentation_Pack_SYNTHETIC.pdf` (Vantor Industrial equipment: WP-400 Panel Saw, SC-900 Bridge Saw, MC-250 Plasma Cutter, GC-120 Glass Cutter).
  - `data/raw/Kestrel_Machinery_Support_Knowledge_Base_SYNTHETIC.docx` (Kestrel Machinery equipment: WR-700 CNC Router).
- **Chunk Extraction:** Generates **68 chunks** (34 PDF chunks + 34 DOCX chunks).
- **Table Normalization:** Converts DOCX maintenance tables into row-by-row markdown formats preserving parameters, error codes, and pressure thresholds.
- **Metadata Retained:** Each chunk stores `source_file`, `file_type`, `page`, `document_reference`, `section`, `category`, and `machine`.
- **Deduplication:** Cleanly overwrites existing file chunks on re-ingestion.

---

## 12. Embedding Model

- **Model:** Google Gemini Embedding 2 (`gemini-embedding-2`).
- **Dimensions:** Fixed at **768 dimensions** via `output_dimensionality=768`.
- **No Local Fallback:** The system strictly rejects fallback models (such as MiniLM) to prevent dimensional collisions and vector contamination.
- **Batch Processing:** Document chunks are embedded in batches of 50 chunks per API call with strict dimension validation.
- **Symmetric Vector Space:** Both ingested document chunks and live runtime customer search queries use `gemini-embedding-2`.

---

## 13. Vector Database

- **Engine:** ChromaDB (Local Persistent Client).
- **Storage Path:** `./chroma_db`.
- **Collection Name:** `technical_documentation`.
- **Distance Metric:** Cosine distance.
- **Normalized Similarity Score:** Calculated as `similarity_score = max(0.0, 1.0 - (distance / 2.0))`.
- **Local Isolation:** Runs completely embedded in-process with zero external cloud vector service dependencies.

---

## 14. Gemini Model

- **Configured Model:** Google Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`).
- **SDK:** `google-genai` official Python SDK.
- **Roles Served:**
  1. Semantic Speaker-Role Classification (`src/speaker_role_classifier.py`).
  2. Consolidated Customer Context Analysis (`src/customer_context_analyzer.py`).
  3. Grounded Support Suggestion Synthesis (`src/llm_suggestions.py`).
- **Optimization:** Persistent client instances (`get_shared_gemini_client()`) eliminate redundant TLS 1.3 socket negotiations across conversational turns.
- **Robust Parsing:** Strips markdown code fences (` ```json `) and parses strict JSON payloads with Python fallbacks.

---

## 15. Real-Time Transcription

- **Provider:** AssemblyAI Streaming API v3 via WebSocket.
- **Audio Specifications:** 16,000 Hz sample rate, single channel (mono), 16-bit PCM format.
- **Zero Disk Storage:** Streamed directly from the `sounddevice` callback buffer across the WebSocket; audio frames are never written to disk.
- **Streaming Hooks:**
  - `on_partial`: Dispatches interim word tokens to display real-time speech previews on the UI.
  - `on_speech_break`: Triggered upon `end_of_turn == True` or terminal silence, producing a finalized transcript turn for downstream evaluation.

---

## 16. Semantic Customer/Agent Role Classification

In single-microphone setups, raw acoustic labels (e.g., Speaker A vs. Speaker B) can drift or swap. To solve this, `src/speaker_role_classifier.py` provides semantic role classification:
- **Decision Engine:** Evaluates conversational discourse using Gemini 3.5 Flash-Lite.
- **Output Labels:** `CUSTOMER`, `AGENT`, or `UNKNOWN`.
- **Confidence Gate:** Requires `confidence >= 0.80`. Any turn scoring below 0.80 falls back safely to `UNKNOWN`.
- **Speaker Continuity:** Tracks confirmed acoustic turn history via `SpeakerContinuityTracker` to maintain speaker consistency without hardcoded locks.
- **Downstream Protection:** Only turns definitively resolved as `CUSTOMER` trigger the downstream RAG and suggestion pipeline; `AGENT` and `UNKNOWN` turns are logged to the dialogue timeline without triggering unnecessary RAG queries.

---

## 17. Sentiment Classification

- **Target:** Classifies the **customer's** emotional tone exclusively, ignoring the support agent's tone.
- **Discrete Classes:**
  - `Positive`: Customer expresses satisfaction, gratitude, relief, or approval.
  - `Neutral`: Customer provides calm, objective, factual statements.
  - `Negative`: Customer expresses dissatisfaction, operational frustration, or product malfunction.
  - `Agitated`: Customer exhibits acute anger, urgency, repeated failure, or aggressive escalation.
- **Output Payload:** Returns `sentiment`, numerical `confidence`, and textual `rationale`.

---

## 18. Query Categorization

- **Classification Schema:** Maps customer queries into three discrete SOW support categories:
  1. `Machine Operation Issues`: Startup sequences, normal operating modes, feed rates, spindle speed adjustments, control panels.
  2. `Maintenance & Parts`: Preventive service schedules, filter replacements, lubrication, consumable wear parts, spare part ordering.
  3. `Technical Troubleshooting`: Fault codes (e.g., E-102, R-101), error alarms, motor stalls, overheating, sensor trips, diagnostics.
- **Output Payload:** Returns `category`, numerical `confidence`, and textual `rationale`.

---

## 19. RAG Support Suggestions

When a customer fault or question is detected:
1. **Context Bounding:** Gathers the latest customer turns (bounded by default to the last 4 customer turns) to construct a focused query.
2. **Semantic Retrieval:** Queries ChromaDB using `gemini-embedding-2` to fetch the top 3 most relevant documentation chunks.
3. **Grounded Synthesis:** Gemini 3.5 Flash-Lite generates a structured recommendation card containing:
   - **Likely Issue:** A concise root-cause diagnosis.
   - **Recommended Actions:** 2–3 actionable, step-by-step troubleshooting instructions.
   - **Safety Protocols:** Mandatory warnings for electrical lock-out/tag-out (LOTO), waiting for rotating cutting blades to stop, or pneumatic relief.

---

## 20. Grounding / Source References

- **Strict Anti-Hallucination Policy:** Suggestions are synthesized strictly from retrieved manual chunks. The model is explicitly forbidden from inventing error codes, part numbers, pneumatic pressures, or wiring pinouts.
- **Provenance Citations:** Every suggestion card displays verified source citations extracted from chunk metadata:
  - Source document name (e.g., `Sample_Technical_Documentation_Pack_SYNTHETIC.pdf`)
  - Page number (e.g., Page 8)
  - Document reference code (e.g., `TS-WP400-3.1`)
  - Specific section title (e.g., `Section 3.1 Error Codes and Alarms`)
- **Insufficient Context Fallback:** If ChromaDB returns no chunks or if the documentation does not contain answers for the customer's query, the generator emits an explicit disclaimer:
  > *"The available documentation does not provide sufficient information to resolve this issue. Escalate to senior technical engineering."*
  In this event, `is_grounded` is marked `False`, and no hallucinated instructions are generated.

---

## 21. Running the Application

Ensure your virtual environment is active and API keys are set in `.env`.

### 1. Ingest Technical Documentation (First-Time Setup)
```bash
python scripts/ingest_documents.py
```

### 2. Launch FastAPI Server
```bash
uvicorn src.main:app --host 127.0.0.1 --port 5000 --reload
```
*(Alternative launcher: `python -m src.main`)*

The server initializes on `http://127.0.0.1:5000` with:
- REST Health Check: `GET http://127.0.0.1:5000/health`
- Agent Dashboard: `GET http://127.0.0.1:5000/dashboard`
- WebSocket Endpoint: `ws://127.0.0.1:5000/ws/live`

---

## 22. Opening the Dashboard

1. Open any modern web browser (Chrome, Edge, Safari, Firefox).
2. Navigate to:
   ```
   http://localhost:5000/dashboard
   ```
3. Observe the dashboard interface:
   - **Header:** Session status beacon, session timer, turn counter, microphone device label, and "Start Session" / "Stop Session" buttons.
   - **Subsystems Bar:** Real-time health indicators for Microphone, AssemblyAI, Speaker Classification, Gemini 3.5 Flash-Lite, and ChromaDB.
   - **Left Column:** Live Conversation timeline showing partial speech previews, speaker tags (`CUSTOMER` vs. `AGENT`), confidence tags, and timestamps.
   - **Right Column:** Customer Sentiment card, Query Category badge, AI Support Suggestions hero card, and Knowledge Base Sources citations.

---

## 23. Running Tests

The test suite covers unit logic, mock integration, live APIs, and end-to-end pipeline execution.

### Run All Unit Tests (186 Tests, Offline / Mocks)
```bash
python -m unittest discover tests
```

### Run Phase 0 Verification
```bash
python scripts/verify_phase0.py
```

### Run Feature-Level End-to-End Pipeline Verification
```bash
python scripts/test_suggestions_e2e.py
```

### Run 12-Case Support Suggestions Dataset Benchmark
```bash
python scripts/test_support_suggestions_dataset.py
```

---

## 24. Expected Test Results

### 1. Unit Test Suite (`unittest discover tests`)
- **Tests Executed:** 186 tests.
- **Pass Rate:** **100% (186/186 PASS)**.
- **Execution Time:** ~12–14 seconds.
- **Coverage:** Fast mock tests for audio capture, transcription buffers, speaker role classification, sentiment, categorization, document ingestion, embeddings, ChromaDB search, LLM suggestions, pipeline coordinator, and FastAPI endpoints.

### 2. Live E2E Pipeline Suite (`scripts/test_suggestions_e2e.py`)
- **Scenarios:** 5 industrial test cases.
- **Result:** **5/5 Passed (100%)**.
  1. *Technical Troubleshooting (PDF):* WP-400 Error E-102 -> Grounded in `TS-WP400-3.1` and `UM-WP400-1.1.3`.
  2. *Maintenance & Parts (PDF):* WP4-FL-DE1 filter replacement -> Grounded in `UM-WP400-1.1.3`.
  3. *Machine Operation (PDF):* WP-400 daily startup -> Grounded in `UM-WP400-1.1.1`.
  4. *DOCX Knowledge Base:* WR-700 CNC Router startup -> Grounded in `OG-WR700-1.1.1`.
  5. *Unknown Query:* Unlisted machine pressure -> Successfully triggered ungrounded disclaimer without hallucinations.

### 3. Support Suggestion Dataset Benchmark (`scripts/test_support_suggestions_dataset.py`)
- **Scenarios:** 12 curated test cases.
- **Grounding Accuracy:** **100% (12/12 Passed)**.
- **Source Citation Accuracy:** **100%**.
- **No-Context Safety Rate:** **100%**.

---

## 25. End-to-End Demo Flow

To conduct a live or simulated 3–5 minute demonstration using the standard industrial scenario:

1. **Start the Application:** Run `uvicorn src.main:app --port 5000` and open `http://localhost:5000/dashboard`.
2. **Start Session:** Click **"Start Session"**. Subsystems turn green, microphone begins listening, and session timer starts.
3. **Turn 1 (Customer Problem):**
   - *Speak or simulate:* *"Hello, my WP-400 panel saw stopped running and is showing error E-102. The dust extraction fan is completely off."*
   - *Result:* AssemblyAI streams partial text; turn finalizes; role classified as **CUSTOMER** (conf > 0.90); sentiment displays **Negative** (or **Agitated**); category displays **Technical Troubleshooting**; ChromaDB retrieves chunks for `TS-WP400-3.1`; AI Support Suggestions renders:
     - **Likely Issue:** Differential pressure trip (>1200 Pa) or fan motor breaker trip.
     - **Recommended Actions:** Check circuit breaker; inspect `WP4-FL-DE1` cartridge filter.
     - **Safety Warning:** Lock out main isolator before opening filter compartment.
     - **Sources:** Cites `Sample_Technical_Documentation_Pack_SYNTHETIC.pdf`, Page 8, Ref `TS-WP400-3.1`.
4. **Turn 2 (Agent Reply):**
   - *Speak or simulate:* *"I understand. Please do not touch the filter while the saw is powered. Check the breaker first."*
   - *Result:* Role classified as **AGENT**; displayed in blue on the timeline; downstream RAG does not re-trigger.
5. **Turn 3 (Customer Clarification):**
   - *Speak or simulate:* *"Okay, the breaker tripped. How often do we need to replace the WP4-FL-DE1 filter?"*
   - *Result:* Role classified as **CUSTOMER**; category switches to **Maintenance & Parts**; suggestions recommend replacement interval (3 months or upon E-102 alarm); cites `UM-WP400-1.1.3`.
6. **Stop Session:** Click **"Stop Session"**. Audio capture shuts down safely.

*(For full narration instructions, see [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)).*

---

## 26. Limitations

1. **Measured Latency Behavior:**
   - Under unthrottled conditions with persistent client reuse, the clean downstream processing pipeline completes in approximately **2.98 to 3.32 seconds** (~1.6s for parallel context analysis and retrieval, and ~1.4s to 1.6s for suggestion synthesis).
   - Under Google AI Studio Free Tier rate limits (15 RPM) or during concurrent requests, server-side queue latency increases turnaround times.
   - **The system does NOT achieve response latencies consistently under 2 seconds.** The initial SOW target (<2.0s) remains an aspirational benchmark requiring future dedicated enterprise throughput or model fine-tuning.
2. **Single Microphone Diarization:**
   - Single-microphone audio uses AssemblyAI single-channel speaker clustering combined with Gemini semantic classification. In production noisy call-center environments, stereo dual-channel capture (Customer on Channel 1, Agent on Channel 2) provides stronger acoustic separation.
3. **Synthetic Knowledge Base:**
   - The ingested documents (`Sample_Technical_Documentation_Pack_SYNTHETIC.pdf` and `Kestrel_Machinery_Support_Knowledge_Base_SYNTHETIC.docx`) are purely sample/synthetic technical documentation created for testing and evaluation purposes.
4. **Internet Dependency:**
   - Requires continuous outbound network connectivity to AssemblyAI WebSocket endpoints and Google Gemini AI endpoints.

---

## 27. Safety / Human Verification Notes

- **AI Copilot Advisory Role:** This assistant is strictly an operator support copilot. It does NOT directly actuate, communicate with, or command industrial hardware.
- **Human Verification Required:** All recommended maintenance actions, voltage checks, and mechanical adjustments must be verified by a qualified human technician or support engineer against official manufacturer documentation.
- **Mandatory Safety Protocols:** Technicians must strictly observe all plant safety procedures, including Lock-Out / Tag-Out (LOTO), verifying complete cessation of rotating machine components, and wearing required personal protective equipment (PPE: safety glasses, hearing protection, gloves) prior to servicing industrial equipment.
- **Zero Disk Audio Retention:** Audio frames are processed strictly in volatile RAM to protect customer voice privacy and meet compliance requirements; audio data is never written to disk.

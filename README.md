# AI-Powered Customer Support Assistant — Proof of Concept (POC)

A localhost AI-assisted customer support copilot designed for industrial machinery support agents. The system captures live microphone audio from the agent's side, performs real-time speech transcription via AssemblyAI, analyzes customer sentiment, categorizes queries, and provides grounded troubleshooting suggestions retrieved from technical manuals using ChromaDB and Google Gemini Flash.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 1 — SLICE 1 ACTIVE**
> 
> - **Phase 0:** Setup, document ingestion, ChromaDB vector store, and microphone test completed & verified.
> - **Phase 1 (Slice 1 Implemented):** Audio Capture (`src/audio_capture.py`), AssemblyAI Streaming API v3 (`src/transcription.py`), in-memory `ConversationState`, and Speech Break trigger logic.
> - **Pending in Phase 1 / 2:** Sentiment analysis, query categorization, grounded RAG suggestions via Gemini Flash, and web dashboard UI.

---

## Architecture Overview

```
Support Agent (Microphone)
        │
        ▼
[sounddevice] Audio Stream Capture (Zero Disk Storage)
        │
        ▼
[AssemblyAI v3] Real-Time Streaming Transcription & Diarization
        │  (Speech Break Token: 'FinalTranscript' / 'utterance_end')
        ▼
[Local Python App] Orchestration & Parsing (FastAPI)
   ├── Tone Analysis ──► [Gemini 3.8 Flash] ──► Sentiment Indicator
   ├── Query Classifier ──► [Gemini 3.8 Flash] ──► Category Badge
   └── Vector Search ──► [ChromaDB Local] ──► Context Retrieval
                              │
                              ▼
            [Gemini 3.8 Flash] RAG Suggestions
                              │
                              ▼
        [HTML / CSS / JavaScript] Agent Dashboard (localhost:5000)
```

### Knowledge Base Offline Pipeline (Phase 0)
```
Synthetic Technical PDF Manual (Sample_Technical_Documentation_Pack_SYNTHETIC.pdf)
        │
        ▼
Text & Section Extraction (PyMuPDF)
        │
        ▼
Semantic Section & Table Chunking (PDF + DOCX, rich metadata)
        │
        ▼
Vector Embeddings (Gemini Embedding 2: gemini-embedding-2, 768 dimensions)
        │
        ▼
Persistent Local Vector Storage (ChromaDB at ./chroma_db)
```

---

## Project Structure

```
AI-Powered Customer Support Assistant/
├── .env.example              # Template for API keys and configuration
├── .gitignore                # Prevents secrets, venv, and binary data from being committed
├── requirements.txt          # Direct Python dependencies
├── README.md                 # Complete project guide and Phase 0 documentation
│
├── data/
│   ├── raw/                  # Source synthetic PDF manual
│   │   └── Sample_Technical_Documentation_Pack_SYNTHETIC.pdf
│   └── processed/            # Extracted chunks with structured metadata
│       └── chunks.json
│
├── chroma_db/                # Local persistent ChromaDB vector storage
│
├── docs/
│   ├── architecture.md       # Detailed system design, data flows, and Mermaid diagram
│   ├── tech_stack.md         # Finalized technology stack table & justifications
│   └── model_evaluation.md   # Model benchmark comparison (Gemini Flash vs DeepSeek vs Qwen)
│
├── src/                      # Core application source code
│   ├── __init__.py
│   ├── audio_capture.py      # Audio capture interface (Phase 1 placeholder)
│   ├── transcription.py      # AssemblyAI streaming integration (Phase 1 placeholder)
│   ├── sentiment_analysis.py # Sentiment analysis module (Phase 1 placeholder)
│   ├── rag_search.py         # ChromaDB search and retrieval module
│   ├── llm_suggestions.py    # Grounded recommendation generator (Phase 1 placeholder)
│   └── main.py               # FastAPI server entry point & Phase 0 health endpoint
│
├── scripts/                  # Executable pipeline and verification scripts
│   ├── ingest_documents.py   # Repeatable PDF extraction, chunking, and ChromaDB ingestion
│   ├── test_microphone.py    # Standalone microphone detection and frame capture test
│   ├── verify_chromadb.py    # ChromaDB collection inspection and similarity search test
│   └── verify_phase0.py      # Master verification runner checking all 6 Phase 0 criteria
│
└── tests/                    # Automated test suite
    ├── __init__.py
    └── test_phase0.py        # Automated unit tests for Phase 0 components
```

---

## Technology Stack

| Component | Selected Technology | Notes |
| :--- | :--- | :--- |
| **Microphone Capture** | `sounddevice` | Streams raw PCM frames in memory. Zero audio files stored on disk. |
| **Speech-to-Text** | AssemblyAI v3 (Streaming API) | Real-time WebSocket streaming with speaker diarization. |
| **LLM Inference** | Google Gemini 3.8 Flash (`gemini-3.8-flash`) | Fast response, high accuracy across sentiment, categories, and RAG suggestions. |
| **Embeddings** | Gemini Embedding 2 (`gemini-embedding-2`) | Converts documents & queries into 768-dimensional vectors (`output_dimensionality=768`). No local fallback. |
| **Vector Database** | ChromaDB (Local) | Persistent local vector store (`./chroma_db`) with 768-dim vectors. Zero cloud setup. |
| **Document Processing** | `PyMuPDF` + `python-docx` | High-speed PDF and DOCX extraction with structured table normalization and page references. |
| **Backend API** | FastAPI / Uvicorn | Async Python framework running locally on port 5000. |
| **Frontend Dashboard**| Vanilla HTML / CSS / JavaScript | Lightweight UI displaying live transcripts, sentiment, and suggestion cards. |

---

## Environment Setup & Installation

### 1. Prerequisites
- macOS, Linux, or Windows
- Python 3.10 to 3.13
- System microphone (built-in or USB headset)
- Internet connection (for AssemblyAI and Gemini APIs)

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
*(On Windows: `venv\Scripts\activate`)*

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the template `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your API keys:
```ini
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=technical_documentation
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

> [!NOTE]
> All document chunks and search queries are embedded using Google Gemini Embedding 2 (`gemini-embedding-2`) at a fixed 768-dimensional vector space. No local MiniLM fallback is used, ensuring strict vector dimensionality consistency across the entire retrieval pipeline.

---

## Execution Commands

### 1. Document Ingestion (Repeatable Pipeline)
Parses the synthetic PDF manual, creates 34 semantic chunks with complete metadata, and stores them in ChromaDB:
```bash
python scripts/ingest_documents.py
```

### 2. Microphone Capture Test
Detects audio hardware, verifies stream initialization, captures audio frames for 2 seconds, confirms zero disk persistence, and safely stops:
```bash
python scripts/test_microphone.py
```

### 3. ChromaDB Verification Test
Loads the vector collection, verifies chunk count, displays sample stored metadata, and runs a test similarity search:
```bash
python scripts/verify_chromadb.py
```

### 4. Master Phase 0 Verification
Executes the full suite of Phase 0 checks (environment, packages, keys, vector DB, metadata, and microphone):
```bash
python scripts/verify_phase0.py
```

### 5. Run Automated Unit Tests
```bash
python -m unittest discover tests
```

### 6. Run FastAPI Health Server (Optional)
```bash
uvicorn src.main:app --port 5000 --reload
```
Visit `http://localhost:5000/health` in your browser.

---

## Expected Phase 0 Results

When running `python scripts/verify_phase0.py`, you should see:
1. **Python Environment:** `[PASS]` Python 3.13.x verified within virtualenv.
2. **Project Dependencies:** `[PASS]` All 9 core dependencies imported successfully.
3. **Environment Variables & API Keys:** `[PASS]` or `[WARNING]` Displays configured status for AssemblyAI and Gemini keys.
4. **ChromaDB & Document Storage:** `[PASS]` Collection `technical_documentation` loaded with 34 chunks.
5. **Document Metadata Integrity:** `[PASS]` All required metadata keys (`source`, `page`, `document_reference`, `section`, `category`, `machine`) verified.
6. **Microphone Hardware & Frame Capture:** `[PASS]` Default microphone detected, stream opened, audio frames captured, zero disk storage confirmed.

---

## Metadata Schema Preserved in Vector Database

Every chunk in ChromaDB retains the following metadata:
```json
{
  "source": "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf",
  "page": 2,
  "document_reference": "UM-WP400-1.1.1",
  "section": "Section 1.1.1 Daily startup procedure",
  "category": "Machine Operation Issues",
  "machine": "WP-400"
}
```

Support Categories in Document:
- `Machine Operation Issues`
- `Maintenance & Parts`
- `Technical Troubleshooting`
- `General Reference` / `Mixed`

Covered Industrial Machines:
- `WP-400` (CNC Panel Saw - Woodworking)
- `SC-900` (Bridge Saw - Stone cutting)
- `MC-250` (CNC Plasma Cutting Table - Metal cutting)
- `GC-120` (Automatic Glass Cutting Table - Glass cutting)
- `All` (Universal guidelines & parts)

---

## Phase 1 — Real-Time Transcription (Slice 1)

### Architecture of this Slice
```
Agent Microphone (Local Hardware)
        │
        ▼ (sounddevice InputStream, callback queue)
Raw Audio Frames (16000 Hz, mono, 16-bit PCM: pcm_s16le)
        │
        ▼ (Streaming bytes over WebSocket)
AssemblyAI Streaming API v3 (RealTimeTranscriber)
        │
        ├── Partial Transcripts (end_of_turn = False) ──► In-progress console display
        │
        └── Final Transcripts (end_of_turn = True) ────► In-memory ConversationState
                                                                   │
                                                                   ▼
                                                       SPEECH BREAK TRIGGER HOOK
                                           ("Ready for downstream processing")
```

### Key Technical Characteristics
1. **Zero Disk Storage:** Audio PCM frames stream directly from the `AudioCapture` generator over WebSocket into AssemblyAI without ever writing a byte to disk.
2. **Ephemeral In-Memory State:** `ConversationState` holds session ID, timestamped segments, speaker labels, and clean transcript text in RAM. No database is created.
3. **Speech Break Detection:** AssemblyAI v3 dispatches a `TurnEvent`. When `end_of_turn == True`, the turn is finalized, stored in `ConversationState`, and triggers the registered downstream processing callback.
4. **Clean Shutdown:** Captures `SIGINT` (Ctrl+C), closes microphone hardware via `sounddevice.InputStream.close()`, and terminates the AssemblyAI session with `transcriber.disconnect(terminate=True)`.

### Environment Configuration
Ensure `.env` contains your AssemblyAI API key:
```ini
ASSEMBLYAI_API_KEY=your_actual_assemblyai_key_here
```

### Execution Commands

#### 1. Test Microphone Audio Capture Independently
```bash
python scripts/test_microphone.py
```

#### 2. Run Real-Time AssemblyAI Streaming Transcription
```bash
python -m src.transcription
```

#### 3. Run Unit Tests (No API Key Required)
```bash
python -m unittest tests/test_transcription.py
```

### Expected Output
When running `python -m src.transcription` with an active key:
```
=================================================================
CUSTOMER SUPPORT ASSISTANT — LIVE TRANSCRIPTION (PHASE 1 - SLICE 1)
=================================================================

Starting microphone...
Microphone: MacBook Air Microphone
Connecting to AssemblyAI Streaming API...
Listening... (Speak into microphone. Press Ctrl+C to stop)

[customer] My machine is showing error E-102
[customer] and the dust extraction has stopped.

FINAL TRANSCRIPT:
[customer] My machine is showing error E-102 and the dust extraction has stopped.

SPEECH BREAK DETECTED
Conversation state updated.
Ready for downstream processing.
```

*(Note: If `ASSEMBLYAI_API_KEY` is not yet set in `.env`, `python -m src.transcription` displays a configuration reminder and runs an offline state machine verification confirming speech-break routing and conversation state).*

### Known Limitations in Slice 1
- **Diarization Attribution:** Single-microphone audio uses AssemblyAI single-channel speaker labels. In production multi-party audio, dual-channel audio separation can be enabled.
- **Network Latency:** AssemblyAI requires stable Internet access for WebSocket streaming.

### Scope Notice: Pending Downstream Features
As specified in the SOW, the following components are deliberately **not** executed in Slice 1:
- ❌ Sentiment Analysis via Gemini Flash (Pending Slice 2)
- ❌ Automatic Query Categorization via Gemini Flash (Pending Slice 2)
- ❌ Grounded RAG Suggestions via ChromaDB & Gemini (Pending Slice 3)
- ❌ HTML/CSS/JavaScript Dashboard UI (Pending Phase 2)

---

## Security & Confidentiality Notes

- **Confidential Engagement:** This project is part of a confidential Mirai Labs engagement under NDA. Do not publish, share, or push this repository to public remotes.
- **Secrets Management:** Real API keys must NEVER be committed to Git. The `.env` file is explicitly ignored in `.gitignore`. Only `.env.example` with blank variables is checked into source control.
- **Synthetic Test Data:** The technical manual `Sample_Technical_Documentation_Pack_SYNTHETIC.pdf` contains purely fictional specifications created specifically to benchmark this assistant. Do not use for real machinery operation or repair.
- **Audio Privacy:** Microphone input is processed strictly as an in-memory stream for live transcription. Audio data is NEVER written to local disk or permanently retained.

---

## Phase 1 & Phase 2 Roadmap

- **Phase 1 — Core Functionality:**
  - [x] **Slice 1:** Microphone capture, AssemblyAI v3 streaming, in-memory conversation state, speech break trigger.
  - [ ] **Slice 2:** Sentiment analysis and query categorization modules using Gemini 3.8 Flash.
  - [ ] **Slice 3:** RAG retrieval and grounded suggestions generator (`src/rag_search.py` + `src/llm_suggestions.py`).
  - [ ] **Slice 4:** End-to-end command-line simulation linking speech break -> sentiment + RAG -> suggestions.
- **Phase 2 — Dashboard & Delivery:**
  - [ ] HTML / CSS / JavaScript agent dashboard interface.
  - [ ] Real-time transcript display, sentiment gauge, and suggestion card rendering.
  - [ ] End-to-end simulated call testing and demo video recording.

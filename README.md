# AI-Powered Customer Support Assistant — Proof of Concept (POC)

A localhost AI-assisted customer support copilot designed for industrial machinery support agents. The system captures live microphone audio from the agent's side, performs real-time speech transcription via AssemblyAI, analyzes customer sentiment, categorizes queries, and provides grounded troubleshooting suggestions retrieved from technical manuals using ChromaDB and Google Gemini Flash.

> [!IMPORTANT]
> **CURRENT PROJECT STATUS: PHASE 0 ONLY**
> 
> This repository is currently in **Phase 0 (Setup & Document Preparation)**. Core downstream features (continuous transcription streaming, live sentiment analysis, auto query categorization, grounded suggestion cards, and the web dashboard UI) are scheduled for **Phase 1** and **Phase 2**.
> 
> All Phase 0 objectives — environment setup, dependency verification, secure environment variable configuration, local ChromaDB initialization, synthetic technical manual ingestion/chunking/embedding, and microphone frame capture testing — have been implemented and verified.

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
Semantic Section & FAQ Chunking (34 chunks with rich metadata)
        │
        ▼
Vector Embeddings (Gemini API / ChromaDB Local ONNX fallback)
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
| **LLM Inference** | Google Gemini Flash | Fast response (<2s), high accuracy across sentiment, categories, and RAG. |
| **Embeddings** | Gemini Embeddings (`text-embedding-004`) | Semantic vector representation of document chunks. |
| **Vector Database** | ChromaDB (Local) | Fully local embedded vector store (`./chroma_db`). Zero cloud setup. |
| **Document Processing** | `PyMuPDF` (`fitz`) | High-speed PDF text and structural layout extraction. |
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
EMBEDDING_MODEL=models/text-embedding-004
```

> [!NOTE]
> Ingestion can run locally even before API keys are added by falling back to ChromaDB's local ONNX embedding engine. Once you add your `GEMINI_API_KEY`, re-running the ingestion script automatically generates Gemini embeddings.

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

## Security & Confidentiality Notes

- **Confidential Engagement:** This project is part of a confidential Mirai Labs engagement under NDA. Do not publish, share, or push this repository to public remotes.
- **Secrets Management:** Real API keys must NEVER be committed to Git. The `.env` file is explicitly ignored in `.gitignore`. Only `.env.example` with blank variables is checked into source control.
- **Synthetic Test Data:** The technical manual `Sample_Technical_Documentation_Pack_SYNTHETIC.pdf` contains purely fictional specifications created specifically to benchmark this assistant. Do not use for real machinery operation or repair.
- **Audio Privacy:** Microphone input is processed strictly as an in-memory stream for live transcription. Audio data is NEVER written to local disk or permanently retained.

---

## What Remains for Phase 1 & Phase 2

- **Phase 1 (Core Functionality):**
  - Implement `src/audio_capture.py` continuous stream generator.
  - Implement `src/transcription.py` with AssemblyAI streaming WebSocket and speech break detection (`FinalTranscript`).
  - Implement `src/sentiment_analysis.py` using Gemini Flash prompt.
  - Implement `src/llm_suggestions.py` with grounded RAG suggestions and document citations.
  - End-to-end command-line simulation.
- **Phase 2 (Dashboard & Integration):**
  - Implement HTML/CSS/JavaScript dashboard.
  - Real-time transcript display, sentiment gauge, and suggestion cards.
  - End-to-end simulated call testing and demo video recording.

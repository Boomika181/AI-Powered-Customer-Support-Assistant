# Finalized Technology Stack

## AI-Powered Customer Support Assistant – POC

| Component | Chosen Technology | Description | Advantages | Disadvantages | Alternative (1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Microphone Capture** | `sounddevice` (PyAudio) | Captures live audio from system microphone and processes the stream locally. | • Easy to install and use<br>• Clean callback API<br>• No data stored on disk | • Requires native installation<br>• Limited to Python environment | `PyAudio` |
| **Speech-to-Text** | AssemblyAI v3 (Streaming API) | Real-time transcription of audio stream with speaker diarization and turn detection. | • High accuracy<br>• Real-time streaming<br>• Supports speaker labels | • Paid service<br>• Dependent on internet connection | `Deepgram` |
| **Sentiment Analysis** | Gemini 3.8 Flash | Classifies customer sentiment as Positive / Neutral / Negative (using structured JSON output). | • High accuracy and consistency<br>• Good understanding of context<br>• Fast response time | • Depends on prompt design<br>• Requires internet access | `DeepSeek` |
| **Query Categorization** | Gemini 3.8 Flash | Classifies the query into Machine Operation Issues, Maintenance & Parts, or Technical Troubleshooting. | • Accurate classification<br>• Handles varied phrasing<br>• Returns structured output | • May occasionally misclassify ambiguous queries<br>• Requires clear category schema | `Qwen` |
| **Document Processing** | PyMuPDF + python-docx | Extracts text from PDF and DOCX manuals and splits into chunks with page numbers. | • Fast and reliable text extraction<br>• Preserves page references<br>• Works for PDFs and DOCX | • Slower for scanned PDFs<br>• OCR needed for images | `Unstructured` |
| **Embeddings** | gemini-embedding-2 | Converts document chunks and queries into vector embeddings. | • Same provider as Gemini LLM<br>• Good semantic understanding<br>• Replaces older text-embedding models | • Requires internet access<br>• Model may update over time | OpenAI (`text-embedding-3`) |
| **Vector Database** | ChromaDB (Local) | Stores embeddings, chunks and metadata (including page numbers) for semantic search. | • Simple to set up<br>• Works well for POC<br>• Supports metadata filtering | • Not ideal for very large scale<br>• Needs separate setup for production | `Qdrant` |
| **Suggestion Generation (RAG)**| Gemini 3.8 Flash | Generates 2–3 grounded support suggestions using retrieved document chunks with source references. | • Grounded and relevant answers<br>• Includes document references and page numbers<br>• Handles "not found" cases well | • Can be verbose without proper prompt<br>• Dependent on retrieval quality | `DeepSeek` |
| **Backend API** | FastAPI (Python) | Handles audio stream, STT integration, LLM calls and RAG pipeline. | • Fast and lightweight<br>• Easy to develop and maintain<br>• Good async support | • Requires Python deployment environment<br>• Needs proper error handling | `Flask` |
| **Frontend** | HTML / CSS / JavaScript | Web interface to display live transcription, sentiment, category and suggestions. | • Simple and lightweight<br>• No build step required<br>• Easy to customize<br>• Good for POC and demos | • Limited built-in components<br>• More manual UI development<br>• Not ideal for large-scale production | `Streamlit` |

---

## Architectural Rationale: Why No Heavy Orchestration?

- **Direct API Calls over Frameworks:** Rather than introducing large frameworks like LangChain or LlamaIndex, the POC uses native Python calls directly to AssemblyAI, ChromaDB, and Google Gemini SDK.
- **Explainability:** In client demonstrations and technical interviews, every line of code can be explained from first principles without nested framework abstractions.
- **Minimal Dependencies:** Significantly fewer transient dependencies, avoiding package version pinning conflicts and reducing latency.
- **Transparent Code Flow:** Easy debugging with predictable synchronous and asynchronous function calls.

---

## Discrepancies Between Original SOW & Finalized Tech Stack

> [!NOTE]
> The following discrepancies exist between the original Statement of Work (SOW v1.0) and the Finalized Architecture / Tech Stack artifact:

1. **LLM Version:**
   - **SOW:** `Google Gemini Flash 2.0`
   - **Finalized Architecture:** `Gemini 3.8 Flash`
   - *Action Item / Confirmation for Hemanth:* Confirm whether the model identifier in the Gemini API should target `gemini-2.0-flash` or the preview/updated tag referenced in the architecture deck (`Gemini 3.8 Flash`).
2. **Embedding Model:**
   - **SOW:** `Google text-embedding-004`
   - **Finalized Architecture:** `gemini-embedding-2`
   - *Action Item / Confirmation for Hemanth:* Confirm API endpoint name: `models/text-embedding-004` vs upcoming `gemini-embedding-2`.
3. **Backend Framework:**
   - **SOW:** Simple Python scripts / Flask
   - **Finalized Architecture:** `FastAPI (Python)`
   - *Decision:* FastAPI is implemented for clean async support and lightweight endpoints.
4. **Vector Database:**
   - **SOW:** ChromaDB or Qdrant
   - **Finalized Architecture:** `ChromaDB (Local)`
   - *Decision:* ChromaDB local persistence requires zero external network infrastructure and runs standalone.

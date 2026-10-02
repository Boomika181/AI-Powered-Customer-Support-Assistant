"""
src/main.py

FastAPI Application Entry Point.
Phase 0 Status: Basic health and configuration check.
Phase 1/2: Endpoints for transcription streaming, RAG suggestions, and web dashboard.
"""

import os
from pathlib import Path
from fastapi import FastAPI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="AI-Powered Customer Support Assistant (POC)",
    description="Localhost Proof of Concept - Phase 0 Setup",
    version="0.1.0"
)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "phase": "Phase 0 (Setup & Document Preparation)",
        "message": "AI-Powered Customer Support Assistant POC backend is running.",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    import chromadb

    chroma_dir = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
    col_name = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")
    
    db_ok = False
    chunk_count = 0
    if chroma_dir.exists():
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            col = client.get_collection(col_name)
            chunk_count = col.count()
            db_ok = chunk_count > 0
        except Exception:
            db_ok = False

    return {
        "status": "healthy" if db_ok else "setup_required",
        "chromadb_initialized": db_ok,
        "indexed_chunks": chunk_count,
        "assemblyai_configured": bool(os.getenv("ASSEMBLYAI_API_KEY")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=5000, reload=True)

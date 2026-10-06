#!/usr/bin/env python3
"""
scripts/ingest_documents.py

Document Ingestion Pipeline for Technical Support Knowledge Base.
1. Ingests Sample_Technical_Documentation_Pack_SYNTHETIC.pdf (PDF).
2. Ingests Kestrel_Machinery_Support_Knowledge_Base_SYNTHETIC.docx (DOCX).
3. Generates 768-dimensional vector embeddings using Gemini Embedding 2 (`gemini-embedding-2`).
4. Strict consistency: No fallback to local models.
5. Upserts chunks, embeddings, and rich metadata into local ChromaDB persistent collection.
6. Saves processed chunks to data/processed/chunks.json for auditability.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.document_ingestion import ingest_document, DEFAULT_COLLECTION, DEFAULT_CHROMA_DIR
from src.embeddings import get_embedding_model_name, get_embedding_dimension

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf"
DOCX_PATH = PROJECT_ROOT / "data" / "raw" / "Kestrel_Machinery_Support_Knowledge_Base_SYNTHETIC.docx"
PROCESSED_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def ingest_all():
    print("=" * 70)
    print("AI-POWERED CUSTOMER SUPPORT ASSISTANT — DOCUMENT INGESTION")
    print("=" * 70)
    print(f"Embedding Model : {get_embedding_model_name()}")
    print(f"Vector Dimension: {get_embedding_dimension()}")
    print(f"Collection Name : {DEFAULT_COLLECTION}")
    print(f"Vector DB Path  : {DEFAULT_CHROMA_DIR}")
    print("-" * 70)

    # 1. Ingest PDF
    if not PDF_PATH.exists():
        print(f"[FAIL] PDF document not found at: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/2] Ingesting PDF Document: {PDF_PATH.name}...")
    pdf_res = ingest_document(PDF_PATH)
    print(f"  -> Inserted {pdf_res['chunks_inserted']} chunks from {pdf_res.get('pages', 0)} pages.")

    # 2. Ingest DOCX
    if not DOCX_PATH.exists():
        print(f"[FAIL] DOCX document not found at: {DOCX_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[2/2] Ingesting DOCX Document: {DOCX_PATH.name}...")
    docx_res = ingest_document(DOCX_PATH)
    print(f"  -> Inserted {docx_res['chunks_inserted']} chunks from {docx_res.get('paragraphs', 0)} paragraphs and {docx_res.get('tables', 0)} tables.")

    # 3. Export audit JSON
    import chromadb
    client = chromadb.PersistentClient(path=str(DEFAULT_CHROMA_DIR))
    col = client.get_collection(DEFAULT_COLLECTION)
    all_chunks = col.get()

    exported = []
    for cid, doc, meta in zip(all_chunks["ids"], all_chunks["documents"], all_chunks["metadatas"]):
        exported.append({
            "chunk_id": cid,
            "text": doc,
            **meta
        })

    PROCESSED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2)
    print(f"\n-> Exported {len(exported)} total chunks to {PROCESSED_JSON_PATH}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Total chunks in ChromaDB: {col.count()} (All embedded with {get_embedding_model_name()})")
    print("=" * 70)
    return col.count()


if __name__ == "__main__":
    try:
        ingest_all()
    except Exception as exc:
        print(f"\n[ERROR] Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)

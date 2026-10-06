#!/usr/bin/env python3
"""
scripts/verify_chromadb.py

ChromaDB Verification Script for Phase 0:
1. Connects to local persistent ChromaDB instance.
2. Loads collection 'technical_documentation'.
3. Displays total stored chunk count.
4. Inspects and prints sample stored metadata fields.
5. Runs a test similarity search to verify embedding retrieval and distance scoring.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")


def verify_chromadb():
    import chromadb

    print("=" * 65)
    print("Phase 0 - ChromaDB Verification")
    print("=" * 65)
    print(f"Chroma DB Path:  {CHROMA_DIR}")
    print(f"Collection Name: {COLLECTION_NAME}")

    if not CHROMA_DIR.exists():
        print(f"[FAIL] ChromaDB directory does not exist at {CHROMA_DIR}")
        print("Please run: python scripts/ingest_documents.py")
        return False

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing_collections = [col.name for col in client.list_collections()]
    print(f"Existing Collections: {existing_collections}")

    if COLLECTION_NAME not in existing_collections:
        print(f"[FAIL] Collection '{COLLECTION_NAME}' not found.")
        print("Please run: python scripts/ingest_documents.py")
        return False

    from src.embeddings import GeminiEmbeddingFunction

    collection = client.get_collection(COLLECTION_NAME, embedding_function=GeminiEmbeddingFunction())
    chunk_count = collection.count()
    print(f"\n[1] Chunk Count: {chunk_count}")
    if chunk_count == 0:
        print("[FAIL] Collection is empty.")
        return False

    # 2. Inspect sample records and metadata
    print("\n[2] Sample Metadata Inspection (First 3 items):")
    sample_records = collection.peek(limit=3)
    for idx, (doc_id, doc_text, meta) in enumerate(zip(
        sample_records["ids"],
        sample_records["documents"],
        sample_records["metadatas"]
    )):
        print(f"\n--- Item {idx + 1} (ID: {doc_id}) ---")
        print(f"  Source:             {meta.get('source')}")
        print(f"  Page:               {meta.get('page')}")
        print(f"  Document Reference: {meta.get('document_reference')}")
        print(f"  Section:            {meta.get('section')}")
        print(f"  Category:           {meta.get('category')}")
        print(f"  Machine:            {meta.get('machine')}")
        print(f"  Text Excerpt:       {doc_text[:120].strip()}...")

    # 3. Test similarity search
    test_query = "What is the compressed-air working pressure for WP-400?"
    print(f"\n[3] Test Similarity Search:")
    print(f"  Query: '{test_query}'")
    results = collection.query(query_texts=[test_query], n_results=2)

    for i, (ret_id, ret_doc, ret_meta, ret_dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(results["ids"][0])
    )):
        print(f"\n  Match {i + 1}: [Distance: {ret_dist:.4f}] ID: {ret_id}")
        print(f"    Doc Ref:  {ret_meta.get('document_reference')} (Page {ret_meta.get('page')})")
        print(f"    Category: {ret_meta.get('category')} | Machine: {ret_meta.get('machine')}")
        print(f"    Snippet:  {ret_doc[:140].strip()}...")

    print("\n" + "=" * 65)
    print("PHASE 0 CHROMADB VERIFICATION: PASS")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    success = verify_chromadb()
    sys.exit(0 if success else 1)

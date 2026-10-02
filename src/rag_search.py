"""
src/rag_search.py

Vector Search and Document Retrieval Module using ChromaDB.
Core infrastructure verified in Phase 0; full retrieval orchestration in Phase 1.

Phase 0 Status:
- Local ChromaDB collection 'technical_documentation' populated with 34 chunks and metadata.
- Tested similarity search in `scripts/verify_chromadb.py`.

Phase 1 Responsibilities:
- Query expansion and filtering by auto-detected category and machine.
- Top-k retrieval of manual chunks with relevance thresholds.
- Formatting context chunks for LLM suggestion prompt.
"""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import chromadb
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")


def query_knowledge_base(
    query_text: str,
    n_results: int = 3,
    category_filter: Optional[str] = None,
    machine_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant technical manual chunks from ChromaDB.
    Available for testing in Phase 0 / Phase 1.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"ChromaDB not initialized at {CHROMA_DIR}. Run ingest_documents.py first.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    where_filter = None
    if category_filter and machine_filter:
        where_filter = {"$and": [{"category": category_filter}, {"machine": machine_filter}]}
    elif category_filter:
        where_filter = {"category": category_filter}
    elif machine_filter:
        where_filter = {"machine": machine_filter}

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter
    )

    retrieved = []
    if results and "documents" in results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append({
                "document": doc,
                "metadata": meta
            })
    return retrieved

"""
src/rag_search.py

Vector Search and Document Retrieval Module using ChromaDB and Gemini Embedding 2.
Provides semantic retrieval of technical support documents from local ChromaDB collections.

Key Principles:
- Query embedding generated using Google Gemini Embedding 2 (`gemini-embedding-2`, 768 dimensions).
- Strict dimension consistency with stored document vectors.
- Latency isolation: measures embedding latency, ChromaDB retrieval latency, and total latency separately.
- Returns rich metadata (source document, file type, page, section, category, machine).
- Preserves backward compatibility with Phase 0 `query_knowledge_base(...)` API.
- Implements `search_knowledge_base(...)` returning distance and normalized similarity score.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import chromadb
from dotenv import load_dotenv

from src.embeddings import (
    embed_query,
    GeminiEmbeddingFunction,
    get_embedding_dimension,
    get_embedding_model_name,
    EmbeddingError
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("rag_search")

DEFAULT_CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    category_filter: Optional[str] = None,
    machine_filter: Optional[str] = None,
    collection_name: Optional[str] = None,
    chroma_dir: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search across technical documentation chunks in ChromaDB.

    Args:
        query: User or agent question/transcript text. Must be non-empty.
        top_k: Number of most relevant chunks to return (default: 5).
        category_filter: Optional filter by SOW category (e.g. 'Machine Operation Issues').
        machine_filter: Optional filter by machine model (e.g. 'WP-400', 'WR-700').
        collection_name: Optional ChromaDB collection name.
        chroma_dir: Optional path to ChromaDB persistent storage.
        api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY from environment.

    Returns:
        List[Dict[str, Any]] containing for each match:
        - text: Chunk content text
        - document: Alias to chunk content text
        - metadata: Rich chunk metadata (source_file, file_type, category, machine, page, section)
        - distance: Raw distance score from ChromaDB
        - similarity_score: Normalized similarity score in [0.0, 1.0] calculated as:
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
        - embedding_latency_ms: Milliseconds spent generating query embedding via Gemini
        - retrieval_latency_ms: Milliseconds spent performing vector search in ChromaDB
        - total_latency_ms: Total search duration in milliseconds

    Raises:
        ValueError: If query is empty or top_k <= 0.
        FileNotFoundError: If the specified ChromaDB directory does not exist.
        EmbeddingError: If Gemini Embedding 2 fails (never falls back to MiniLM).
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer greater than 0.")

    target_dir = Path(chroma_dir or DEFAULT_CHROMA_DIR).resolve()
    target_collection_name = collection_name or DEFAULT_COLLECTION

    if not target_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at: {target_dir}. "
            "Please run document ingestion before searching."
        )

    client = chromadb.PersistentClient(path=str(target_dir))
    existing_collections = [c.name for c in client.list_collections()]
    if target_collection_name not in existing_collections:
        logger.warning("Collection '%s' does not exist in ChromaDB at %s", target_collection_name, target_dir)
        return []

    collection = client.get_collection(target_collection_name)
    total_docs = collection.count()
    if total_docs == 0:
        logger.warning("Collection '%s' is empty (0 documents).", target_collection_name)
        return []

    # Adjust top_k if collection has fewer items
    query_limit = min(top_k, total_docs)

    # Build metadata filter if specified
    where_filter = None
    if category_filter and machine_filter:
        where_filter = {"$and": [{"category": category_filter}, {"machine": machine_filter}]}
    elif category_filter:
        where_filter = {"category": category_filter}
    elif machine_filter:
        where_filter = {"machine": machine_filter}

    # STRICT REQUIREMENT 3 & 10: Generate query vector with Gemini Embedding 2 and record latency
    t_embed_start = time.perf_counter()
    query_vector = embed_query(query.strip(), api_key=api_key)
    embedding_latency_ms = round((time.perf_counter() - t_embed_start) * 1000, 2)

    # Search ChromaDB using the 768-dim query vector and record retrieval latency
    t_retrieve_start = time.perf_counter()
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=query_limit,
            where=where_filter
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        raise
    retrieval_latency_ms = round((time.perf_counter() - t_retrieve_start) * 1000, 2)
    total_latency_ms = round(embedding_latency_ms + retrieval_latency_ms, 2)

    retrieved: List[Dict[str, Any]] = []
    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            # Distance in default ChromaDB L2 space ranges from 0.0 upwards
            # Convert distance to normalized similarity score [0.0, 1.0]
            # Formula: max(0.0, 1.0 - (dist / 2.0))
            sim_score = round(max(0.0, 1.0 - (float(dist) / 2.0)), 4)

            retrieved.append({
                "text": doc_text,
                "document": doc_text,  # alias for backward compatibility
                "metadata": meta,
                "distance": round(float(dist), 4),
                "similarity_score": sim_score,
                "embedding_latency_ms": embedding_latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "total_latency_ms": total_latency_ms
            })

    logger.debug(
        "Search completed: embedding=%.2fms, retrieval=%.2fms, total=%.2fms",
        embedding_latency_ms, retrieval_latency_ms, total_latency_ms
    )
    return retrieved


def query_knowledge_base(
    query_text: str,
    n_results: int = 3,
    category_filter: Optional[str] = None,
    machine_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant technical manual chunks from ChromaDB.
    Maintained for full backward compatibility with Phase 0 API callers.

    Returns:
        List of dicts with {"document": str, "metadata": dict}
    """
    matches = search_knowledge_base(
        query=query_text,
        top_k=n_results,
        category_filter=category_filter,
        machine_filter=machine_filter
    )
    return [{"document": m["text"], "metadata": m["metadata"]} for m in matches]

"""
src/embeddings.py

Gemini Embedding 2 Vector Generation Module.
Provides document and query embedding generation using Google's stable
Gemini Embedding 2 (`gemini-embedding-2`) model via the google-genai SDK.

Key Principles:
- Uses ONLY `gemini-embedding-2`. No local fallback (no MiniLM fallback).
- Configurable via `GEMINI_EMBEDDING_MODEL` environment variable. Default: `gemini-embedding-2`.
- Fixed, explicit dimensionality: 768 dimensions (`output_dimensionality=768`).
- Strict dimension mismatch protection.
- High-performance batching: encodes multiple document chunks in batched requests.
- ChromaDB EmbeddingFunction protocol compatibility for seamless collection integration.
"""

import os
import time
import logging
from typing import List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from google import genai
from google.genai import types
import chromadb

logger = logging.getLogger("embeddings")

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768


class EmbeddingError(RuntimeError):
    """Raised when Gemini Embedding 2 generation fails."""
    pass


class DimensionMismatchError(EmbeddingError):
    """Raised when an embedding vector dimensionality does not match the configured dimension."""
    pass


def get_embedding_model_name() -> str:
    """Returns the configured Gemini embedding model name (defaults to 'gemini-embedding-2')."""
    return os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_embedding_dimension() -> int:
    """Returns the configured embedding dimensionality (defaults to 768)."""
    dim_str = os.getenv("GEMINI_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)).strip()
    try:
        return int(dim_str)
    except ValueError:
        return DEFAULT_EMBEDDING_DIMENSION


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initializes and returns a Google GenAI Client using the configured API key.
    Raises ValueError if no valid key is found.
    """
    resolved_key = (
        api_key
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    if not resolved_key or resolved_key.startswith("your_"):
        raise ValueError("GEMINI_API_KEY is required for Gemini Embedding 2. None was found in environment or parameters.")

    return genai.Client(api_key=resolved_key)


def embed_texts(
    texts: List[str],
    model: Optional[str] = None,
    output_dimensionality: Optional[int] = None,
    api_key: Optional[str] = None,
    batch_size: int = 50
) -> List[List[float]]:
    """
    Generates embeddings for a list of text strings using Gemini Embedding 2.

    Args:
        texts: List of text strings to embed.
        model: Model identifier. Defaults to GEMINI_EMBEDDING_MODEL or 'gemini-embedding-2'.
        output_dimensionality: Target vector dimension. Defaults to 768.
        api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY.
        batch_size: Number of chunks per API batch call (default: 50).

    Returns:
        List[List[float]]: List of vector embeddings, each having length `output_dimensionality`.

    Raises:
        ValueError: If texts list is empty or contains non-string elements.
        DimensionMismatchError: If any returned vector does not match the required dimensionality.
        EmbeddingError: If the Gemini API call fails. Never falls back to a local model.
    """
    if not texts:
        return []

    target_model = model or get_embedding_model_name()
    target_dim = output_dimensionality if output_dimensionality is not None else get_embedding_dimension()
    client = get_gemini_client(api_key)

    all_embeddings: List[List[float]] = []

    # Process in batches to stay within API payload recommendations
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        contents = [
            types.Content(parts=[types.Part.from_text(text=t)])
            for t in batch
        ]

        try:
            logger.debug(
                "Calling Gemini embed_content (model=%s, dim=%d, batch_count=%d)",
                target_model, target_dim, len(batch)
            )
            config = types.EmbedContentConfig(output_dimensionality=target_dim)
            response = client.models.embed_content(
                model=target_model,
                contents=contents,
                config=config
            )
        except Exception as exc:
            err_msg = f"Gemini Embedding 2 call failed for model '{target_model}': {exc}"
            logger.error(err_msg)
            # STRICT REQUIREMENT 6: No fallback embedding. Preserve original exception.
            raise EmbeddingError(err_msg) from exc

        if not response or not response.embeddings:
            raise EmbeddingError(f"Gemini Embedding 2 returned empty response for model '{target_model}'")

        if len(response.embeddings) != len(batch):
            raise EmbeddingError(
                f"Gemini Embedding 2 returned {len(response.embeddings)} embeddings for {len(batch)} input items."
            )

        for idx, emb in enumerate(response.embeddings):
            values = emb.values
            if len(values) != target_dim:
                # STRICT REQUIREMENT 4 & 7: Dimension mismatch protection
                raise DimensionMismatchError(
                    f"Embedding dimension mismatch: expected {target_dim}, but model '{target_model}' "
                    f"returned vector of length {len(values)} for item index {i + idx}."
                )
            all_embeddings.append(values)

    return all_embeddings


def embed_text(
    text: str,
    model: Optional[str] = None,
    output_dimensionality: Optional[int] = None,
    api_key: Optional[str] = None
) -> List[float]:
    """
    Generates an embedding vector for a single text string using Gemini Embedding 2.
    """
    if not text or not text.strip():
        raise ValueError("Text to embed cannot be empty.")

    embeddings = embed_texts(
        texts=[text],
        model=model,
        output_dimensionality=output_dimensionality,
        api_key=api_key,
        batch_size=1
    )
    return embeddings[0]


def embed_query(
    query: str,
    model: Optional[str] = None,
    output_dimensionality: Optional[int] = None,
    api_key: Optional[str] = None
) -> List[float]:
    """
    Generates an embedding vector for a customer search query using Gemini Embedding 2.
    Ensures query vectors match document vector dimensionality (768).
    """
    return embed_text(
        text=query,
        model=model,
        output_dimensionality=output_dimensionality,
        api_key=api_key
    )


def embed_documents(
    documents: List[str],
    model: Optional[str] = None,
    output_dimensionality: Optional[int] = None,
    api_key: Optional[str] = None,
    batch_size: int = 50
) -> List[List[float]]:
    """
    Generates embedding vectors for a list of document chunks using Gemini Embedding 2.
    """
    return embed_texts(
        texts=documents,
        model=model,
        output_dimensionality=output_dimensionality,
        api_key=api_key,
        batch_size=batch_size
    )


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    ChromaDB-compatible EmbeddingFunction using Gemini Embedding 2.
    Ensures ChromaDB collections use Gemini Embedding 2 natively for queries or documents.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        self.model = model or get_embedding_model_name()
        self.output_dimensionality = output_dimensionality if output_dimensionality is not None else get_embedding_dimension()
        self.api_key = api_key

    @staticmethod
    def name() -> str:
        return "gemini-embedding-2"

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "output_dimensionality": self.output_dimensionality
        }

    @classmethod
    def build_from_config(cls, config: dict) -> "GeminiEmbeddingFunction":
        return cls(
            model=config.get("model"),
            output_dimensionality=config.get("output_dimensionality")
        )

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        """
        Embeds a sequence of documents/queries using Gemini Embedding 2.
        """
        return embed_documents(
            documents=list(input),
            model=self.model,
            output_dimensionality=self.output_dimensionality,
            api_key=self.api_key
        )

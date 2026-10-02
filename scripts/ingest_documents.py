#!/usr/bin/env python3
"""
scripts/ingest_documents.py

Document Ingestion Pipeline for Phase 0:
1. Extracts text and structural sections from Sample_Technical_Documentation_Pack_SYNTHETIC.pdf.
2. Chunks document into semantically meaningful sections and Q&A items.
3. Extracts and associates rich metadata:
   - source document
   - page number
   - document reference
   - section title
   - support category (Machine Operation Issues, Maintenance & Parts, Technical Troubleshooting, etc.)
   - machine model (WP-400, SC-900, MC-250, GC-120, All)
4. Generates embeddings:
   - Uses Gemini embedding API (models/text-embedding-004) if GEMINI_API_KEY is available in .env.
   - Falls back gracefully to ChromaDB local ONNX embedding if GEMINI_API_KEY is not configured yet,
     allowing offline testing and full pipeline validation without blocker.
5. Upserts chunks, embeddings, and metadata into local ChromaDB persistent collection.
6. Saves processed chunks to data/processed/chunks.json for auditability.
"""

import os
import sys
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf"
PROCESSED_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")


def extract_and_chunk_pdf(pdf_path: Path):
    """
    Extracts text from PDF and segments into meaningful chunks preserving metadata.
    """
    import pymupdf

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    chunks = []

    # 1. Page 1: Overview & Document Map
    p1_text = doc[0].get_text()
    p1_clean_lines = [
        line.strip()
        for line in p1_text.splitlines()
        if "SYNTHETIC SAMPLE DOCUMENTATION" not in line
        and not re.match(r"^Page \d+$", line.strip())
        and line.strip()
    ]
    chunks.append({
        "chunk_id": "DOC-MAP-01",
        "source": pdf_path.name,
        "page": 1,
        "section": "Document Overview & Machine Reference Map",
        "document_reference": "DOC-MAP-01",
        "category": "General Reference",
        "machine": "All",
        "text": "\n".join(p1_clean_lines)
    })

    # 2. Extract lines across pages 2-10 with page tracking
    full_doc_lines = []
    for p_idx in range(1, len(doc)):
        p_num = p_idx + 1
        raw_text = doc[p_idx].get_text()
        for line in raw_text.splitlines():
            s = line.strip()
            if not s:
                continue
            if "SYNTHETIC SAMPLE DOCUMENTATION" in s:
                continue
            if re.match(r"^Page \d+$", s):
                continue
            full_doc_lines.append((p_num, s))

    # Find line indices containing 'Doc ref:'
    doc_ref_indices = [idx for idx, (p, l) in enumerate(full_doc_lines) if l.startswith("Doc ref:")]

    for i, ref_idx in enumerate(doc_ref_indices):
        page_num, ref_line = full_doc_lines[ref_idx]
        m = re.match(r"Doc ref:\s*([^\s|]+)\s*\|\s*Category:\s*([^|\n]+)\s*\|\s*Machine:\s*([^\n]+)", ref_line)
        if not m:
            continue

        doc_ref = m.group(1).strip()
        category = m.group(2).strip()
        machine = m.group(3).strip()

        # Section title is typically the line immediately before Doc ref
        sec_title = full_doc_lines[ref_idx - 1][1] if ref_idx > 0 else doc_ref

        # Preceding contextual headers (e.g., 'PART 1...', '1.1 WP-400...')
        headers = []
        start_back = max(0, ref_idx - 3)
        for b_idx in range(start_back, ref_idx - 1):
            prev_line = full_doc_lines[b_idx][1]
            if prev_line.startswith("PART ") or re.match(r"^\d\.\d\s", prev_line):
                headers.append(prev_line)

        # End index is before next section's title and header
        if i + 1 < len(doc_ref_indices):
            next_ref_idx = doc_ref_indices[i + 1]
            next_start = next_ref_idx - 1
            while next_start > ref_idx and (
                full_doc_lines[next_start - 1][1].startswith("PART ")
                or re.match(r"^\d\.\d\s", full_doc_lines[next_start - 1][1])
            ):
                next_start -= 1
            end_idx = next_start
        else:
            end_idx = len(full_doc_lines)

        content_lines = headers + [sec_title, ref_line] + [l for p, l in full_doc_lines[ref_idx + 1:end_idx]]
        chunk_text = "\n".join(content_lines)

        # Append section-level chunk
        chunks.append({
            "chunk_id": doc_ref,
            "source": pdf_path.name,
            "page": page_num,
            "section": sec_title,
            "document_reference": doc_ref,
            "category": category,
            "machine": machine,
            "text": chunk_text
        })

    # 3. Dedicated granular extraction for Part 5 FAQs (Page 10)
    p10_text = doc[9].get_text()
    q_pattern = re.findall(r"\[([^\]]+)\]\s*Q:\s*(.+?)\nA:\s*(.+?)(?=\n\[|\Z)", p10_text, re.DOTALL)
    for q_idx, (faq_machine, q, a) in enumerate(q_pattern, start=1):
        faq_machine = faq_machine.strip()
        q_text = q.strip().replace("\n", " ")
        a_text = a.strip().replace("\n", " ")
        faq_id = f"FAQ-{faq_machine.replace('-', '')}-{q_idx:02d}"

        # Categorize FAQ logically based on machine and question
        if "E-101" in q_text or "extraction" in q_text or "water" in q_text:
            faq_category = "Machine Operation Issues"
        elif "blade" in q_text or "consumables" in q_text or "oil" in q_text or "warranty" in q_text:
            faq_category = "Maintenance & Parts"
        else:
            faq_category = "Technical Troubleshooting"

        chunks.append({
            "chunk_id": faq_id,
            "source": pdf_path.name,
            "page": 10,
            "section": f"FAQ: {q_text[:40]}...",
            "document_reference": "FAQ-ALL-5",
            "category": faq_category,
            "machine": faq_machine,
            "text": f"[{faq_machine}] Question: {q_text}\nAnswer: {a_text}"
        })

    return chunks


def get_embeddings(texts: list[str], gemini_key: str | None) -> list[list[float]] | None:
    """
    Generates embeddings using Gemini embedding API if key is present,
    or returns None to let ChromaDB generate local embeddings.
    """
    if not gemini_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        embeddings = []
        print(f"Generating embeddings for {len(texts)} chunks using Gemini ({EMBEDDING_MODEL})...")
        for i, text in enumerate(texts):
            # Clean model name if passed with 'models/'
            model_name = EMBEDDING_MODEL.replace("models/", "")
            res = client.models.embed_content(
                model=model_name,
                contents=text
            )
            embeddings.append(res.embeddings[0].values)
            if (i + 1) % 10 == 0 or (i + 1) == len(texts):
                print(f"  Embedded {i + 1}/{len(texts)} chunks...")
        return embeddings
    except Exception as e:
        print(f"[WARNING] Gemini embedding call failed ({e}). Falling back to local ChromaDB embedding.")
        return None


def ingest():
    """
    Executes the ingestion process and updates ChromaDB.
    """
    import chromadb

    print("=" * 60)
    print("AI-Powered Customer Support Assistant - Document Ingestion")
    print("=" * 60)
    print(f"PDF Source: {PDF_PATH}")
    print(f"Vector DB Path: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        print(f"Gemini API Key: Available (Masked: {gemini_key[:4]}...{gemini_key[-4:]})")
    else:
        print("Gemini API Key: Not set in .env. Using ChromaDB local embedding engine.")

    # Step 1: Extract & Chunk
    print("\n[Step 1/4] Extracting text and parsing semantic chunks...")
    chunks = extract_and_chunk_pdf(PDF_PATH)
    print(f"-> Successfully extracted {len(chunks)} semantic chunks.")

    # Save processed chunks to JSON
    PROCESSED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"-> Saved processed chunks to {PROCESSED_JSON_PATH}")

    # Step 2: Prepare vectors
    print("\n[Step 2/4] Generating vector embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts, gemini_key)
    if embeddings:
        print("-> Using Gemini API embeddings.")
    else:
        print("-> Using local ChromaDB default embeddings.")

    # Step 3: Initialize ChromaDB
    print("\n[Step 3/4] Initializing local ChromaDB persistent collection...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset or retrieve collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Vantor Industrial synthetic technical manuals"}
    )

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{
        "source": c["source"],
        "page": c["page"],
        "section": c["section"],
        "document_reference": c["document_reference"],
        "category": c["category"],
        "machine": c["machine"]
    } for c in chunks]

    # Step 4: Upsert chunks
    print("\n[Step 4/4] Upserting chunks into ChromaDB...")
    if embeddings:
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    else:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    total_count = collection.count()
    print(f"\n[SUCCESS] Document ingestion complete!")
    print(f"Total chunks stored in collection '{COLLECTION_NAME}': {total_count}")

    # Quick verification search
    test_query = "What is the compressed-air working pressure for WP-400?"
    print(f"\n[Verification Query]: '{test_query}'")
    results = collection.query(query_texts=[test_query], n_results=2)
    for idx, (doc_text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\nResult {idx + 1}:")
        print(f"  Reference: {meta.get('document_reference')} | Machine: {meta.get('machine')} | Page: {meta.get('page')}")
        print(f"  Category:  {meta.get('category')}")
        print(f"  Snippet:   {doc_text[:120].strip()}...")

    print("=" * 60)
    return total_count


if __name__ == "__main__":
    try:
        ingest()
    except Exception as exc:
        print(f"\n[ERROR] Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)

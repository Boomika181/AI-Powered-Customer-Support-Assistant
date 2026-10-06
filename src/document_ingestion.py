"""
src/document_ingestion.py

Knowledge Base Document Ingestion Pipeline for PDF and DOCX.
Extracts, cleans, chunks, embeds, and stores technical support documents in ChromaDB.

Supported Formats:
- PDF (.pdf) via PyMuPDF (fitz)
- Word Documents (.docx) via python-docx

Key Principles:
- Preserves technical identifiers (E-102, WP4-FL-DE1, WR7-CL-12, R-101, etc.) and engineering units.
- Extracts DOCX tables into structured, searchable text representations.
- Associates page numbers (PDF) and structural sections/headings with every chunk.
- Deterministic deduplication: re-ingesting a document cleanly replaces previous chunks for that file.
- Embedding Generation: Uses Google Gemini Embedding 2 (`gemini-embedding-2`, 768 dimensions).
- Strict consistency: No fallback to local models.
"""

import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import chromadb
from dotenv import load_dotenv

from src.embeddings import (
    embed_documents,
    GeminiEmbeddingFunction,
    get_embedding_dimension,
    get_embedding_model_name,
    EmbeddingError
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("document_ingestion")

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")
DEFAULT_CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")


@dataclass
class UnifiedChunk:
    """Unified internal representation of an extracted document chunk."""
    chunk_id: str
    source_file: str
    file_type: str  # "pdf" | "docx"
    chunk_index: int
    title: str
    section: str
    category: str
    machine: str
    text: str
    page: int = -1  # 1-indexed for PDF, -1 for DOCX if page unavailable
    document_reference: str = ""


class IngestionError(Exception):
    """Raised when document ingestion fails."""
    pass


# ==============================================================================
# 1. Validation & Cleaning
# ==============================================================================

def validate_document_file(file_path: Union[str, Path]) -> Path:
    """
    Validates document path, extension, and file existence.
    Rejects unsupported formats with an informative ValueError.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Target path is not a regular file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document format '{suffix}'. Supported formats are: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Document file is empty (0 bytes): {path.name}")

    return path


def clean_text(text: str) -> str:
    """
    Cleans extracted document text while strictly preserving technical identifiers.

    Why cleaning is performed:
    - Normalizes inconsistent whitespace, non-breaking spaces (\\xa0), and carriage returns.
    - Removes document artifacts such as running headers/footers (e.g. 'Page X of Y', synthetic notices)
      that dilute vector similarity and waste context space.
    - Preserves intact:
      * Error codes: E-102, E-101, E-205, E-310, E-410, R-101, W-10, L-05, H-02, P-11, G-03
      * Part numbers: WP4-FL-DE1, WP4-BL-300, WR7-CL-12, MF1-PW-27, SC9-WF-02, GC1-SW-6
      * Machine names: WP-400, WR-700, SC-900, MC-250, GC-120, SJ-650, MF-180, GL-300
      * Technical units: 6.0 bar, 1200 Pa, 4000 rpm, 16000 rpm, 3800 bar, 20 m/s
    """
    if not text:
        return ""

    # 1. Replace non-breaking spaces and line separators
    cleaned = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # 2. Filter out synthetic running headers/footers
    lines = []
    for line in cleaned.splitlines():
        trimmed = line.strip()
        if not trimmed:
            lines.append("")
            continue
        if "SYNTHETIC SAMPLE DOCUMENTATION - not for real equipment" in trimmed:
            continue
        if "SYNTHETIC TEST DATA" in trimmed and "NOT REAL" in trimmed:
            continue
        if re.match(r"^Page \d+(?: of \d+)?$", trimmed, re.IGNORECASE):
            continue
        if re.match(r"^Vantor Industrial.*?Page \d+$", trimmed, re.IGNORECASE):
            continue
        lines.append(trimmed)

    cleaned = "\n".join(lines)

    # 3. Collapse 3 or more consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 4. Collapse consecutive spaces on single lines (preserving newlines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    return cleaned.strip()


# ==============================================================================
# 2. PDF Extraction (PyMuPDF)
# ==============================================================================

def _extract_pdf(pdf_path: Path) -> List[UnifiedChunk]:
    """
    Extracts text from PDF page-by-page, preserving page boundaries and structure.
    Handles structured manuals with 'Doc ref:' as well as general PDF documents.
    """
    import pymupdf

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise IngestionError(f"Failed to open PDF document '{pdf_path.name}': {exc}") from exc

    if len(doc) == 0:
        raise ValueError(f"PDF document contains no pages: {pdf_path.name}")

    chunks: List[UnifiedChunk] = []
    source_name = pdf_path.name

    # Check whether PDF contains structured 'Doc ref:' tags
    sample_text = "\n".join([doc[i].get_text() for i in range(min(3, len(doc)))])
    has_doc_refs = "Doc ref:" in sample_text

    if has_doc_refs:
        # Structured technical manual extraction (matches Phase 0 logic while expanding metadata)
        # Page 1: Overview
        p1_text = clean_text(doc[0].get_text())
        chunks.append(UnifiedChunk(
            chunk_id="DOC-MAP-01",
            source_file=source_name,
            file_type="pdf",
            chunk_index=0,
            title="Document Overview & Machine Reference Map",
            section="Document Overview & Machine Reference Map",
            category="General Reference",
            machine="All",
            text=p1_text,
            page=1,
            document_reference="DOC-MAP-01"
        ))

        # Pages 2..N extraction
        full_doc_lines: List[tuple[int, str]] = []
        for p_idx in range(1, len(doc)):
            p_num = p_idx + 1
            raw_text = doc[p_idx].get_text()
            for line in raw_text.splitlines():
                s = line.strip()
                if not s or "SYNTHETIC SAMPLE DOCUMENTATION" in s or re.match(r"^Page \d+$", s):
                    continue
                full_doc_lines.append((p_num, s))

        doc_ref_indices = [idx for idx, (p, l) in enumerate(full_doc_lines) if l.startswith("Doc ref:")]

        chunk_idx_counter = 1
        for i, ref_idx in enumerate(doc_ref_indices):
            page_num, ref_line = full_doc_lines[ref_idx]
            m = re.match(r"Doc ref:\s*([^\s|]+)\s*\|\s*Category:\s*([^|\n]+)\s*\|\s*Machine:\s*([^\n]+)", ref_line)
            if not m:
                continue

            doc_ref = m.group(1).strip()
            category = m.group(2).strip()
            machine = m.group(3).strip()

            sec_title = full_doc_lines[ref_idx - 1][1] if ref_idx > 0 else doc_ref

            headers = []
            start_back = max(0, ref_idx - 3)
            for b_idx in range(start_back, ref_idx - 1):
                prev_line = full_doc_lines[b_idx][1]
                if prev_line.startswith("PART ") or re.match(r"^\d\.\d\s", prev_line):
                    headers.append(prev_line)

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
            chunk_text = clean_text("\n".join(content_lines))

            chunks.append(UnifiedChunk(
                chunk_id=doc_ref,
                source_file=source_name,
                file_type="pdf",
                chunk_index=chunk_idx_counter,
                title=sec_title,
                section=sec_title,
                category=category,
                machine=machine,
                text=chunk_text,
                page=page_num,
                document_reference=doc_ref
            ))
            chunk_idx_counter += 1

        # Check for dedicated Part 5 FAQ section (typically last page)
        last_page_idx = len(doc) - 1
        last_page_text = doc[last_page_idx].get_text()
        q_pattern = re.findall(r"\[([^\]]+)\]\s*Q:\s*(.+?)\nA:\s*(.+?)(?=\n\[|\Z)", last_page_text, re.DOTALL)
        for q_idx, (faq_machine, q, a) in enumerate(q_pattern, start=1):
            faq_machine = faq_machine.strip()
            q_text = q.strip().replace("\n", " ")
            a_text = a.strip().replace("\n", " ")
            faq_id = f"FAQ-{faq_machine.replace('-', '')}-{q_idx:02d}"

            if "E-101" in q_text or "extraction" in q_text or "water" in q_text:
                faq_cat = "Machine Operation Issues"
            elif "blade" in q_text or "consumables" in q_text or "oil" in q_text or "warranty" in q_text:
                faq_cat = "Maintenance & Parts"
            else:
                faq_cat = "Technical Troubleshooting"

            chunks.append(UnifiedChunk(
                chunk_id=faq_id,
                source_file=source_name,
                file_type="pdf",
                chunk_index=chunk_idx_counter,
                title=f"FAQ: {q_text[:40]}...",
                section=f"FAQ: {q_text[:40]}...",
                category=faq_cat,
                machine=faq_machine,
                text=clean_text(f"[{faq_machine}] Question: {q_text}\nAnswer: {a_text}"),
                page=last_page_idx + 1,
                document_reference="FAQ-ALL-5"
            ))
            chunk_idx_counter += 1

    else:
        # General unstructured PDF: page-aware windowed chunking
        chunk_idx = 0
        for p_idx, page in enumerate(doc):
            page_num = p_idx + 1
            raw_text = clean_text(page.get_text())
            if not raw_text:
                continue

            # Deterministic chunking: ~1000 characters with 150 character overlap
            window_size = 1000
            overlap = 150
            start = 0
            while start < len(raw_text):
                end = min(start + window_size, len(raw_text))
                # Break at newline or space if possible
                if end < len(raw_text):
                    last_space = raw_text.rfind(" ", start + window_size // 2, end)
                    if last_space != -1:
                        end = last_space

                chunk_str = raw_text[start:end].strip()
                if chunk_str:
                    c_id = f"{pdf_path.stem}_p{page_num}_{chunk_idx:03d}"
                    chunks.append(UnifiedChunk(
                        chunk_id=c_id,
                        source_file=source_name,
                        file_type="pdf",
                        chunk_index=chunk_idx,
                        title=f"{pdf_path.stem} (Page {page_num})",
                        section=f"Page {page_num}",
                        category="General Reference",
                        machine="All",
                        text=chunk_str,
                        page=page_num,
                        document_reference=f"P{page_num}-{chunk_idx}"
                    ))
                    chunk_idx += 1

                if end >= len(raw_text):
                    break
                start = end - overlap

    doc.close()
    if not chunks:
        raise ValueError(f"No extractable text found in PDF document: {pdf_path.name}")
    return chunks


# ==============================================================================
# 3. DOCX Extraction (python-docx)
# ==============================================================================

def _format_docx_table(table) -> str:
    """
    Converts a python-docx Table into structured, searchable markdown-style text.
    Preserves all column headers, cell relationships, and technical values.
    """
    if not table.rows:
        return ""

    headers = [cell.text.strip().replace("\n", " ") for cell in table.rows[0].cells]
    # Deduplicate repetitive columns caused by merged cells
    clean_headers = []
    prev = None
    for h in headers:
        clean_headers.append(h if h != prev or not h else "")
        prev = h

    lines = ["Table:"]
    lines.append("| " + " | ".join(clean_headers) + " |")

    for row in table.rows[1:]:
        row_vals = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def _extract_docx(docx_path: Path) -> List[UnifiedChunk]:
    """
    Extracts paragraphs, headings, and tables from a DOCX file in body order.
    Preserves tables as searchable text, links headings to context,
    and supports both structured 'Doc ref:' manuals and general DOCX files.
    """
    import docx
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    try:
        doc = docx.Document(str(docx_path))
    except Exception as exc:
        raise IngestionError(f"Failed to open DOCX document '{docx_path.name}': {exc}") from exc

    source_name = docx_path.name

    # Extract all body items in sequential document order
    body_items: List[tuple[str, str, str]] = []  # (type, content, style/tag)
    for child in doc.element.body:
        if child.tag.endswith("p"):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if text:
                style_name = p.style.name if p.style else "Normal"
                body_items.append(("p", text, style_name))
        elif child.tag.endswith("tbl"):
            t = Table(child, doc)
            table_text = _format_docx_table(t)
            if table_text:
                body_items.append(("tbl", table_text, "Table"))

    if not body_items:
        raise ValueError(f"DOCX document contains no readable paragraphs or tables: {docx_path.name}")

    chunks: List[UnifiedChunk] = []

    # Check whether the DOCX follows the structured 'Doc ref:' format
    doc_ref_indices = [
        idx for idx, (kind, content, _) in enumerate(body_items)
        if kind == "p" and "Doc ref:" in content
    ]

    if doc_ref_indices:
        # Structured DOCX manual parsing (e.g. Kestrel Machinery Support Knowledge Base)
        # 1. Overview chunk (everything before the first 'PART 1')
        first_ref_idx = doc_ref_indices[0]
        overview_end = first_ref_idx
        for b_idx in range(first_ref_idx):
            if body_items[b_idx][0] == "p" and body_items[b_idx][1].startswith("PART 1"):
                overview_end = b_idx
                break

        overview_texts = [item[1] for item in body_items[:overview_end]]
        if overview_texts:
            overview_content = clean_text("\n\n".join(overview_texts))
            chunks.append(UnifiedChunk(
                chunk_id=f"{docx_path.stem}_DOC_MAP",
                source_file=source_name,
                file_type="docx",
                chunk_index=0,
                title="Machines Covered & Document Map",
                section="Machines Covered & Document Map",
                category="General Reference",
                machine="All",
                text=overview_content,
                page=-1,
                document_reference="DOC-MAP-DOCX"
            ))

        chunk_counter = 1

        for i, ref_idx in enumerate(doc_ref_indices):
            ref_line = body_items[ref_idx][1]
            m = re.match(r"Doc ref:\s*([^\s|]+)\s*\|\s*Category:\s*([^|\n]+)\s*\|\s*Machine:\s*([^\n]+)", ref_line)
            if not m:
                continue

            doc_ref = m.group(1).strip()
            category = m.group(2).strip()
            machine = m.group(3).strip()

            # Find section title immediately preceding the Doc ref line
            sec_title = doc_ref
            for back_idx in range(ref_idx - 1, -1, -1):
                if body_items[back_idx][0] == "p":
                    candidate = body_items[back_idx][1]
                    if candidate.startswith("Section ") or "Part " in candidate or "PART " in candidate:
                        sec_title = candidate
                        break
                    elif not candidate.startswith("Doc ref:"):
                        sec_title = candidate
                        break

            # Find context headings preceding this section
            headers = []
            for back_idx in range(max(0, ref_idx - 4), ref_idx - 1):
                if body_items[back_idx][0] == "p":
                    cand = body_items[back_idx][1]
                    if cand.startswith("PART ") or re.match(r"^\d\.\d\s", cand):
                        headers.append(cand)

            # Determine end boundary: next doc_ref or end of document
            if i + 1 < len(doc_ref_indices):
                next_ref_idx = doc_ref_indices[i + 1]
                # Look backwards from next_ref_idx to exclude its title and headers
                end_idx = next_ref_idx
                for back_cand in range(next_ref_idx - 1, ref_idx, -1):
                    if body_items[back_cand][0] == "p":
                        btext = body_items[back_cand][1]
                        if btext.startswith("PART ") or re.match(r"^\d\.\d\s", btext) or btext.startswith("Section "):
                            end_idx = back_cand
                        else:
                            break
            else:
                end_idx = len(body_items)

            body_content_parts = [item[1] for item in body_items[ref_idx + 1:end_idx]]

            # Special case: Part 5 FAQ section with multiple Q&As
            if doc_ref == "FAQ-ALL-5":
                # Granularly extract each Q&A pair as a separate chunk
                current_faq_q = None
                current_faq_a = []
                current_faq_machine = "All"
                faq_counter = 1

                for item_kind, item_text, _ in body_items[ref_idx + 1:end_idx]:
                    q_match = re.match(r"\[([^\]]+)\]\s*Q:\s*(.+)", item_text)
                    if q_match:
                        if current_faq_q and current_faq_a:
                            full_q = current_faq_q
                            full_a = " ".join(current_faq_a)
                            faq_cat = "General Reference"
                            if "(Operation)" in full_a:
                                faq_cat = "Machine Operation Issues"
                            elif "(Parts)" in full_a:
                                faq_cat = "Maintenance & Parts"
                            elif "(Troubleshooting)" in full_a:
                                faq_cat = "Technical Troubleshooting"

                            chunks.append(UnifiedChunk(
                                chunk_id=f"FAQ-DOCX-{current_faq_machine.replace('-', '')}-{faq_counter:02d}",
                                source_file=source_name,
                                file_type="docx",
                                chunk_index=chunk_counter,
                                title=f"FAQ: {full_q[:40]}...",
                                section=f"FAQ: {full_q[:40]}...",
                                category=faq_cat,
                                machine=current_faq_machine,
                                text=clean_text(f"[{current_faq_machine}] Question: {full_q}\nAnswer: {full_a}"),
                                page=-1,
                                document_reference="FAQ-ALL-5"
                            ))
                            chunk_counter += 1
                            faq_counter += 1

                        current_faq_machine = q_match.group(1).strip()
                        current_faq_q = q_match.group(2).strip()
                        current_faq_a = []
                    elif item_text.startswith("A:"):
                        current_faq_a.append(item_text[2:].strip())
                    elif current_faq_q:
                        current_faq_a.append(item_text)

                if current_faq_q and current_faq_a:
                    full_q = current_faq_q
                    full_a = " ".join(current_faq_a)
                    faq_cat = "General Reference"
                    if "(Operation)" in full_a:
                        faq_cat = "Machine Operation Issues"
                    elif "(Parts)" in full_a:
                        faq_cat = "Maintenance & Parts"
                    elif "(Troubleshooting)" in full_a:
                        faq_cat = "Technical Troubleshooting"

                    chunks.append(UnifiedChunk(
                        chunk_id=f"FAQ-DOCX-{current_faq_machine.replace('-', '')}-{faq_counter:02d}",
                        source_file=source_name,
                        file_type="docx",
                        chunk_index=chunk_counter,
                        title=f"FAQ: {full_q[:40]}...",
                        section=f"FAQ: {full_q[:40]}...",
                        category=faq_cat,
                        machine=current_faq_machine,
                        text=clean_text(f"[{current_faq_machine}] Question: {full_q}\nAnswer: {full_a}"),
                        page=-1,
                        document_reference="FAQ-ALL-5"
                    ))
                    chunk_counter += 1

                continue

            # Standard section-level chunk
            full_section_lines = headers + [sec_title, ref_line] + body_content_parts
            chunk_text = clean_text("\n\n".join(full_section_lines))

            chunks.append(UnifiedChunk(
                chunk_id=f"DOCX-{doc_ref}",
                source_file=source_name,
                file_type="docx",
                chunk_index=chunk_counter,
                title=sec_title,
                section=sec_title,
                category=category,
                machine=machine,
                text=chunk_text,
                page=-1,
                document_reference=doc_ref
            ))
            chunk_counter += 1

    else:
        # General unstructured DOCX: segment by headings and paragraphs/tables
        current_heading = docx_path.stem
        current_paragraphs: List[str] = []
        chunk_idx = 0

        for kind, content, style in body_items:
            is_heading = kind == "p" and (
                "heading" in style.lower() or "title" in style.lower()
            )

            if is_heading:
                if current_paragraphs:
                    c_text = clean_text(f"{current_heading}\n\n" + "\n\n".join(current_paragraphs))
                    if c_text:
                        chunks.append(UnifiedChunk(
                            chunk_id=f"{docx_path.stem}_chk_{chunk_idx:03d}",
                            source_file=source_name,
                            file_type="docx",
                            chunk_index=chunk_idx,
                            title=current_heading,
                            section=current_heading,
                            category="General Reference",
                            machine="All",
                            text=c_text,
                            page=-1,
                            document_reference=f"DOCX-{chunk_idx}"
                        ))
                        chunk_idx += 1
                    current_paragraphs = []
                current_heading = content
            else:
                current_paragraphs.append(content)

        if current_paragraphs:
            c_text = clean_text(f"{current_heading}\n\n" + "\n\n".join(current_paragraphs))
            if c_text:
                chunks.append(UnifiedChunk(
                    chunk_id=f"{docx_path.stem}_chk_{chunk_idx:03d}",
                    source_file=source_name,
                    file_type="docx",
                    chunk_index=chunk_idx,
                    title=current_heading,
                    section=current_heading,
                    category="General Reference",
                    machine="All",
                    text=c_text,
                    page=-1,
                    document_reference=f"DOCX-{chunk_idx}"
                ))

    return chunks


# ==============================================================================
# 4. Public Ingestion API & ChromaDB Storage
# ==============================================================================

def ingest_document(
    file_path: Union[str, Path],
    collection_name: Optional[str] = None,
    chroma_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Ingests a knowledge document (.pdf or .docx) into the ChromaDB vector database.

    Workflow:
    1. Validates document path, extension, and non-empty status.
    2. Detects file type (.pdf or .docx) and executes format-specific extraction.
    3. Normalizes tables into searchable text representations.
    4. Cleans text while strictly preserving technical identifiers and units.
    5. Deduplicates: deletes any existing chunks in ChromaDB with the same source_file.
    6. Stores chunk texts, deterministic IDs, and rich metadata in ChromaDB.
    7. Generates 768-dimensional vector embeddings using Gemini Embedding 2 (gemini-embedding-2). No local fallback.

    Args:
        file_path: Absolute or relative path to .pdf or .docx document.
        collection_name: Optional ChromaDB collection name (defaults to CHROMA_COLLECTION_NAME).
        chroma_dir: Optional ChromaDB persistent directory (defaults to CHROMA_PERSIST_DIRECTORY).

    Returns:
        Dict[str, Any] containing ingestion statistics:
        - source_file: str
        - file_type: 'pdf' | 'docx'
        - pages / paragraphs / tables (format specific)
        - chunks_created: int
        - chunks_inserted: int
        - collection_name: str
    """
    valid_path = validate_document_file(file_path)
    file_type = valid_path.suffix.lower().lstrip(".")
    target_collection = collection_name or DEFAULT_COLLECTION
    target_chroma_dir = Path(chroma_dir or DEFAULT_CHROMA_DIR).resolve()

    logger.info("Starting ingestion of %s (%s) into collection '%s'", valid_path.name, file_type, target_collection)

    # Step 1: Extraction
    if file_type == "pdf":
        chunks = _extract_pdf(valid_path)
        import pymupdf
        with pymupdf.open(str(valid_path)) as doc:
            page_count = len(doc)
        stat_meta = {"pages": page_count}
    elif file_type == "docx":
        chunks = _extract_docx(valid_path)
        import docx
        doc = docx.Document(str(valid_path))
        stat_meta = {
            "paragraphs": len(doc.paragraphs),
            "tables": len(doc.tables)
        }
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    if not chunks:
        raise IngestionError(f"Extraction produced 0 chunks for {valid_path.name}")

    # Step 2: Connect to ChromaDB
    target_chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target_chroma_dir))
    embedding_fn = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=target_collection,
        embedding_function=embedding_fn,
        metadata={
            "description": "Customer Support Technical Documentation",
            "embedding_model": get_embedding_model_name(),
            "dimension": get_embedding_dimension()
        }
    )

    # Step 3: Deduplication — clean out previous records for this source_file
    source_filename = valid_path.name
    try:
        existing_records = collection.get(where={"source": source_filename})
        if existing_records and existing_records.get("ids"):
            logger.info("Removing %d existing chunks for %s to prevent duplication", len(existing_records["ids"]), source_filename)
            collection.delete(ids=existing_records["ids"])
    except Exception as exc:
        logger.debug("Deduplication lookup returned no records or failed safely: %s", exc)

    # Step 4: Prepare records for ChromaDB
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source": chunk.source_file,
            "source_file": chunk.source_file,
            "file_type": chunk.file_type,
            "chunk_index": chunk.chunk_index,
            "title": chunk.title,
            "section": chunk.section,
            "category": chunk.category,
            "machine": chunk.machine,
            "page": chunk.page,
            "document_reference": chunk.document_reference
        }
        for chunk in chunks
    ]

    # Step 5: Generate embeddings via Gemini Embedding 2 (768 dimensions)
    logger.info("Generating Gemini embeddings for %d chunks in %s...", len(documents), source_filename)
    embeddings = embed_documents(documents)

    # Step 6: Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    logger.info("Successfully inserted %d chunks for %s into '%s'", len(chunks), source_filename, target_collection)

    result = {
        "source_file": source_filename,
        "file_type": file_type,
        "chunks_created": len(chunks),
        "chunks_inserted": len(chunks),
        "collection_name": target_collection,
        "embedding_model": get_embedding_model_name(),
        "embedding_dimension": get_embedding_dimension(),
        **stat_meta
    }
    return result

"""
tests/test_rag_search.py

Unit tests for RAG Vector Search & Retrieval module.
Validates search accuracy, top_k limits, metadata integrity, semantic query retrieval,
category and error-code matching, error handling, and backward compatibility.
All tests run in isolated temporary ChromaDB environments.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import docx
import chromadb

from src.document_ingestion import ingest_document
from src.rag_search import search_knowledge_base, query_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRAGSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create an isolated temporary directory and populate with test documents
        cls.temp_dir = tempfile.mkdtemp()
        cls.chroma_dir = Path(cls.temp_dir) / "test_chroma_rag"
        cls.collection_name = "test_support_docs"

        # Build a rich test document covering all categories, parts, and error codes
        test_doc_path = Path(cls.temp_dir) / "test_technical_manual.docx"
        doc = docx.Document()

        # Operation section
        doc.add_heading("Section 1.1 Daily Startup", level=1)
        doc.add_paragraph("Doc ref: OG-WP400-1.1 | Category: Machine Operation Issues | Machine: WP-400")
        doc.add_paragraph(
            "Check the compressed-air gauge. Working pressure must read 6.0 bar (0.6 MPa). "
            "Do not start below 5.5 bar. Press HOME to complete homing on all axes."
        )

        # Maintenance section
        doc.add_heading("Section 2.1 Spindle Maintenance", level=1)
        doc.add_paragraph("Doc ref: MP-WP400-2.1 | Category: Maintenance & Parts | Machine: WP-400")
        doc.add_paragraph(
            "Replace spindle belt WP4-BT-85 every 500 hours or if cracked. "
            "Replace dust extraction filter cartridge WP4-FL-DE1 when alarm E-102 occurs or every 3 months. "
            "Lubricate linear rails with grease LG-2."
        )

        # Technical troubleshooting section
        doc.add_heading("Section 3.1 Troubleshooting and Error Codes", level=1)
        doc.add_paragraph("Doc ref: TS-WP400-3.1 | Category: Technical Troubleshooting | Machine: WP-400")
        doc.add_paragraph(
            "Error code E-102 indicates dust extraction fault. Likely cause: filter cartridge blocked "
            "(pressure drop above 1200 Pa) or fan motor stopped. Clean or replace cartridge WP4-FL-DE1."
        )
        doc.add_paragraph(
            "Alarm E-410 indicates emergency-stop circuit open. Likely cause: E-stop pressed or guard door open. "
            "Action: Release E-stop, close door, press RESET."
        )

        doc.save(str(test_doc_path))

        # Ingest into test collection
        cls.ingest_stats = ingest_document(
            file_path=test_doc_path,
            collection_name=cls.collection_name,
            chroma_dir=cls.chroma_dir
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Search Basics & top_k
    # --------------------------------------------------------------------------
    def test_search_returns_results(self):
        results = search_knowledge_base(
            query="compressed air working pressure",
            top_k=2,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertGreater(len(results), 0)
        self.assertIn("text", results[0])
        self.assertIn("metadata", results[0])
        self.assertIn("distance", results[0])
        self.assertIn("similarity_score", results[0])
        self.assertTrue(0.0 <= results[0]["similarity_score"] <= 1.0)

    def test_search_respects_top_k(self):
        results_1 = search_knowledge_base(
            query="machine",
            top_k=1,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertEqual(len(results_1), 1)

        results_2 = search_knowledge_base(
            query="machine",
            top_k=2,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertEqual(len(results_2), 2)

    # --------------------------------------------------------------------------
    # 2. Category Relevance Searches
    # --------------------------------------------------------------------------
    def test_operation_query_retrieves_operation_document(self):
        results = search_knowledge_base(
            query="How do I start the machine and what is the air pressure needed?",
            top_k=1,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertEqual(top["metadata"]["category"], "Machine Operation Issues")
        self.assertIn("6.0 bar", top["text"])

    def test_maintenance_query_retrieves_maintenance_document(self):
        results = search_knowledge_base(
            query="When should I service the spindle belt and what grease should I use?",
            top_k=1,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertEqual(top["metadata"]["category"], "Maintenance & Parts")
        self.assertIn("WP4-BT-85", top["text"])

    def test_troubleshooting_query_retrieves_troubleshooting_document(self):
        results = search_knowledge_base(
            query="The machine stopped with emergency stop circuit open alarm",
            top_k=1,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertEqual(top["metadata"]["category"], "Technical Troubleshooting")
        self.assertIn("E-410", top["text"])

    # --------------------------------------------------------------------------
    # 3. Technical Identifiers (Error codes & Part numbers)
    # --------------------------------------------------------------------------
    def test_error_code_query_retrieves_correct_document(self):
        results = search_knowledge_base(
            query="E-102 dust extraction fault pressure drop",
            top_k=1,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertIn("E-102", top["text"])
        self.assertIn("1200 Pa", top["text"])

    def test_part_number_query_retrieves_correct_document(self):
        results = search_knowledge_base(
            query="WP4-FL-DE1 replacement filter cartridge",
            top_k=1,
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertIn("WP4-FL-DE1", top["text"])

    # --------------------------------------------------------------------------
    # 4. Filters & Edge Cases
    # --------------------------------------------------------------------------
    def test_category_and_machine_filters(self):
        results = search_knowledge_base(
            query="maintenance instructions",
            top_k=5,
            category_filter="Maintenance & Parts",
            machine_filter="WP-400",
            collection_name=self.collection_name,
            chroma_dir=self.chroma_dir
        )
        for r in results:
            self.assertEqual(r["metadata"]["category"], "Maintenance & Parts")
            self.assertEqual(r["metadata"]["machine"], "WP-400")

    def test_empty_query_raises_valueerror(self):
        with self.assertRaises(ValueError):
            search_knowledge_base("", collection_name=self.collection_name, chroma_dir=self.chroma_dir)
        with self.assertRaises(ValueError):
            search_knowledge_base("   \n\t  ", collection_name=self.collection_name, chroma_dir=self.chroma_dir)

    def test_invalid_top_k_raises_valueerror(self):
        with self.assertRaises(ValueError):
            search_knowledge_base("test query", top_k=0, collection_name=self.collection_name, chroma_dir=self.chroma_dir)

    def test_nonexistent_directory_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            search_knowledge_base("test query", chroma_dir=Path(self.temp_dir) / "does_not_exist")

    def test_empty_collection_returns_empty_list(self):
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        client.get_or_create_collection("empty_col")
        results = search_knowledge_base("test query", collection_name="empty_col", chroma_dir=self.chroma_dir)
        self.assertEqual(results, [])

    # --------------------------------------------------------------------------
    # 5. Phase 0 Backward Compatibility
    # --------------------------------------------------------------------------
    def test_query_knowledge_base_backward_compatibility(self):
        # Must return list of dicts with {"document": str, "metadata": dict}
        results = query_knowledge_base(
            query_text="What is the air pressure needed?",
            n_results=2
        )
        self.assertIsInstance(results, list)
        if len(results) > 0:
            first = results[0]
            self.assertIn("document", first)
            self.assertIn("metadata", first)
            self.assertIsInstance(first["document"], str)
            self.assertIsInstance(first["metadata"], dict)


if __name__ == "__main__":
    unittest.main()

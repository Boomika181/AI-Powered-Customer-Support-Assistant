"""
tests/test_phase0.py

Automated Test Suite for Phase 0 Setup and Ingestion.
"""

import os
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase0Setup(unittest.TestCase):

    def test_env_example_exists(self):
        env_example = PROJECT_ROOT / ".env.example"
        self.assertTrue(env_example.exists(), ".env.example must exist.")
        content = env_example.read_text()
        self.assertIn("ASSEMBLYAI_API_KEY", content)
        self.assertIn("GEMINI_API_KEY", content)
        self.assertIn("CHROMA_PERSIST_DIRECTORY", content)

    def test_synthetic_pdf_exists(self):
        pdf_path = PROJECT_ROOT / "data" / "raw" / "Sample_Technical_Documentation_Pack_SYNTHETIC.pdf"
        self.assertTrue(pdf_path.exists(), "Sample synthetic PDF must be in data/raw/.")
        self.assertGreater(pdf_path.stat().st_size, 10000, "PDF should not be empty.")

    def test_processed_chunks_json(self):
        chunks_json = PROJECT_ROOT / "data" / "processed" / "chunks.json"
        self.assertTrue(chunks_json.exists(), "data/processed/chunks.json must exist.")
        with open(chunks_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        self.assertGreaterEqual(len(chunks), 20, "Should have extracted at least 20 chunks.")

        required_keys = {"source", "page", "document_reference", "section", "category", "machine", "text"}
        for c in chunks[:5]:
            self.assertTrue(required_keys.issubset(set(c.keys())), f"Missing keys in chunk: {set(c.keys())}")

    def test_chromadb_collection_and_query(self):
        import chromadb
        chroma_dir = PROJECT_ROOT / "chroma_db"
        self.assertTrue(chroma_dir.exists(), "ChromaDB directory must exist.")

        client = chromadb.PersistentClient(path=str(chroma_dir))
        col = client.get_collection("technical_documentation")
        count = col.count()
        self.assertGreater(count, 0, "ChromaDB collection should contain stored chunks.")

        # Test simple query
        res = col.query(query_texts=["WP-400 saw blade replacement"], n_results=1)
        self.assertGreaterEqual(len(res["documents"][0]), 1)
        meta = res["metadatas"][0][0]
        self.assertIn("document_reference", meta)
        self.assertIn("page", meta)

    def test_rag_search_module(self):
        from src.rag_search import query_knowledge_base
        results = query_knowledge_base("spindle speed for melamine chipboard", n_results=2)
        self.assertGreaterEqual(len(results), 1)
        first_doc = results[0]
        self.assertIn("metadata", first_doc)
        self.assertIn("document", first_doc)

    def test_audio_input_detected(self):
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        self.assertGreater(len(input_devs), 0, "At least one input audio device must be detected.")


if __name__ == "__main__":
    unittest.main()

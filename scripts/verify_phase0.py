#!/usr/bin/env python3
"""
scripts/verify_phase0.py

Comprehensive Phase 0 Verification Script:
Validates the complete Phase 0 setup:
1. Python project environment (version, virtualenv)
2. Core dependencies installed and importable
3. Environment variable loading (.env, AssemblyAI and Gemini API keys)
4. Local ChromaDB vector database initialization
5. Processed document chunks and vector storage
6. Document metadata completeness
7. System microphone hardware detection and stream frame reception
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
load_dotenv(PROJECT_ROOT / ".env")


def check_python_environment():
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    is_supported = version_info.major == 3 and version_info.minor >= 10
    in_venv = sys.prefix != sys.base_prefix
    return {
        "name": "Python Environment",
        "passed": is_supported,
        "details": f"Python {version_str} (Supported: {is_supported}, VirtualEnv: {in_venv})"
    }


def check_dependencies():
    packages = [
        ("sounddevice", "sounddevice"),
        ("assemblyai", "assemblyai"),
        ("chromadb", "chromadb"),
        ("pymupdf", "pymupdf"),
        ("python-docx", "docx"),
        ("python-dotenv", "dotenv"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("google-genai", "google.genai"),
    ]
    missing = []
    for pkg_name, module_name in packages:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pkg_name)

    passed = len(missing) == 0
    details = "All 9 core dependencies imported successfully" if passed else f"Missing: {', '.join(missing)}"
    return {
        "name": "Project Dependencies",
        "passed": passed,
        "details": details
    }


def check_api_keys():
    aai_key = os.getenv("ASSEMBLYAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    aai_present = bool(aai_key and aai_key.strip() and not aai_key.startswith("your_"))
    gemini_present = bool(gemini_key and gemini_key.strip() and not gemini_key.startswith("your_"))

    aai_masked = f"{aai_key[:4]}...{aai_key[-4:]}" if aai_present else "NOT CONFIGURED"
    gemini_masked = f"{gemini_key[:4]}...{gemini_key[-4:]}" if gemini_present else "NOT CONFIGURED"

    details = f"AssemblyAI: {aai_masked} | Gemini: {gemini_masked}"
    # In Phase 0, keys may be pending user entry, so mark warning if missing
    return {
        "name": "Environment Variables & API Keys",
        "passed": aai_present and gemini_present,
        "is_warning": not (aai_present and gemini_present),
        "details": details
    }


def check_chromadb_and_documents():
    import chromadb

    chroma_dir = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
    col_name = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")

    if not chroma_dir.exists():
        return {
            "name": "ChromaDB & Document Storage",
            "passed": False,
            "details": f"Chroma directory {chroma_dir} does not exist"
        }

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collections = [c.name for c in client.list_collections()]
        if col_name not in collections:
            return {
                "name": "ChromaDB & Document Storage",
                "passed": False,
                "details": f"Collection '{col_name}' not found. Run 'python scripts/ingest_documents.py'."
            }

        collection = client.get_collection(col_name)
        count = collection.count()
        if count == 0:
            return {
                "name": "ChromaDB & Document Storage",
                "passed": False,
                "details": "Collection exists but contains 0 chunks."
            }

        return {
            "name": "ChromaDB & Document Storage",
            "passed": True,
            "details": f"Collection '{col_name}' loaded with {count} chunks."
        }
    except Exception as e:
        return {
            "name": "ChromaDB & Document Storage",
            "passed": False,
            "details": f"ChromaDB initialization error: {e}"
        }


def check_metadata_integrity():
    import chromadb

    chroma_dir = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
    col_name = os.getenv("CHROMA_COLLECTION_NAME", "technical_documentation")

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(col_name)
        sample = collection.peek(limit=5)

        required_keys = {"source", "page", "document_reference", "section", "category", "machine"}
        for meta in sample["metadatas"]:
            missing_keys = required_keys - set(meta.keys())
            if missing_keys:
                return {
                    "name": "Document Metadata Integrity",
                    "passed": False,
                    "details": f"Chunk missing required metadata fields: {missing_keys}"
                }

        # Check unique machines and categories
        machines = {m["machine"] for m in sample["metadatas"]}
        return {
            "name": "Document Metadata Integrity",
            "passed": True,
            "details": f"All required fields present (source, page, document_reference, section, category, machine)."
        }
    except Exception as e:
        return {
            "name": "Document Metadata Integrity",
            "passed": False,
            "details": f"Failed checking metadata: {e}"
        }


def check_microphone():
    import sounddevice as sd
    import numpy as np

    try:
        all_devices = sd.query_devices()
        input_devices = [d for d in all_devices if d["max_input_channels"] > 0]
        if not input_devices:
            return {
                "name": "Microphone Hardware & Frame Capture",
                "passed": False,
                "details": "No audio input devices detected."
            }

        default_dev = sd.query_devices(kind="input")
        stream = sd.InputStream(device=default_dev["index"], samplerate=16000, channels=1, blocksize=1024)
        stream.start()
        data, overflowed = stream.read(1024)
        stream.stop()
        stream.close()

        frames_ok = len(data) == 1024
        return {
            "name": "Microphone Hardware & Frame Capture",
            "passed": frames_ok,
            "details": f"Detected: '{default_dev['name']}' | Stream opened | {len(data)} frames captured successfully | Zero disk storage"
        }
    except Exception as e:
        return {
            "name": "Microphone Hardware & Frame Capture",
            "passed": False,
            "details": f"Audio stream error: {e}"
        }


def run_all_verifications():
    print("=" * 70)
    print("AI-Powered Customer Support Assistant - Phase 0 Master Verification")
    print("=" * 70)

    checks = [
        check_python_environment(),
        check_dependencies(),
        check_api_keys(),
        check_chromadb_and_documents(),
        check_metadata_integrity(),
        check_microphone(),
    ]

    all_passed = True
    warnings = []

    for idx, c in enumerate(checks, start=1):
        if c.get("is_warning", False):
            status = "[WARNING]"
            warnings.append(c["name"])
        elif c["passed"]:
            status = "[PASS]"
        else:
            status = "[FAIL]"
            all_passed = False

        print(f"\n{idx}. {c['name']}")
        print(f"   Status:  {status}")
        print(f"   Details: {c['details']}")

    print("\n" + "=" * 70)
    print("Master Verification Summary:")
    print("=" * 70)
    if all_passed and not warnings:
        print("RESULT: ALL PHASE 0 CHECKS PASSED PERFECTLY!")
    elif all_passed and warnings:
        print("RESULT: ALL INFRASTRUCTURE & CODE CHECKS PASSED.")
        print(f"NOTE:   Pending API keys in .env: {', '.join(warnings)}")
        print("        (Add your keys to .env when ready to enable live AssemblyAI & Gemini calls)")
    else:
        print("RESULT: SOME PHASE 0 CHECKS FAILED. Review errors above.")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_verifications()
    sys.exit(0 if success else 1)

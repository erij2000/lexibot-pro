
import os
import sys
import asyncio
import logging
import platform
import torch
from pathlib import Path
from datetime import datetime

# Add engine to path
ENGINE_PATH = Path("backend/lexios_engine").resolve()
sys.path.append(str(ENGINE_PATH.parent))
sys.path.append(str(ENGINE_PATH))

from config import settings, validate_environment
from ocr_pipeline import LegalOCR, LexiosDoc
from rag_service import HybridRetriever
from pipeline_orchestrator import LexiosPipeline

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backend/logs/test_suite.log", encoding="utf-8")
    ]
)
log = logging.getLogger("lexios.test")

TEST_INPUTS = Path("backend/data/test_inputs")

class LexiosTestSuite:
    def __init__(self):
        self.results = []
        self.ocr = None
        self.retriever = None
        self.pipeline = None

    def log_result(self, category, test_name, success, message=""):
        status = "✅ PASS" if success else "❌ FAIL"
        log.info(f"[{category}] {test_name}: {status} - {message}")
        self.results.append({
            "category": category,
            "test_name": test_name,
            "success": success,
            "message": message
        })

    async def run_env_check(self):
        log.info("--- STEP 1: ENVIRONMENT CHECK ---")
        success = validate_environment()
        self.log_result("ENV", "Config Validation", success)
        
        # Check packages
        try:
            import surya
            import marker
            self.log_result("ENV", "OCR Packages (Surya/Marker)", True)
        except ImportError as e:
            self.log_result("ENV", "OCR Packages", False, f"Missing: {e}")

        # Check CUDA
        cuda_ok = torch.cuda.is_available()
        self.log_result("ENV", "CUDA Availability", cuda_ok, f"Device: {torch.cuda.get_device_name(0) if cuda_ok else 'CPU'}")

    async def run_ocr_tests(self):
        log.info("--- STEP 2: OCR TESTING ---")
        self.ocr = LegalOCR()
        test_files = [
            ("test_digital_like.pdf", "fr"),
            ("test_scanned.pdf", "fr"),
            ("test_multilingual.png", "mixed"),
        ]

        docs = []
        for filename, expected_lang in test_files:
            path = TEST_INPUTS / filename
            if not path.exists():
                self.log_result("OCR", f"File {filename}", False, "File not found")
                continue
            
            try:
                log.info(f"Processing OCR for {filename}...")
                doc = await self.ocr.process_file(path)
                if doc and doc.raw_text.strip():
                    # Check language
                    lang_ok = doc.language == expected_lang or expected_lang == "mixed"
                    self.log_result("OCR", f"Extraction {filename}", True, f"Lang: {doc.language}, Chunks: {len(doc.chunks)}")
                    docs.append(doc)
                else:
                    self.log_result("OCR", f"Extraction {filename}", False, "Empty output")
            except Exception as e:
                self.log_result("OCR", f"Extraction {filename}", False, str(e))
        
        return docs

    async def run_rag_tests(self, docs):
        log.info("--- STEP 3: RAG TESTING ---")
        if not docs:
            self.log_result("RAG", "Ingestion", False, "No docs from OCR")
            return

        self.retriever = HybridRetriever()
        try:
            for doc in docs:
                log.info(f"Indexing {doc.source_file}...")
                # Convert LexiosDoc to dict for indexing
                await self.retriever.index_document(doc.to_dict())
            
            self.log_result("RAG", "Ingestion", True, f"Indexed {len(docs)} documents")

            # Test Retrieval
            query = "ARTICLE 1: DEFINITIONS"
            chunks, stats, _ = await self.retriever.retrieve(query, top_k=5)
            
            if chunks:
                top_text = chunks[0].text
                found_art = "ARTICLE 1" in top_text or "DEFINITIONS" in top_text
                self.log_result("RAG", "Retrieval Accuracy", found_art, f"Top result: {top_text[:50]}...")
            else:
                self.log_result("RAG", "Retrieval Accuracy", False, "No chunks retrieved")

        except Exception as e:
            self.log_result("RAG", "Pipeline Error", False, str(e))

    async def run_edge_cases(self):
        log.info("--- STEP 4: EDGE CASES ---")
        # 1. Corrupted PDF
        path = TEST_INPUTS / "corrupted.pdf"
        try:
            doc = await self.ocr.process_file(path)
            self.log_result("EDGE", "Corrupted PDF", doc is None or doc.raw_text == "", "Graceful failure/Empty output")
        except Exception as e:
            self.log_result("EDGE", "Corrupted PDF", True, f"Caught expected crash: {e}")

        # 2. Non-existent file
        path = TEST_INPUTS / "missing.pdf"
        doc = await self.ocr.process_file(path)
        self.log_result("EDGE", "Missing File", doc is None, "Correctly returned None")

    async def run_e2e_pipeline(self):
        log.info("--- STEP 5: E2E PIPELINE ---")
        if not self.retriever:
            self.log_result("E2E", "Pipeline", False, "Retriever not initialized")
            return

        self.pipeline = LexiosPipeline(self.retriever)
        await self.pipeline.initialize()
        
        query = "C'est quoi le Service selon l'article 1 ?"
        try:
            result = await self.pipeline.process(query)
            success = len(result.answer) > 50
            self.log_result("E2E", "Full Flow", success, f"Latency: {result.timings['total_ms']:.0f}ms")
            log.info(f"Answer: {result.answer[:100]}...")
        except Exception as e:
            self.log_result("E2E", "Full Flow", False, str(e))

    def print_summary(self):
        print("\n" + "="*50)
        print("📊 FINAL TEST SUMMARY")
        print("="*50)
        passed = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        print(f"OVERALL: {passed}/{total} PASSED")
        print("-" * 50)
        for r in self.results:
            icon = "✅" if r["success"] else "❌"
            print(f"{icon} [{r['category']}] {r['test_name']}")
            if not r["success"]:
                print(f"   Error: {r['message']}")
        print("="*50 + "\n")

    async def cleanup(self):
        if self.ocr: await self.ocr.close()
        if self.retriever: await self.retriever.close()

async def main():
    suite = LexiosTestSuite()
    try:
        await suite.run_env_check()
        docs = await suite.run_ocr_tests()
        await suite.run_rag_tests(docs)
        await suite.run_edge_cases()
        await suite.run_e2e_pipeline()
    finally:
        suite.print_summary()
        await suite.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

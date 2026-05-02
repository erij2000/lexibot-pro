"""
real_data_test.py — Lexios Full End-to-End Pipeline Test
=========================================================
Tests OCR on real Drive documents then runs RAG queries.
"""
import asyncio
import sys
import os
import json
import platform
from pathlib import Path
from datetime import datetime

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if platform.system() == "Windows":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # LEXIBOT_PRO/
ENGINE = ROOT / "backend" / "lexios_engine"
sys.path.insert(0, str(ENGINE.parent))              # …/backend
sys.path.insert(0, str(ENGINE))                     # …/backend/lexios_engine

from ocr_pipeline import LegalOCR, LexiosDoc        # noqa: E402
from config import settings                          # noqa: E402


# ── Real data paths ───────────────────────────────────────────────────────────
PDF_PATH  = Path(r"G:\Mon Drive\lexiOS\civil\Diwan   Documents   (Ar) Code de la comptabilité publique - Version 2024.pdf")
HEIC_PATH = Path(r"G:\Mon Drive\lexiOS\IMG_1919 (1).HEIC")


def sep(char="=", n=60):
    print(char * n)


async def run_ocr_phase(ocr: LegalOCR) -> list:
    """STEP 1 — OCR on real files."""
    sep()
    print("STEP 1 — OCR PHASE (Real Data)")
    sep()

    test_files = [PDF_PATH, HEIC_PATH]
    docs = []

    for path in test_files:
        if not path.exists():
            print(f"[SKIP] File not found: {path}")
            continue

        suffix = path.suffix.lower()
        engine_name = "Marker-PDF" if suffix == ".pdf" else "Surya-OCR"
        print(f"\n[FILE] {path.name}")
        print(f"       Engine : {engine_name}")

        try:
            t0 = datetime.now()
            doc: LexiosDoc = await ocr.process_file(path)
            elapsed = (datetime.now() - t0).total_seconds()

            if doc and doc.raw_text.strip():
                print(f"[OK]   Duration     : {elapsed:.1f}s")
                print(f"       Language     : {doc.language}")
                print(f"       Pages        : {doc.pages}")
                print(f"       Doc Type     : {doc.doc_type_detected} (conf={doc.doc_type_confidence:.2f})")
                print(f"       Chunks       : {len(doc.chunks)}")
                print(f"       Entities     : {doc.entities}")
                print("-" * 50)
                print("       TEXT SAMPLE (300 chars):")
                print(doc.raw_text[:300])
                print("-" * 50)

                # Append doc FIRST so RAG phase can use it even if JSON check fails
                docs.append(doc)

                # Verify JSON saved to disk
                src_stem = Path(doc.source_file).stem
                out_path = Path(settings.OCR_OUTPUT_DIR) / f"{src_stem}_{doc.uid}.json"
                if out_path.exists():
                    with open(out_path, encoding="utf-8") as f:
                        saved = json.load(f)
                    assert saved.get("uid") and saved.get("chunks"), "JSON missing fields"
                    print(f"[OK]   JSON saved : {out_path.name}")
                else:
                    print(f"[WARN] JSON not found at {out_path}")

                docs.append(doc)
            else:
                print("[FAIL] OCR returned empty result")

        except Exception as exc:
            print(f"[FAIL] {exc}")

    return docs


async def run_rag_phase(docs: list):
    """STEP 2+3 — Index + Hybrid Retrieval."""
    sep()
    print("STEP 2 — RAG INDEXING & RETRIEVAL")
    sep()

    if not docs:
        print("[SKIP] No OCR docs to index.")
        return None

    try:
        from rag_service import HybridRetriever
    except ImportError as e:
        print(f"[FAIL] Cannot import rag_service: {e}")
        return None

    retriever = HybridRetriever()

    # Index all docs
    for doc in docs:
        print(f"[INDEX] {doc.uid} ({len(doc.chunks)} chunks)...")
        await retriever.index_document(doc.to_dict())
    retriever.bm25.flush()
    print("[OK]   All documents indexed.")

    # Run 3 canonical queries
    queries = [
        "resume ce document",
        "quels articles sont mentionnes ?",
        "explique ce texte juridique",
    ]

    sep("-")
    print("STEP 3 — RETRIEVAL DEBUG")
    sep("-")

    for query in queries:
        print(f"\n[QUERY] {query}")
        chunks, stats, _ = await retriever.retrieve(query, top_k=3)

        if not chunks:
            print("  [WARN] No chunks retrieved.")
            continue

        for i, c in enumerate(chunks, 1):
            print(f"  Chunk {i}:")
            print(f"    article_id : {c.article_id}")
            print(f"    score      : {c.final_score:.4f}")
            print(f"    text       : {c.text[:120]}...")

    return retriever


async def run_llm_phase(retriever, docs: list):
    """STEP 4 — Full LLM synthesis via pipeline orchestrator."""
    sep()
    print("STEP 4 — LLM SYNTHESIS (pipeline_orchestrator)")
    sep()

    if not retriever or not docs:
        print("[SKIP] Missing retriever or docs.")
        return

    try:
        from pipeline_orchestrator import LexiosPipeline
    except ImportError as e:
        print(f"[FAIL] Cannot import pipeline_orchestrator: {e}")
        return

    pipeline = LexiosPipeline(retriever)
    await pipeline.initialize()

    test_queries = [
        "resume ce document",
        "quels articles sont mentionnes ?",
        "donne moi les informations importantes",
    ]

    for query in test_queries:
        print(f"\n[Q] {query}")
        try:
            result = await pipeline.process(query)
            print(f"[A] {result.answer[:400]}")
            timing = result.timings.get("total_ms", 0)
            print(f"    Latency : {timing:.0f} ms")
            if hasattr(result, "sources") and result.sources:
                print(f"    Sources : {[s.get('article_id') for s in result.sources[:3]]}")
        except Exception as e:
            print(f"[FAIL] {e}")


async def run_ragas_phase(retriever, docs: list):
    """STEP 5 — RAGAS Quality Evaluation."""
    sep()
    print("STEP 5 — RAGAS EVALUATION")
    sep()

    if not retriever or not docs:
        print("[SKIP] Missing data.")
        return

    try:
        from ragas_evaluator import LexiosRAGEvaluator
    except ImportError:
        print("[SKIP] ragas_evaluator not importable (dependencies missing?).")
        return

    evaluator = LexiosRAGEvaluator()
    query = "resume ce document"

    chunks, _, _ = await retriever.retrieve(query, top_k=5)
    if not chunks:
        print("[WARN] No chunks — cannot evaluate.")
        return

    context = [c.text for c in chunks]
    answer  = "Ce document porte sur les dispositions de la comptabilite publique tunisienne."

    try:
        report = await evaluator.evaluate(
            question=query,
            answer=answer,
            contexts=context,
            ground_truth=None,
        )
        print(f"[OK] Faithfulness       : {report.metrics.faithfulness:.3f}")
        print(f"     Answer Relevancy   : {report.metrics.answer_relevancy:.3f}")
        print(f"     Context Precision  : {report.metrics.context_precision:.3f}")
        print(f"     Composite Score    : {report.metrics.composite_score:.3f}")
        is_ok = "PASS" if report.metrics.is_acceptable else "FAIL"
        print(f"     VERDICT            : {is_ok}")
    except Exception as e:
        print(f"[FAIL] RAGAS evaluation error: {e}")


async def main():
    sep("*")
    print("  LEXIOS END-TO-END PIPELINE TEST")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sep("*")

    ocr = LegalOCR()

    docs     = await run_ocr_phase(ocr)
    retriever = await run_rag_phase(docs)
    await run_llm_phase(retriever, docs)
    await run_ragas_phase(retriever, docs)

    await ocr.close()
    if retriever:
        await retriever.close()

    sep("*")
    print("  PIPELINE TEST COMPLETE")
    sep("*")


if __name__ == "__main__":
    asyncio.run(main())

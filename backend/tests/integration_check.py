"""
integration_check.py — Lexios RAG/LLM End-to-End Validation
=============================================================
Test de l'intégration OCR -> VectorDB -> LLM avec un document simulé.
"""

import sys
import platform
import asyncio
import json
from pathlib import Path
from datetime import datetime

if platform.system() == "Windows":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "backend" / "lexios_engine"
sys.path.insert(0, str(ENGINE.parent))
sys.path.insert(0, str(ENGINE))

from config import settings
from ocr_pipeline import LexiosDoc
from rag_service import LexiosRAG

def sep(char="=", n=60):
    print(char * n)

async def main():
    sep()
    print("🚀 LEXIOS PRO - MASTER INTEGRATION CHECK")
    sep()
    
    # 1. LOAD MOCK DATA
    ocr_file = Path(settings.OCR_OUTPUT_DIR) / "Diwan Documents (Ar) Code de la comptabilité publique - Version 2024_ce2570c44a7204c7.json"
    
    if not ocr_file.exists():
        print(f"[ERROR] Mock OCR file missing: {ocr_file}")
        return
        
    print(f"[1/3] Loading OCR Data: {ocr_file.name}")
    with open(ocr_file, encoding="utf-8") as f:
        data = json.load(f)
        
    if "processed_at" not in data:
        data["processed_at"] = datetime.now().isoformat()
    for i, c in enumerate(data.get("chunks", [])):
        if "article_id" not in c:
            c["article_id"] = c.get("metadata", {}).get("section", f"الفصل {i+1}")
    doc = LexiosDoc(**data)
    print(f"      => Extracted: {len(doc.chunks)} chunks, Language: {doc.language}")

    # 2. RAG INGESTION
    print("\n[2/3] Vectorizing & Ingesting to ChromaDB (Safe Mode: Fast & Cool)...")
    settings.USE_RERANKER = False # Désactive le modèle lourd pour éviter la surchauffe
    rag = LexiosRAG()
    
    try:
        await rag.ingest_document(data)
        print("      => [OK] Ingestion Successful (BGE-M3 Embeddings + Hybrid)")
    except Exception as e:
        print(f"      => [ERROR] Exception during ingestion: {e}")
        return
        
    # 3. LLM QUERY
    query = "ماهي مكونات ميزانية الدولة حسب قانون المالية؟"
    print(f"\n[3/3] Testing LLM Query (Qwen-2.5-32b via Groq)")
    print(f"      Query: {query}")
    print("      Processing...")
    
    try:
        t0 = datetime.now()
        answer = await rag.query(query)
        elapsed = (datetime.now() - t0).total_seconds()
        
        sep("-")
        print("💡 REPONSE GENEREE PAR LEXIBOT:")
        print(answer.get("answer", answer))
        sep("-")
        print(f"      => [OK] RAG Cycle completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"      => [ERROR] LLM Query Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

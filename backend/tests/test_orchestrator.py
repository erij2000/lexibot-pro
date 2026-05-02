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
from pipeline_orchestrator import LexiosPipeline

def sep(char="=", n=60):
    print(char * n)

async def main():
    sep()
    print("🚀 LEXIOS PRO - FULL ORCHESTRATOR & LIGHTRAG TEST")
    sep()
    
    # Mode Safe (Désactiver Reranker pour éviter la surchauffe)
    settings.USE_RERANKER = False
    
    # 1. LOAD MOCK DATA
    ocr_file = Path(settings.OCR_OUTPUT_DIR) / "Diwan Documents (Ar) Code de la comptabilité publique - Version 2024_ce2570c44a7204c7.json"
    
    with open(ocr_file, encoding="utf-8") as f:
        data = json.load(f)
        
    if "processed_at" not in data:
        data["processed_at"] = datetime.now().isoformat()
    for i, c in enumerate(data.get("chunks", [])):
        if "article_id" not in c:
            c["article_id"] = c.get("metadata", {}).get("section", f"الفصل {i+1}")
            
    print(f"[1/4] Loaded Data: {ocr_file.name} ({len(data.get('chunks', []))} chunks)")

    # 2. INIT ORCHESTRATOR
    print("\n[2/4] Initializing LexiosPipeline (HyDE + LightRAG + LexiosRAG)...")
    base_rag = LexiosRAG()
    pipeline = LexiosPipeline(base_rag)
    await pipeline.initialize()
    
    # 3. INGESTION (Inclut LightRAG Graph generation)
    print("\n[3/4] Ingesting Document into Vector DB & LightRAG Graph...")
    try:
        await base_rag.ingest_document(data)
        print("      => [OK] Ingestion Successful (Vectors + Graph)")
    except Exception as e:
        print(f"      => [ERROR] Exception during ingestion: {e}")
        return
        
    # 4. FULL PIPELINE QUERY
    query = "ما هي العلاقة بين ميزانية الدولة وقانون المالية والبرامج؟" # "What is the relationship between state budget, finance law and programs?"
    print(f"\n[4/4] Testing Complex Orchestrated Query")
    print(f"      Query: {query}")
    print("      Processing via HyDE -> LexiosRAG -> LightRAG -> Groq...")
    
    try:
        t0 = datetime.now()
        result = await pipeline.process(query)
        elapsed = (datetime.now() - t0).total_seconds()
        
        sep("-")
        print("💡 REPONSE GENEREE PAR LEXIBOT (ORCHESTRATOR):")
        print(result.answer)
        sep("-")
        
        print("\n📊 MÉTRIQUES DE L'ORCHESTRATEUR:")
        print(f" - Temps total : {elapsed:.1f}s")
        print(f" - Route choisie : {result.route.mode}")
        print(f" - HyDE activé : {result.route.use_hyde}")
        print(f" - LightRAG activé : {result.route.use_lightrag}")
        print(f" - Chunks utilisés : {result.context.chunks_used}")
        print(f" - Faits du Graphe utilisés : {result.context.graph_facts_used}")
        
    except Exception as e:
        print(f"      => [ERROR] Pipeline Query Failed: {e}")
        
    await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())

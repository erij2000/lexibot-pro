"""
ingestion.py — Lexios Ingestion Engine v6 (Clean Article-Aware)
================================================================
Orchestration : OCR → Article-Aware Indexation
- Pas de FAISS (supprimé)
- Pas d'EmbeddingCache redondant (géré par RTX2050SafeEmbedder)
- ChromaDB persistant (déjà géré par PersistentClient)
- BM25 persistance optionnelle sur disque
- Utilise LexiosRAG directement pour tout le pipeline d'indexation
"""

import asyncio
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List

from config import settings

try:
    from lightrag_bridge import LightRAGBridge, get_lightrag_bridge
    HAS_LIGHTRAG_BRIDGE = True
except ImportError:
    HAS_LIGHTRAG_BRIDGE = False
from ocr_pipeline import LegalOCR, LexiosDoc
from rag_service import LexiosRAG

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

log = logging.getLogger("lexios.ingestion")


class IngestionPipeline:
    """Orchestre OCR → Indexation via LexiosRAG."""

    def __init__(self):
        self.rag = LexiosRAG()
        self.ocr = LegalOCR()
        
        # Chemins de persistance BM25 (ChromaDB est déjà persistant par défaut)
        self.bm25_path = Path(settings.CACHE_DIR) / "bm25_index.pkl"
        self.meta_path = Path(settings.CACHE_DIR) / "ingestion_meta.json"
        Path(settings.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        log.info("IngestionPipeline v6 initialisé (Article-Aware, no FAISS)")

    async def run(self, input_dir: str) -> Dict[str, Any]:
        log.info("=" * 60)
        log.info("PHASE 1: OCR Extraction")
        log.info("=" * 60)

        docs = await self.ocr.process_folder(input_dir)
        if not docs:
            log.warning("Aucun document à indexer")
            return {"status": "no_docs", "ocr_count": 0, "indexed_chunks": 0}

        log.info("=" * 60)
        log.info("PHASE 2: Indexation Article-Aware (Lexios)")
        log.info("=" * 60)

        total_chunks = 0
        for doc in docs:
            result = await self.rag.ingest_document(doc.to_dict())
            if result["status"] == "success":
                total_chunks += result["chunks_indexed"]

        # ── PHASE 2b: Indexation LightRAG (graphe) ──
        lightrag_indexed = 0
        if HAS_LIGHTRAG_BRIDGE and settings.LIGHTRAG_ENABLED:
            log.info("=" * 60)
            log.info("PHASE 2b: Indexation LightRAG (Knowledge Graph)")
            log.info("=" * 60)
            try:
                bridge = await get_lightrag_bridge(self.rag)
                for doc in docs:
                    success = await bridge.index_document(doc.to_dict())
                    if success:
                        lightrag_indexed += 1
                log.info(f"📊 LightRAG: {lightrag_indexed}/{len(docs)} documents indexés")
            except Exception as e:
                log.warning(f"LightRAG indexation skipped: {e}")

        # Persistance BM25 (ChromaDB est déjà persistant via PersistentClient)
        self._persist_bm25()

        stats = {
            "status": "success",
            "ocr_count": len(docs),
            "indexed_chunks": total_chunks,
            "lightrag_indexed": lightrag_indexed,
            "health": self.rag.get_health_status()
        }
        
        # Sauvegarde métadonnées d'ingestion
        self._save_meta(stats)
        
        log.info(f"Ingestion terminée: {stats}")
        return stats

    def _persist_bm25(self):
        """Sauvegarde l'index BM25 sur disque pour rechargement futur."""
        if not HAS_BM25:
            return
        try:
            bm25_svc = self.rag.retriever.bm25
            if bm25_svc.index and bm25_svc.tokenized_corpus:
                payload = {
                    "tokenized": bm25_svc.tokenized_corpus,
                    "chunk_map": bm25_svc.chunk_map,
                    "chunk_texts": bm25_svc.chunk_texts,
                }
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump(payload, f)
                log.info(f"BM25 persisté: {len(bm25_svc.tokenized_corpus)} chunks")
        except Exception as e:
            log.warning(f"Persistance BM25 échouée: {e}")

    def load_bm25(self) -> bool:
        """Charge l'index BM25 depuis disque."""
        if not HAS_BM25 or not self.bm25_path.exists():
            return False
        try:
            with open(self.bm25_path, 'rb') as f:
                data = pickle.load(f)
            
            bm25_svc = self.rag.retriever.bm25
            bm25_svc.tokenized_corpus = data["tokenized"]
            bm25_svc.chunk_map = data["chunk_map"]
            bm25_svc.chunk_texts = data["chunk_texts"]
            bm25_svc.index = BM25Okapi(bm25_svc.tokenized_corpus)
            
            log.info(f"BM25 chargé depuis disque: {len(bm25_svc.tokenized_corpus)} chunks")
            return True
        except Exception as e:
            log.warning(f"Chargement BM25 échoué: {e}")
            return False

    def _save_meta(self, stats: Dict[str, Any]):
        """Sauvegarde les métadonnées de la dernière ingestion."""
        try:
            self.meta_path.write_text(
                json.dumps({**stats, "timestamp": __import__('time').time()}, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"Sauvegarde métadonnées échouée: {e}")

    def last_ingestion_stats(self) -> Dict[str, Any]:
        """Retourne les stats de la dernière ingestion si disponibles."""
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    async def close(self):
        await self.ocr.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lexios Ingestion v6")
    parser.add_argument("input_dir", help="Dossier source documents")
    parser.add_argument("--output", "-o", help="Dossier sortie OCR", default=None)
    parser.add_argument("--load-bm25", action="store_true", help="Charger BM25 existant avant ingestion")
    args = parser.parse_args()

    async def main():
        pipeline = IngestionPipeline()
        if args.output:
            pipeline.ocr.output_dir = Path(args.output)
        
        if args.load_bm25:
            pipeline.load_bm25()
        
        try:
            result = await pipeline.run(args.input_dir)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        finally:
            await pipeline.close()

    asyncio.run(main())
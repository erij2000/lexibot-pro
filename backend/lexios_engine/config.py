"""
lexios_engine/config.py 
=============================================================

Optimisé pour: 24GB RAM (MSI) | BGE-M3 Hybrid Search | Groq Qwen-2.5
=============================================================
Ce fichier centralise toute l'intelligence de configuration du backend.
Il évite les duplications (Hardcoding) et protège la RAM du système.
"""

import os
import sys
import logging
import platform
import multiprocessing

# Windows console unicode fix
if platform.system() == "Windows":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# ============================================================================
# 1. LOGGING SETUP (avant toute utilisation)
# ============================================================================

logger = logging.getLogger("lexios.config")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# 2. PATH DEFINITIONS (avant la classe Settings)
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
LIGHTRAG_DIR = BASE_DIR / "lightrag_data"
OCR_OUTPUT_DIR = BASE_DIR / "ocr_output"
CHROMA_DIR = CACHE_DIR / "chroma_db"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
LIGHTRAG_DIR.mkdir(parents=True, exist_ok=True)
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 3. GPU DETECTION & SAFETY
# ============================================================================

try:
    import torch
except ImportError:
    torch = None


def detect_gpu_config() -> Dict[str, Any]:
    config = {
        "device": "cpu",
        "batch_size": 16,
        "half_precision": False,
        "memory_fraction": 0.8,
        "num_workers": multiprocessing.cpu_count() // 2,
        "vram_total": 0
    }

    if torch is not None:
        try:
            # Protection contre CUDA corrompu ou drivers instables
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)
                config["device"] = "cuda"
                config["vram_total"] = round(vram_gb, 2)

                # Ajustement dynamique selon la puissance de la carte graphique
                if vram_gb < 5:  # Cartes type GTX (ex: RTX 2050 4Go)
                    config.update({"batch_size": 8, "half_precision": True, "memory_fraction": 0.6})
                elif vram_gb < 9:  # Cartes type RTX 3060/4060
                    config.update({"batch_size": 24, "half_precision": True, "memory_fraction": 0.75})
                else:  # Cartes High-end
                    config.update({"batch_size": 32, "half_precision": False, "memory_fraction": 0.85})
            else:
                logger.info("ℹ️ CUDA non disponible. Utilisation du CPU.")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de l'accès au GPU : {e}. Fallback CPU sécurisé.")

    return config


_SYS = detect_gpu_config()

# ========================================================================
# 4. SETTINGS CORE CLASS
# ========================================================================

class Settings:
    """Conteneur principal des paramètres de l'application."""

    # --- INFOS PROJET ---
    PROJECT_NAME: str = "Lexibot PRO"
    VERSION: str = "6.4.0"
    ENVIRONMENT: str = os.getenv("ENV", "development")
    START_TIME: str = datetime.now().isoformat()

    # --- LLM (GROQ) ---
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_BACKUP_MODEL: str = "mixtral-8x7b-32768"
    LLM_TIMEOUT: int = 120
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 4096
    GROQ_RETRY_ATTEMPTS: int = 3
    GROQ_RETRY_DELAY: float = 1.5
    GROQ_MIN_INTERVAL: float = 3.0  # req/min (configurable)

    # --- EMBEDDINGS (BGE-M3) ---
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DIM: int = 1024
    EMBED_DEVICE: str = _SYS["device"]
    EMBED_BATCH_SIZE: int = _SYS["batch_size"]
    EMBED_HALF_PRECISION: bool = False # Forcé à False pour la compatibilité XLM-Roberta / BGE-M3
    EMBED_NORMALIZE: bool = True
    EMBED_MAX_LENGTH: int = 8192  # BGE-M3 supporte de longs contextes

    # --- RERANKER (BAAI V2) ---
    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K: int = 10
    RERANK_WEIGHT: float = 0.45  # Influence sur le score final fusionné

    # --- RETRIEVAL LOGIC ---
    RETRIEVAL_TOP_K: int = 50
    RETRIEVAL_FINAL_K: int = 12
    HYBRID_ALPHA: float = 0.5  # 0.5 = Équilibre parfait entre Vecteurs et BM25

    # Options de fusion d'articles: "weighted", "mean", "max"
    ARTICLE_AGGREGATION: str = "weighted"
    ARTICLE_BOOST: float = 1.3

    # --- PIPELINE & ORCHESTRATION ---
    PIPELINE_MAX_CONTEXT_CHARS: int = 15000
    MAX_CONCURRENT_TASKS: int = 4

    # --- CACHE & PERFORMANCE (24GB MSI OPTIMIZED) ---
    USE_CACHE: bool = True
    CACHE_MAX_SIZE: int = 2500
    CACHE_EXPIRATION_HOURS: int = 24
    GPU_MEMORY_FRACTION: float = _SYS["memory_fraction"]
    CLEAR_CACHE_ON_START: bool = False

    # --- DOMAINES JURIDIQUES ---
    DOMAINS: List[str] = [
        "pénal", "civil", "commercial",
        "administratif", "constitutionnel", "social",
        "procédure", "immobilier", "fiscal"
    ]

    # --- DATABASE (CHROMA) ---
    CHROMA_COLLECTION_NAME: str = "lexios_legal_corpus"
    CHROMA_PERSISTENCE: bool = True

    # --- FRONTEND SYNC ---
    CORS_ORIGINS: List[str] = ["http://localhost:4200", "http://127.0.0.1:4200"]

    # --- DEBUG & LOGGING ---
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    SAVE_LLM_TRACES: bool = True

    # --- MISSING SETTINGS ADDED BY AUDIT (P0/P1 Fixes) ---
    ARTICLE_TOP_N: int = 20
    CACHE_TTL_HOURS: int = 24  # Alias for compatibility
    CACHE_DIR: str = str(CACHE_DIR)
    GPU_CLEAR_CACHE_FREQ: int = 50
    LIGHTRAG_ENABLED: bool = True
    LIGHTRAG_WORKING_DIR: str = str(LIGHTRAG_DIR)
    LIGHTRAG_DEFAULT_MODE: str = "hybrid"
    RAGAS_ENABLED: bool = True
    RAGAS_METRICS: str = "faithfulness,answer_relevancy,context_precision,context_recall"
    RAGAS_THRESHOLD: float = 0.7
    USE_SEMANTIC_CHUNKING: bool = False
    OCR_OUTPUT_DIR: str = str(OCR_OUTPUT_DIR)
    OCR_CHUNK_SIZE: int = 600
    OCR_CHUNK_OVERLAP: int = 120
    MAX_CONCURRENT_OCR: int = 2
    OLLAMA_MODEL: str = "qwen2.5:14b"
    OLLAMA_HOST: str = "http://localhost:11434"
    CHROMA_COLLECTION: str = "lexios_legal_corpus"
    CHROMA_DISTANCE: str = "cosine"
    CHROMA_DIR: str = str(CHROMA_DIR)

    def __repr__(self):
        return f"<Settings Lexios v{self.VERSION} | Device: {self.EMBED_DEVICE}>"


# Instance unique des paramètres
settings = Settings()

# ========================================================================
# 5. VALIDATION & MÉTRIQUES (HEALTH CHECK)
# ========================================================================

def validate_environment() -> bool:
    """Vérifie que tout est prêt pour le décollage."""
    issues = []

    if not settings.GROQ_API_KEY:
        issues.append("❌ Clé API GROQ_API_KEY absente.")

    if settings.EMBED_DEVICE == "cpu" and _SYS["vram_total"] > 0:
        issues.append("⚠️ GPU détecté mais non utilisé par Torch (Problème de version?).")

    # Affichage du rapport au démarrage
    print(f"\n{'='*50}")
    print(f"🚀 {settings.PROJECT_NAME} - SYSTEM REPORT")
    print(f"{'='*50}")
    print(f"OS          : {platform.system()} {platform.release()}")
    print(f"RAM Totale  : ~24 GB")
    print(f"AI Device   : {settings.EMBED_DEVICE.upper()}")
    if settings.EMBED_DEVICE == "cuda":
        print(f"VRAM        : {_SYS['vram_total']} GB")
    print(f"Batch Size  : {settings.EMBED_BATCH_SIZE}")
    print(f"LLM Model   : {settings.GROQ_MODEL}")
    print(f"{'='*50}\n")

    if issues:
        for issue in issues:
            logger.warning(issue)
        return False

    logger.info("✅ Configuration validée avec succès.")
    return True


# Lancement du check automatique au chargement du module
_ready = validate_environment()
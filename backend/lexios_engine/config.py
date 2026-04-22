"""
config.py — Lexios Engine Clean v6.0
=====================================
Configuration maître — Architecture Article-Aware Hybrid RAG
Composants actifs: BM25 + ChromaDB (dense) + Article-Aware + Reranker
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# ============================================================================
# GPU DETECTION & SAFETY
# ============================================================================

def detect_gpu_config() -> Dict[str, Any]:
    config = {
        "device": "cpu",
        "batch_size": 16,
        "half_precision": False,
        "memory_fraction": 0.8,
        "clear_cache_freq": 5
    }
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        config["device"] = "cuda"
        if vram_gb < 5.0:
            config.update({
                "batch_size": 8, "half_precision": True,
                "memory_fraction": 0.65, "clear_cache_freq": 3
            })
        elif vram_gb < 8.0:
            config.update({
                "batch_size": 16, "half_precision": True,
                "memory_fraction": 0.75
            })
        else:
            config.update({
                "batch_size": 32, "half_precision": False,
                "memory_fraction": 0.85
            })
    return config

_GPU_CONFIG = detect_gpu_config()

# ============================================================================
# SETTINGS
# ============================================================================

class Settings:

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # Groq
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen-2.5-32b")
    GROQ_BACKUP_MODEL: str = "llama-3.1-70b-versatile"
    GROQ_MAX_TOKENS: int = 4096
    GROQ_TEMPERATURE: float = 0.1
    GROQ_RETRY_ATTEMPTS: int = 3
    GROQ_RETRY_DELAY: float = 1.0

    # Ollama fallback
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")

    # Embeddings
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DIM: int = 1024
    EMBED_DEVICE: str = _GPU_CONFIG["device"]
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", str(_GPU_CONFIG["batch_size"])))
    EMBED_HALF_PRECISION: bool = _GPU_CONFIG["half_precision"]
    EMBED_NORMALIZE: bool = True

    # GPU Safety
    GPU_MEMORY_FRACTION: float = _GPU_CONFIG["memory_fraction"]
    GPU_CLEAR_CACHE_FREQ: int = _GPU_CONFIG["clear_cache_freq"]
    GPU_FALLBACK_ON_OOM: bool = True

    # Reranker
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANK_TOP_K: int = 10

    # Hybrid Retrieval
    RETRIEVAL_TOP_K: int = 50
    RETRIEVAL_FINAL_K: int = 8
    HYBRID_ALPHA: float = 0.6
    ARTICLE_TOP_N: int = 3
    ARTICLE_AGGREGATION: str = "max"

    # ChromaDB ONLY
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./data/chroma_db")
    CHROMA_COLLECTION: str = "lexios_legal"
    CHROMA_DISTANCE: str = "cosine"

    # OCR
    OCR_OUTPUT_DIR: str = os.getenv("OCR_OUTPUT_DIR", "./data/markdown")
    OCR_CHUNK_SIZE: int = 512
    OCR_CHUNK_OVERLAP: int = 100
    MAX_CONCURRENT_OCR: int = 2
    OCR_DEVICE: str = "cpu"

    # Cache
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/cache")
    CACHE_TTL_HOURS: int = 24
    CACHE_THRESHOLD: float = 0.82
    CACHE_MAX_SIZE: int = 10000

    # HyDE
    USE_HYDE: bool = os.getenv("USE_HYDE", "true").lower() == "true"
    HYDE_MODE: str = os.getenv("HYDE_MODE", "combined")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Domains
    DOMAINS: list = ["pénal", "civil", "commercial", "administratif", "constitutionnel", "social"]

    # Déterminisme & Robustesse
    LLM_SEED: int = int(os.getenv("LLM_SEED", "42"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_FREQUENCY_PENALTY: float = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))
    LLM_PRESENCE_PENALTY: float = float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))

    # Pipeline Orchestrator
    PIPELINE_MAX_CONTEXT_CHARS: int = int(os.getenv("PIPELINE_MAX_CONTEXT_CHARS", "12000"))
    PIPELINE_SIMPLE_TOP_K: int = int(os.getenv("PIPELINE_SIMPLE_TOP_K", "30"))
    PIPELINE_COMPLEX_TOP_K: int = int(os.getenv("PIPELINE_COMPLEX_TOP_K", "50"))

    # Query Router
    ROUTER_COMPLEX_THRESHOLD: float = float(os.getenv("ROUTER_COMPLEX_THRESHOLD", "2.0"))
    ROUTER_INTERMEDIATE_THRESHOLD: float = float(os.getenv("ROUTER_INTERMEDIATE_THRESHOLD", "1.0"))

    # Post-Processor
    POSTPROCESS_MIN_LENGTH: int = int(os.getenv("POSTPROCESS_MIN_LENGTH", "30"))
    POSTPROCESS_CONTEXT_OVERLAP_MIN: float = float(os.getenv("POSTPROCESS_CONTEXT_OVERLAP_MIN", "0.1"))
settings = Settings()

# ============================================================================
# VALIDATION
# ============================================================================


    

def validate_config():
    errors = []
    if settings.LLM_PROVIDER == "groq" and not settings.GROQ_API_KEY:
        errors.append("GROQ_API_KEY manquante")
    if settings.EMBED_DEVICE == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA demandé mais non disponible")
    if errors:
        print("❌ ERREURS CRITIQUES:")
        for e in errors:
            print(f"   - {e}")
        sys.exit(1)
    else:
        print("✅ Configuration validée")

validate_config()
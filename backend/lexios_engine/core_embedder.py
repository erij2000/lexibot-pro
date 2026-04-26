"""
core_embedder.py — Lexios Core Embedder
========================================
Shared embedding layer to prevent duplication and circular imports.
Used by: rag_service.py, lightrag_bridge.py
"""

from __future__ import annotations

import time
import re
import hashlib
import asyncio
import logging
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict



import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from config import settings

log = logging.getLogger("lexios.core_embedder")

_executor = ThreadPoolExecutor(max_workers=4)

# ============================================================================
# TOKEN UTILITIES
# ============================================================================

try:
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
except Exception as e:
    log.warning(f"Failed to load BGE-M3 tokenizer: {e}")
    _tokenizer = None


def count_tokens(text: str) -> int:
    if _tokenizer:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    # Better fallback: Arabic ≈ 0.9 chars/token, Latin ≈ 4 chars/token
    ar_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if ar_count / max(len(text), 1) > 0.3:
        return max(1, int(len(text) / 0.9))
    return max(1, len(text) // 4)



def truncate_to_tokens(text: str, max_tokens: int, respect_sentence: bool = True) -> str:
    if not text or count_tokens(text) <= max_tokens:
        return text

    if _tokenizer:
        tokens = _tokenizer.encode(text, add_special_tokens=False)
        truncated = _tokenizer.decode(tokens[:max_tokens])
    else:
        max_chars = max_tokens * 4
        truncated = text[:max_chars]

    if not respect_sentence:
        return truncated

    last_boundary = max(
        truncated.rfind(". "),
        truncated.rfind("? "),
        truncated.rfind("! "),
        truncated.rfind(".\n"),
    )
    if last_boundary > len(truncated) * 0.7:
        return truncated[:last_boundary + 1]
    return truncated


# ============================================================================
# GLOBAL EMBEDDING CACHE (LRU + TTL)
# ============================================================================
from threading import Lock

class GlobalEmbeddingCache:
    def __init__(self, max_size=100, ttl=86400):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                val, ts = self.cache[key]
                if time.time() - ts > self.ttl:
                    del self.cache[key]
                    return None
                self.cache.move_to_end(key)
                return val
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (value, time.time())
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def __len__(self):
        with self.lock:
            return len(self.cache)

# Thread-safe semantic model singleton
_semantic_model_lock = Lock()
_global_semantic_model = None

def semantic_chunk_split(sentences: List[str], threshold: float = 0.75, embedder: Any = None) -> List[str]:
    """Semantic chunking with fallback for long documents. Thread-safe."""
    global _global_semantic_model
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return []

    # Fallback: process in blocks to avoid GPU overhead
    if len(sentences) > 300:
        log.warning(f"Doc too long ({len(sentences)} sentences), chunking by blocks.")
        chunks = []
        for i in range(0, len(sentences), 150):
            block = sentences[i : i + 150]
            chunks.extend(semantic_chunk_split(block, threshold, embedder=embedder))
        return chunks

    if embedder:
        embeddings = embedder.encode_safe(sentences)
    else:
        # Singleton-protect access to _global_semantic_model
        if _global_semantic_model is None:
            with _semantic_model_lock:
                if _global_semantic_model is None:  # Double-check locking
                    _global_semantic_model = SentenceTransformer(
                        "BAAI/bge-m3", device=settings.EMBED_DEVICE
                    )
        embeddings = _global_semantic_model.encode(sentences, normalize_embeddings=True)

    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()

        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))

    return chunks

def _basic_chunk(text: str, chunk_size=600) -> List[str]:
    if _tokenizer:
        tokens = _tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunks.append(_tokenizer.decode(tokens[i:i+chunk_size]))
        return chunks
    else:
        # crude char fallback
        chars = chunk_size * 4
        return [text[i:i+chars] for i in range(0, len(text), chars)]


# ============================================================================
# EMBEDDINGS (RTX 2050 Safe)
# ============================================================================

class RTX2050SafeEmbedder:
    def __init__(self):
        self.device = settings.EMBED_DEVICE
        self.model_name = settings.EMBED_MODEL
        self.batch_size = settings.EMBED_BATCH_SIZE
        self.half = settings.EMBED_HALF_PRECISION and self.device == "cuda"
        log.info(f"🚀 Embedder: {self.model_name} on {self.device}")
        self._load_model()
        self.total_embedded = 0
        self.oom_fallbacks = 0
        self._on_cpu_fallback = False
        self._query_cache = GlobalEmbeddingCache(max_size=settings.CACHE_MAX_SIZE, ttl=86400)
        self._doc_embed_cache = GlobalEmbeddingCache(max_size=settings.CACHE_MAX_SIZE, ttl=86400)

    def _load_model(self):
        try:
            self.model = SentenceTransformer(
                self.model_name, device=self.device, trust_remote_code=True
            )
            if self.half:
                self.model = self.model.half()
                log.info("   ⚡ FP16 activé")
            if self.device == "cuda":
                torch.cuda.set_per_process_memory_fraction(settings.GPU_MEMORY_FRACTION)
                torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"❌ Erreur chargement modèle: {e}")
            raise

    def encode_safe(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.array([])
        all_embeddings = []
        effective_batch = min(self.batch_size, len(texts))
        for i in range(0, len(texts), effective_batch):
            batch = texts[i : i + effective_batch]
            
            # Check global cache
            batch_embs = []
            uncached_texts = []
            uncached_indices = []
            
            for idx, text in enumerate(batch):
                key = hashlib.md5(text.encode()).hexdigest()
                cached = self._doc_embed_cache.get(key)
                if cached is not None:
                    batch_embs.append(cached)
                else:
                    batch_embs.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
                    
            if uncached_texts:
                try:
                    with torch.no_grad():
                        computed_embs = self.model.encode(
                            uncached_texts,
                            batch_size=min(len(uncached_texts), self.batch_size),
                            show_progress_bar=False,
                            normalize_embeddings=settings.EMBED_NORMALIZE,
                            convert_to_numpy=True,
                            device=self.device,
                        )
                    
                    # Store in cache and merge
                    for u_idx, text_idx in enumerate(uncached_indices):
                        emb = computed_embs[u_idx]
                        batch_embs[text_idx] = emb
                        key = hashlib.md5(uncached_texts[u_idx].encode()).hexdigest()
                        self._doc_embed_cache.put(key, emb)
                        
                    self.total_embedded += len(uncached_texts)
                    
                    if (
                        (i // effective_batch + 1) % settings.GPU_CLEAR_CACHE_FREQ == 0
                        and self.device == "cuda"
                    ):
                        torch.cuda.empty_cache()
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        embeddings = self._handle_oom(uncached_texts)
                        for u_idx, text_idx in enumerate(uncached_indices):
                            emb = embeddings[u_idx]
                            batch_embs[text_idx] = emb
                            key = hashlib.md5(uncached_texts[u_idx].encode()).hexdigest()
                            self._doc_embed_cache.put(key, emb)
                    else:
                        log.warning(f"Embedding error, fallback CPU: {e}")
                        self.model.to("cpu")
                        
            all_embeddings.append(np.array(batch_embs))
            
        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def _handle_oom(self, batch: List[str]) -> np.ndarray:
        """Handle OOM by falling back to CPU. Recovery is explicit via recover_gpu()."""
        log.warning("⚠️ OOM GPU, fallback CPU")
        self.oom_fallbacks += 1

        if not self._on_cpu_fallback:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._on_cpu_fallback = True

        with torch.no_grad():
            embeddings = self.model.encode(
                batch, show_progress_bar=False, normalize_embeddings=True, device="cpu"
            )

        return embeddings

    def recover_gpu(self):
        """Explicitly recover GPU access after OOM fallback."""
        if self._on_cpu_fallback and self.device == "cuda" and torch.cuda.is_available():
            try:
                self.model.to(self.device)
                torch.cuda.empty_cache()
                self._on_cpu_fallback = False
                log.info("✅ GPU recovered after OOM")
            except Exception as e:
                log.error(f"❌ GPU recovery failed: {e}")
                self._on_cpu_fallback = True

    def encode_query(self, query: str) -> List[float]:
        key = hashlib.md5(query.encode()).hexdigest()
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        emb = self.encode_safe([query])[0].tolist()
        self._query_cache.put(key, emb)
        return emb

    async def encode_async(self, texts: List[str]) -> List[List[float]]:
        """Async encoding using asyncio.to_thread (no executor double-wrapping)."""
        embs = await asyncio.to_thread(self.encode_safe, texts)
        return embs.tolist() if len(embs) > 0 else []

    async def encode_batch_async(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.encode_safe, texts)

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "device": self.device,
            "model": self.model_name,
            "total_embedded": self.total_embedded,
            "oom_fallbacks": self.oom_fallbacks,
            "query_cache_size": len(self._query_cache),
        }
        if self.device == "cuda" and torch.cuda.is_available():
            stats["memory_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        return stats
# Singleton global
_embedder_instance: Optional[RTX2050SafeEmbedder] = None
_embedder_lock = Lock()

def get_embedder() -> RTX2050SafeEmbedder:
    """Thread-safe singleton for the global embedder."""
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                _embedder_instance = RTX2050SafeEmbedder()
    return _embedder_instance

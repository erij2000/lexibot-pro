"""
cache.py — Smart Semantic Cache v5
===================================
Cache par similarité cosinus (BGE-M3) avec TTL et thread-safety.
Cohérent avec config.py v5.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

log = logging.getLogger("lexios.cache")


# Singleton embedding
_embed_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()


def _get_embedding_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        with _model_lock:
            if _embed_model is None:
                log.info(f"Loading cache embedding model: {settings.EMBED_MODEL}")
                _embed_model = SentenceTransformer(settings.EMBED_MODEL)
    return _embed_model


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / norm) if norm > 0 else 0.0


@dataclass
class CacheEntry:
    __slots__ = ["question", "answer", "embedding", "timestamp", "hits", "ttl", "accessed_at", "category_hint"]
    
    question: str
    answer: str
    embedding: List[float]
    timestamp: float = field(default_factory=time.time)
    hits: int = 0
    ttl: int = field(default_factory=lambda: settings.CACHE_TTL_HOURS * 3600)  # Utilise config
    accessed_at: float = field(default_factory=time.time)
    category_hint: Optional[str] = None

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl
    
    def touch(self):
        self.accessed_at = time.time()
        self.hits += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "hits": self.hits,
            "ttl": self.ttl,
            "accessed_at": self.accessed_at,
            "category_hint": self.category_hint,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheEntry":
        entry = cls(
            question=d["question"],
            answer=d["answer"],
            embedding=d["embedding"],
            timestamp=d.get("timestamp", time.time()),
            category_hint=d.get("category_hint"),
        )
        entry.hits = d.get("hits", 0)
        entry.ttl = d.get("ttl", settings.CACHE_TTL_HOURS * 3600)
        entry.accessed_at = d.get("accessed_at", entry.timestamp)
        return entry


class LexiosCache:
    def __init__(
        self,
        threshold: Optional[float] = None,
        max_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        default_ttl_hours: Optional[int] = None,
    ):
        # Utilise settings par défaut si non spécifié
        self.threshold = threshold if threshold is not None else settings.CACHE_THRESHOLD
        self.max_size = max_size if max_size is not None else 500
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = (default_ttl_hours or settings.CACHE_TTL_HOURS) * 3600
        
        self._lock = threading.RLock()
        self._entries: List[CacheEntry] = []
        self._stats = {"hits": 0, "misses": 0, "total_queries": 0, "evictions": 0}
        
        self._load()

    def _cache_file(self) -> Path:
        return self.cache_dir / "semantic_cache.json"

    def _embed(self, text: str) -> List[float]:
        try:
            model = _get_embedding_model()
            emb = model.encode([text[:512]], show_progress_bar=False)
            return emb[0].tolist()
        except Exception as e:
            log.error(f"Embedding error: {e}")
            return [0.0] * settings.EMBED_DIM

    def _load(self):
        cache_file = self._cache_file()
        if not cache_file.exists():
            return
            
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            entries_raw = data.get("entries", [])
            
            entries = [CacheEntry.from_dict(e) for e in entries_raw]
            self._entries = [e for e in entries if not e.is_expired()]
            
            self._stats = data.get("stats", self._stats)
            log.info(f"Cache loaded: {len(self._entries)} entries")
        except Exception as e:
            log.error(f"Cache load error: {e}")
            self._entries = []

    def _save(self):
        cache_file = self._cache_file()
        temp_file = cache_file.with_suffix(".tmp")
        
        try:
            data = {
                "entries": [e.to_dict() for e in self._entries],
                "stats": self._stats,
                "saved_at": time.time(),
            }
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_file.replace(cache_file)
        except Exception as e:
            log.error(f"Cache save error: {e}")
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    def get(self, question: str) -> Optional[Tuple[str, float]]:
        with self._lock:
            self._stats["total_queries"] += 1
            
            if not self._entries:
                self._stats["misses"] += 1
                return None
            
            try:
                q_emb = self._embed(question)
            except Exception as e:
                log.error(f"Embedding error: {e}")
                return None
            
            best_score = 0.0
            best_entry: Optional[CacheEntry] = None
            
            for entry in self._entries:
                if entry.is_expired():
                    continue
                score = _cosine_similarity(q_emb, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry
            
            if best_score >= self.threshold and best_entry:
                best_entry.touch()
                self._stats["hits"] += 1
                if best_entry.hits % 10 == 0:
                    self._save()
                return best_entry.answer, best_score
            
            self._stats["misses"] += 1
            return None

    def set(self, question: str, answer: str, ttl_hours: Optional[int] = None, category_hint: Optional[str] = None):
        with self._lock:
            if not answer or len(answer.strip()) < 20:
                return
            
            # Déduplication (98% similarité)
            try:
                q_emb = self._embed(question)
                for entry in self._entries:
                    if _cosine_similarity(q_emb, entry.embedding) > 0.98:
                        entry.answer = answer
                        entry.timestamp = time.time()
                        self._save()
                        return
            except Exception as e:
                log.warning(f"Duplicate check error: {e}")
            
            entry = CacheEntry(
                question=question,
                answer=answer,
                embedding=self._embed(question),
                ttl=(ttl_hours or settings.CACHE_TTL_HOURS) * 3600,
                category_hint=category_hint
            )
            
            self._entries.append(entry)
            
            if len(self._entries) > self.max_size:
                self._entries.sort(key=lambda e: (e.accessed_at, e.hits))
                self._entries.pop(0)
                self._stats["evictions"] += 1
            
            self._save()

    def invalidate_by_category(self, category: str) -> int:
        with self._lock:
            before = len(self._entries)
            cat_lower = category.lower()
            
            self._entries = [
                e for e in self._entries 
                if not ((e.category_hint and cat_lower in e.category_hint.lower()) or cat_lower in e.question.lower())
            ]
            removed = before - len(self._entries)
            if removed > 0:
                self._save()
            return removed

    def clear(self, confirm: bool = False):
        if not confirm:
            log.warning("Use confirm=True to clear cache")
            return
            
        with self._lock:
            self._entries = []
            self._stats = {"hits": 0, "misses": 0, "total_queries": 0, "evictions": 0}
            self._save()

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._stats["total_queries"]
            hits = self._stats["hits"]
            hit_rate = hits / total if total > 0 else 0.0
            
            return {
                **self._stats,
                "size": len(self._entries),
                "hit_rate": round(hit_rate, 4),
                "hit_rate_percent": round(hit_rate * 100, 2),
                "threshold": self.threshold,
            }

    def score_report(self) -> str:
        s = self.stats
        return (
            f"Cache: {s['size']} entries | "
            f"Hit Rate: {s['hit_rate_percent']}% | "
            f"Hits: {s['hits']} | Misses: {s['misses']}"
        )


# Singleton global
_cache_instance: Optional[LexiosCache] = None

def get_cache() -> LexiosCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LexiosCache()
    return _cache_instance
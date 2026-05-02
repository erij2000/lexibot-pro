"""
rag_service.py — Lexios Brain Two
=================================================================
Architecture:
  Chunker (token-aware, legal boundaries) → BM25 (unicode) + ChromaDB (dense)
  → Adaptive hybrid scoring → Article aggregation (mean/max + log penalty)
  → Legal boosting → Deduplication → Reranker (BGE-CrossEncoder)
  → [LightRAG context fusion] → LLM Generation (single call)

Features:
  - Token-aware chunking with legal boundary preservation
  - Unicode-aware BM25 (FR + AR legal text)
  - Adaptive alpha (query-dependent hybrid weight)
  - Legal boosting (Article/Art./المادة headers)
  - LightRAG graph context injection (GRAPH LAYER ONLY — no scoring influence)
  - Strict GPU lifecycle, score clamping, sentence-aware truncation
  - Circular-dependency free (core_llm + core_embedder)
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
from sentence_transformers import CrossEncoder

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from config import settings

# FIXED: break circular dependency — import shared core layers
from core_llm import GroqLLMWrapper
from core_embedder import RTX2050SafeEmbedder, count_tokens, truncate_to_tokens, _tokenizer, semantic_chunk_split

log = logging.getLogger("lexios.rag")


# ============================================================================
# CHUNKER (Token-aware, legal boundaries)
# ============================================================================

class LegalChunker:
    """
    Token-aware chunker with overlap and clean legal boundaries.
    Uses real tokenizer when available; falls back to char approximation.
    """

    def __init__(
        self,
        chunk_size: int = 600,      # tokens
        overlap: int = 120,         # tokens
        min_chunk_tokens: int = 100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_tokens = min_chunk_tokens

    def _token_aware_split(self, text: str) -> List[Tuple[int, int]]:
        """Return list of (start, end) char positions respecting token budget."""
        if not text.strip():
            return []

        # Use tokenizer if available for accurate token positions
        if _tokenizer:
            tokens = _tokenizer.encode(text)
            total_tokens = len(tokens)
            if total_tokens <= self.chunk_size:
                return [(0, len(text))]

            decoded_cache = []
            current = ""
            for t in tokens:
                current += _tokenizer.decode([t])
                decoded_cache.append(current)

            boundaries = []
            pos = 0
            while pos < total_tokens:
                end_tok = min(pos + self.chunk_size, total_tokens)
                # Find nearest sentence boundary
                search_start = max(pos, end_tok - 50)
                boundary_tok = end_tok
                for t in range(end_tok, search_start, -1):
                    if t < total_tokens:
                        char_pos = _tokenizer.decode(tokens[t-1:t])
                        if char_pos and char_pos[0] in ".!?\n":
                            boundary_tok = t
                            break

                char_start = len(decoded_cache[pos-1]) if pos > 0 else 0
                char_end = len(decoded_cache[boundary_tok-1])
                boundaries.append((char_start, char_end))

                # Advance with overlap
                overlap_tok = min(self.overlap, boundary_tok - pos - 10)
                pos = max(pos + 1, boundary_tok - overlap_tok)

            return boundaries
        else:
            # Fallback: char approximation (4 chars/token)
            char_size = self.chunk_size * 4
            char_overlap = self.overlap * 4
            boundaries = []
            pos = 0
            while pos < len(text):
                end = min(pos + char_size, len(text))
                if end < len(text):
                    # Find legal boundary
                    search_start = max(pos, end - 200)
                    best = end
                    for m in re.finditer(r"[.!?]\s+", text[search_start:end]):
                        best = search_start + m.end()
                    end = best
                boundaries.append((pos, end))
                pos = max(pos + 1, end - char_overlap)
            return boundaries

    def chunk(
        self,
        text: str,
        doc_uid: str,
        article_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        text = text.strip()
        aid = article_id or doc_uid
        boundaries = self._token_aware_split(text)
        chunks = []

        for idx, (start, end) in enumerate(boundaries):
            segment = text[start:end].strip()
            if not segment:
                continue

            # Skip ultra-small chunks
            if count_tokens(segment) < self.min_chunk_tokens:
                continue

            raw_id = f"{doc_uid}:{aid}:{idx}:{segment[:40]}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "text": segment,
                "normalized_text": self._normalize(segment),
                "article_id": aid,
                "doc_uid": doc_uid,
                "index": idx,
            })

        return chunks

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        return text.strip().lower()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    article_id: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "text": self.text[:500] if len(self.text) > 500 else self.text,
            "scores": {
                "bm25": round(self.bm25_score, 4),
                "dense": round(self.dense_score, 4),
                "hybrid": round(self.hybrid_score, 4),
                "rerank": round(self.rerank_score, 4),
                "final": round(self.final_score, 4),
            },
            "metadata": self.metadata,
            "rank": self.rank,
        }


@dataclass
class Article:
    article_id: str
    title: Optional[str]
    section: Optional[str]
    doc_uid: str
    chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# BM25 SERVICE (Unicode-aware, incremental-safe)
# ============================================================================

class BM25Service:
    """BM25 with unicode-aware tokenization and min-max normalization."""

    def __init__(self, rebuild_threshold: int = 500):
        self.index: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_texts: Dict[str, str] = {}
        self.chunk_map: Dict[int, str] = {}
        self._unrebuilt_count = 0
        self._rebuild_threshold = rebuild_threshold
        self._dirty = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Unicode-aware tokenizer for FR + AR legal text."""
        # Normalize accents and Arabic forms
        text = text.lower()
        # Split on non-alphanumeric + Arabic + French accented chars
        tokens = re.findall(
            r"[\w\u0600-\u06FF\u00C0-\u017F]+",
            text
        )
        # Light stopword removal (very short tokens)
        return [t for t in tokens if len(t) > 1]

    def add_chunks(self, chunks: List[Dict[str, Any]], defer_rebuild: bool = False):
        if not HAS_BM25 or not chunks:
            return

        for c in chunks:
            tokens = self._tokenize(c.get("normalized_text", c.get("text", "")))
            self.tokenized_corpus.append(tokens)
            cid = c["chunk_id"]
            idx = len(self.tokenized_corpus) - 1
            self.chunk_map[idx] = cid
            self.chunk_texts[cid] = c.get("text", "")

        if defer_rebuild:
            self._unrebuilt_count += len(chunks)
            self._dirty = True
            if self._unrebuilt_count >= self._rebuild_threshold:
                self._rebuild()
        else:
            self._rebuild()

    def _rebuild(self):
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
            self._unrebuilt_count = 0
            self._dirty = False
            log.info(f"📚 BM25 rebuilt: {len(self.tokenized_corpus)} chunks")

    def flush(self):
        if self._dirty:
            self._rebuild()

    def search(self, query: str, top_k: int = 50) -> Dict[str, float]:
        if not self.index:
            return {}
        query_tokens = self._tokenize(query)
        scores = self.index.get_scores(query_tokens)

        if len(scores) == 0:
            return {}

        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        if s_max == s_min:
            return {}
        denom = s_max - s_min

        results = {}
        for idx, score in enumerate(scores):
            if score > 0:
                cid = self.chunk_map.get(idx)
                if cid:
                    results[cid] = (float(score) - s_min) / denom

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k])


# ============================================================================
# CHROMADB SERVICE
# ============================================================================

class ChromaService:
    def __init__(self, embedder: RTX2050SafeEmbedder):
        self.embedder = embedder
        self.client = None
        self.collection = None
        if HAS_CHROMA:
            self._init()

    def _init(self):
        try:
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": settings.CHROMA_DISTANCE},
            )
            log.info(f"📊 ChromaDB: {settings.CHROMA_COLLECTION}")
        except Exception as e:
            log.error(f"❌ ChromaDB init: {e}")

    async def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        if not self.collection or not chunks:
            return
        if len(embeddings) != len(chunks):
            log.warning(
                f"Embedding/chunk mismatch: {len(embeddings)} vs {len(chunks)}"
            )
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "doc_uid": c.get("doc_uid", ""),
                "article_id": c.get("article_id", c.get("doc_uid", "")),
                "category": c.get("category", ""),
                "nature": c.get("nature", ""),
                "source_file": c.get("source_file", ""),
                "page": str(c.get("page", "")),
                "section": c.get("section", ""),
            }
            metadatas.append(meta)

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size].tolist(),
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    async def query(
        self, query_text: str, top_k: int = 50, filter_dict: Optional[Dict] = None
    ) -> Dict[str, float]:
        if not self.collection:
            return {}

        query_emb = [self.embedder.encode_query(query_text)]

        try:
            results = self.collection.query(
                query_embeddings=query_emb,
                n_results=top_k,
                where=filter_dict,
                include=["distances"],
            )
        except Exception as e:
            log.error(f"ChromaDB query failed: {e}")
            return {}

        output = {}
        if results and results["ids"] and results["ids"][0]:
            distances = results["distances"][0]
            max_dist = max(distances) if distances else 1.0
            min_dist = min(distances) if distances else 0.0
            denom = max_dist - min_dist if max_dist != min_dist else 1.0
            for cid, dist in zip(results["ids"][0], distances):
                score = (max_dist - dist) / denom
                output[cid] = max(0.0, min(1.0, float(score)))
        return output

    async def get_texts(self, chunk_ids: List[str]) -> Dict[str, str]:
        if not self.collection or not chunk_ids:
            return {}
        try:
            results = self.collection.get(ids=chunk_ids, include=["documents"])
            return {cid: doc for cid, doc in zip(results["ids"], results["documents"])}
        except Exception as e:
            log.warning(f"Chroma get_texts error: {e}")
            return {}


# ============================================================================
# ARTICLE INDEX
# ============================================================================

class ArticleIndex:
    def __init__(self):
        self.articles: Dict[str, Article] = {}
        self.chunk_to_article: Dict[str, str] = {}

    def add_document(self, doc: Dict[str, Any]):
        doc_uid = doc.get("uid", "")
        chunks = doc.get("chunks", [])
        from collections import defaultdict

        groups = defaultdict(list)
        for c in chunks:
            aid = c.get("article_id", doc_uid)
            groups[aid].append(c)

        for aid, chunk_list in groups.items():
            article = Article(
                article_id=aid,
                title=chunk_list[0].get("section", ""),
                section=chunk_list[0].get("section", ""),
                doc_uid=doc_uid,
                chunk_ids=[c["chunk_id"] for c in chunk_list],
                metadata={
                    "source_file": doc.get("source_file", ""),
                    "nature": doc.get("nature", ""),
                    "category": doc.get("drive", {}).get("category", ""),
                },
            )
            self.articles[aid] = article
            for c in chunk_list:
                self.chunk_to_article[c["chunk_id"]] = aid

    def get_article(self, chunk_id: str) -> Optional[Article]:
        aid = self.chunk_to_article.get(chunk_id)
        return self.articles.get(aid) if aid else None


# ============================================================================
# RERANKER SERVICE
# ============================================================================

class RerankerService:
    def __init__(self):
        self.model = None
        self.device = settings.EMBED_DEVICE
        self._loaded = False
        if not settings.USE_RERANKER:
            log.info("Reranker désactivé par config")
            return
        self._load()

    def _load(self):
        try:
            self.model = CrossEncoder(settings.RERANKER_MODEL, device=self.device)
            self._loaded = True
            log.info(f"🎯 Reranker chargé: {settings.RERANKER_MODEL}")
        except Exception as e:
            log.error(f"❌ Reranker load failed: {e}")
            self.model = None

    def rerank(
        self, query: str, chunks: List[RetrievedChunk], top_k: int = None
    ) -> List[RetrievedChunk]:
        if not self.model or not chunks:
            for c in chunks:
                c.final_score = c.hybrid_score
                c.rerank_score = 0.0
            chunks.sort(key=lambda x: x.final_score, reverse=True)
            for i, c in enumerate(chunks):
                c.rank = i + 1
            return chunks[:top_k] if top_k else chunks

        if top_k is None:
            top_k = settings.RERANK_TOP_K

        pairs = [[query, c.text] for c in chunks]
        try:
            scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
            for chunk, score in zip(chunks, scores):
                rerank_norm = 1.0 / (1.0 + math.exp(-float(score)))
                chunk.rerank_score = rerank_norm
                chunk.final_score = (
                    (1 - settings.RERANK_WEIGHT) * chunk.hybrid_score
                    + settings.RERANK_WEIGHT * rerank_norm
                )
        except Exception as e:
            log.warning(f"Reranker inference failed: {e}")
            for c in chunks:
                c.final_score = 0.7 * c.hybrid_score + 0.3 * c.bm25_score

        chunks.sort(key=lambda x: x.final_score, reverse=True)
        for i, c in enumerate(chunks):
            c.rank = i + 1
        return chunks[:top_k]


# ============================================================================
# QUERY UNDERSTANDING (Light)
# ============================================================================

class QueryUnderstanding:
    """Lightweight query analysis for adaptive retrieval."""

    ARTICLE_PATTERN = re.compile(
        r"(?:art\.?|article|المادة)\s*[.\-]?\s*(\d+)", re.IGNORECASE
    )
    DEFINITION_PATTERNS = [
        re.compile(r"^qu\'est-ce que", re.IGNORECASE),
        re.compile(r"^définition", re.IGNORECASE),
        re.compile(r"^c\'est quoi", re.IGNORECASE),
        re.compile(r"^ما هو", re.IGNORECASE),
        re.compile(r"^تعريف", re.IGNORECASE),
    ]
    PROCEDURAL_KEYWORDS = [
        "procédure", "délai", "délais", "compétence", "juridiction",
        "appel", "cassation", "pourvoi", "recours", "مسطرة", "أجل", "محكمة",
    ]

    @classmethod
    def normalize_and_expand(cls, query: str) -> List[str]:
        q_lower = query.lower()
        
        # 1. Contextual Regex (art \d+ -> article \d+)
        q_lower = re.sub(r"\b(art|art\.|article)\b\s*(\d+)", r"article \2", q_lower)
        
        # 2. Basic Mapping
        mappings = {
            "c koi": "qu'est-ce que",
            "c'est quoi": "qu'est-ce que",
            "delai": "délai procédure",
            "délais": "délai procédure",
            "pb": "problème",
            "prq": "pourquoi"
        }
        for k, v in mappings.items():
            q_lower = q_lower.replace(k, v)
            
        original_norm = q_lower.strip()
        
        # 3. Simple expansion
        expanded = original_norm
        if "appel" in original_norm and "cour" not in original_norm:
            expanded += " cour"
        if "divorce" in original_norm and "juge" not in original_norm:
            expanded += " juge"
            
        if expanded != original_norm:
            return [original_norm, expanded]
        return [original_norm]

    @classmethod
    def analyze(cls, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        ar_count = sum(1 for c in query if "\u0600" <= c <= "\u06FF")
        lang = "ar" if ar_count / max(len(query), 1) > 0.3 else "fr"

        # Detect article references
        article_refs = cls.ARTICLE_PATTERN.findall(query)

        # Detect definition query
        is_definition = any(p.match(query) for p in cls.DEFINITION_PATTERNS)

        # Detect procedural query
        is_procedural = any(kw in q_lower for kw in cls.PROCEDURAL_KEYWORDS)

        # Detect precision level
        has_numbers = bool(re.search(r"\d+", query))
        has_article_kw = bool(re.search(r"art\.?|article|المادة", q_lower))
        is_precise = has_numbers and has_article_kw
        word_count = len(query.split())

        # Real Router Logic
        strategy = "hybrid"
        if is_precise:
            strategy = "bm25_heavy"
        elif is_definition:
            strategy = "semantic"
            
        use_lightrag = is_procedural or is_definition or "différence" in q_lower or "relation" in q_lower

        alpha = 0.8
        if strategy == "bm25_heavy":
            alpha = min(0.9, 0.5 + 0.02 * word_count)
        elif strategy == "semantic":
            alpha = 0.4
        else:
            alpha = min(0.8, 0.4 + 0.02 * word_count)

        return {
            "lang": lang,
            "article_refs": article_refs,
            "is_definition": is_definition,
            "is_procedural": is_procedural,
            "is_precise": is_precise,
            "word_count": word_count,
            "strategy": strategy,
            "alpha": alpha,
            "use_lightrag": use_lightrag,
        }


# ============================================================================
# HYBRID RETRIEVER (Production-Grade)
# ============================================================================

class HybridRetriever:
    def __init__(self):
        log.info("🚀 Initialisation Article-Aware Hybrid Retriever v15...")
        self.embedder = RTX2050SafeEmbedder()
        self.llm = GroqLLMWrapper()
        self.bm25 = BM25Service(rebuild_threshold=500)
        self.chroma = ChromaService(self.embedder)
        self.article_index = ArticleIndex()
        self.reranker = RerankerService()
        self.chunker = LegalChunker(chunk_size=600, overlap=120)
        self._lightrag_bridge = None
        self._lightrag_lock = asyncio.Lock()
        self._query_result_cache: Dict[str, Dict] = {}
        log.info("✅ Retriever prêt")

    async def close(self):
        """Ferme proprement les ressources (LLM client, etc.)."""
        if self.llm:
            await self.llm.close()

    # GPU lifecycle helpers
    def _embedder_to_cpu(self):
        if self.embedder.device == "cuda" and torch.cuda.is_available():
            self.embedder.model.to("cpu")
            torch.cuda.empty_cache()

    def _embedder_to_gpu(self):
        if self.embedder.device == "cuda" and torch.cuda.is_available():
            self.embedder.model.to(self.embedder.device)
            torch.cuda.empty_cache()

    def _reranker_to_gpu(self):
        """Move reranker to GPU. MUST call _embedder_to_cpu() first!"""
        if self.reranker.model and self.embedder.device == "cuda" and torch.cuda.is_available():
            # Ensure embedder is off GPU before loading reranker
            self._embedder_to_cpu()
            torch.cuda.empty_cache()
            self.reranker.model.to(self.embedder.device)
            torch.cuda.empty_cache()

    def _reranker_to_cpu(self):
        if self.reranker.model and torch.cuda.is_available():
            self.reranker.model.to("cpu")
            torch.cuda.empty_cache()
            # Optionally remount embedder after reranking
            self._embedder_to_gpu()

    def _compress_context(self, chunks: List[RetrievedChunk], light: bool = False) -> List[RetrievedChunk]:
        """Compress by article: dedup sentences."""
        from collections import defaultdict
        
        by_article = defaultdict(list)
        for c in chunks:
            by_article[c.article_id].append(c)
            
        compressed_chunks = []
        for aid, art_chunks in by_article.items():
            seen = set()
            for c in art_chunks:
                sentences = re.split(r'(?<=[.!?])\s+', c.text)
                kept = []
                for s in sentences:
                    s_norm = re.sub(r'[^\w]', '', s.lower())
                    has_article_ref = bool(re.search(r"(Art\.?|Article|المادة)\s*\d+", s, re.IGNORECASE))
                    
                    if not light and len(s_norm) < 30 and not has_article_ref:
                        continue
                        
                    if s_norm not in seen or has_article_ref:
                        seen.add(s_norm)
                        kept.append(s)
                c.text = " ".join(kept)
                if len(c.text.strip()) > 20:
                    compressed_chunks.append(c)
                    
        compressed_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return compressed_chunks

    async def index_document(self, doc: Dict[str, Any]):
        doc_uid = doc.get("uid", "")
        raw_text = doc.get("raw_text", "")
        existing_chunks = doc.get("chunks", [])

        chunks = []
        if existing_chunks:
            chunks = existing_chunks
        elif raw_text:
            if getattr(settings, "USE_SEMANTIC_CHUNKING", False):
                sentences = re.split(r'(?<=[.!?])\s+', raw_text)
                raw_chunks = semantic_chunk_split(sentences, embedder=self.embedder)
                chunks = []
                for idx, segment in enumerate(raw_chunks):
                    if len(segment.strip()) < 10:
                        continue
                    raw_id = f"{doc_uid}:{doc_uid}:{idx}:{segment[:40]}"
                    chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": segment,
                        "normalized_text": self.chunker._normalize(segment),
                        "article_id": doc_uid,
                        "doc_uid": doc_uid,
                        "index": idx,
                    })
            else:
                chunks = self.chunker.chunk(raw_text, doc_uid=doc_uid)
        else:
            return

        self.article_index.add_document({"uid": doc_uid, "chunks": chunks, **doc})

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode_safe(texts)
        await self.chroma.add_chunks(chunks, embeddings)
        self.bm25.add_chunks(chunks, defer_rebuild=True)

        log.info(
            f"📄 Indexé {len(chunks)} chunks / "
            f"{len(set(c['article_id'] for c in chunks))} articles pour {doc_uid}"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: Optional[Dict] = None,
    ) -> Tuple[List[RetrievedChunk], Dict[str, Any], Optional[Any]]:
        """
        Returns: (chunks, stats, graph_context)
        graph_context is None if LightRAG not triggered.
        """
        # Light query understanding
        query_info = QueryUnderstanding.analyze(query)
        alpha = query_info["alpha"]
        
        if top_k is None:
            top_k = min(20, 5 + query_info["word_count"])

        article_top_n = settings.ARTICLE_TOP_N

        stats = {"timings": {}, "articles": {}, "query_info": query_info}
        start_total = time.time()
        
        # Check Semantic Query Cache (Intent-aware)
        from sentence_transformers import util
        queries = QueryUnderstanding.normalize_and_expand(query)
        primary_query = queries[0]
        
        query_embs = []
        valid_queries = []
        for q in queries:
            emb = self.embedder.encode_query(q)
            skip = False
            for prev_emb in query_embs:
                if util.cos_sim(torch.tensor(prev_emb, dtype=torch.float32), torch.tensor(emb, dtype=torch.float32)).item() > 0.90:
                    skip = True
                    break
            if skip:
                continue
            query_embs.append(emb)
            valid_queries.append(q)
            
        primary_emb = query_embs[0]
        
        cache_key = hashlib.md5(
            (primary_query + str(metadata_filter) + str(top_k)).encode()
        ).hexdigest()
        
        cached = self._query_result_cache.get(cache_key)
        if cached and time.time() - cached["time"] <= 3600:
            if cached["intent"] == query_info["is_precise"]:
                log.info(f"⚡ Query Cache Hit")
                stats["timings"]["total_ms"] = (time.time() - start_total) * 1000
                return cached["chunks"], stats, cached["graph_ctx"]
        elif cached:
            del self._query_result_cache[cache_key]

        # Ensure BM25 is up to date
        self.bm25.flush()

        # 1. BM25 Retrieval (sync loop)
        t0 = time.time()
        bm25_results_list = [self.bm25.search(q, top_k=settings.RETRIEVAL_TOP_K) for q in valid_queries]
        stats["timings"]["bm25_ms"] = (time.time() - t0) * 1000

        # 2. Dense Retrieval (async parallel)
        t0 = time.time()
        dense_tasks = [self.chroma.query(q, top_k=settings.RETRIEVAL_TOP_K, filter_dict=metadata_filter) for q in valid_queries]
        dense_results_list = await asyncio.gather(*dense_tasks)
        stats["timings"]["dense_ms"] = (time.time() - t0) * 1000

        # 3. Hybrid Fusion — normalize on UNION
        weights = [1.0] + [0.8] * (len(valid_queries) - 1)
        w_sum = sum(weights)
        
        merged_bm25 = {}
        merged_dense = {}
        
        all_bm25_ids = set(cid for r in bm25_results_list for cid in r.keys())
        for cid in all_bm25_ids:
            merged_bm25[cid] = sum(w * r.get(cid, 0.0) for w, r in zip(weights, bm25_results_list)) / w_sum
            
        all_dense_ids = set(cid for r in dense_results_list for cid in r.keys())
        for cid in all_dense_ids:
            merged_dense[cid] = sum(w * r.get(cid, 0.0) for w, r in zip(weights, dense_results_list)) / w_sum

        all_ids = list(set(merged_bm25.keys()) | set(merged_dense.keys()))
        all_bm25_scores = [merged_bm25.get(cid, 0.0) for cid in all_ids]
        all_dense_scores = [merged_dense.get(cid, 0.0) for cid in all_ids]

        b_min, b_max = min(all_bm25_scores) if all_bm25_scores else 0.0, max(all_bm25_scores) if all_bm25_scores else 1.0
        d_min, d_max = min(all_dense_scores) if all_dense_scores else 0.0, max(all_dense_scores) if all_dense_scores else 1.0
        
        b_denom = b_max - b_min if b_max != b_min else 1.0
        d_denom = d_max - d_min if d_max != d_min else 1.0

        chunk_scores: Dict[str, Dict[str, float]] = {}
        for cid in all_ids:
            b_raw = merged_bm25.get(cid, 0.0)
            d_raw = merged_dense.get(cid, 0.0)
            
            # Normalization on UNION
            b = (b_raw - b_min) / b_denom
            d = (d_raw - d_min) / d_denom

            # Safe clamp [0,1]
            b = max(0.0, min(1.0, b))
            d = max(0.0, min(1.0, d))
            hybrid = max(0.0, min(1.0, (alpha * (b ** 1.2)) + ((1 - alpha) * (d ** 1.1))))
            
            # Recency/Source Weighting
            article = self.article_index.get_article(cid)
            if article and article.metadata:
                if "date" in article.metadata or "year" in article.metadata:
                    hybrid *= 1.05
                    hybrid = min(1.0, hybrid)
            
            text_for_year = self.bm25.chunk_texts.get(cid) or ""
            year_match = re.search(r"\b(19|20)\d{2}\b", text_for_year)
            if year_match:
                year = int(year_match.group(0))
                # More subtle and legally neutral boost
                boost = 1 + max(0, year - 2020) * 0.001
                hybrid = min(1.0, hybrid * boost)

            chunk_scores[cid] = {"bm25": b, "dense": d, "hybrid": hybrid}

        # 4. Article Aggregation with penalty
        article_scores: Dict[str, List[float]] = {}
        article_chunks_map: Dict[str, List[str]] = {}

        for cid, scores in chunk_scores.items():
            article = self.article_index.get_article(cid)
            aid = article.article_id if article else f"doc_{cid}"
            article_scores.setdefault(aid, []).append(scores["hybrid"])
            article_chunks_map.setdefault(aid, []).append(cid)

        final_article_scores = {}
        for aid, scores_list in article_scores.items():
            # Align with settings.ARTICLE_AGGREGATION
            if settings.ARTICLE_AGGREGATION == "mean":
                agg = sum(scores_list) / len(scores_list)
            elif settings.ARTICLE_AGGREGATION == "max":
                agg = max(scores_list)
            else:  # weighted (0.6 mean + 0.4 max)
                agg = 0.6 * (sum(scores_list) / len(scores_list)) + 0.4 * max(scores_list)
            
            std = np.std(scores_list) if len(scores_list) > 1 else 0.0
            penalty = math.log(1 + len(scores_list))
            
            # Rule #2: Aggregation formula with std and penalty
            final_article_scores[aid] = (agg + 0.2 * (1 - std)) / penalty

        sorted_articles = sorted(
            final_article_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_articles = [aid for aid, _ in sorted_articles[:article_top_n]]

        stats["articles"]["total"] = len(sorted_articles)
        stats["articles"]["selected"] = len(top_articles)
        stats["articles"]["top_scores"] = [
            round(s, 3) for _, s in sorted_articles[:article_top_n]
        ]

        # Pre-fetch Chroma texts
        all_cids = [
            cid for aid in top_articles for cid in article_chunks_map.get(aid, [])
        ]
        chroma_texts = await self.chroma.get_texts(all_cids)

        # 5. Chunk Selection with deduplication + legal boosting
        candidates: List[RetrievedChunk] = []
        seen_hashes = set()

        for aid in top_articles:
            for cid in article_chunks_map.get(aid, []):
                sc = chunk_scores[cid]
                article = self.article_index.get_article(cid)

                text = (
                    chroma_texts.get(cid)
                    or self.bm25.chunk_texts.get(cid)
                    or "[EMPTY_CHUNK]"
                )

                if text == "[EMPTY_CHUNK]":
                    log.warning(f"Empty chunk fallback triggered for {cid}")

                # Legal boosting: boost if chunk contains article headers
                boost = 1.0
                if re.search(r"(Art\.?|Article|المادة)\s*\d+", text, re.IGNORECASE):
                    boost = settings.ARTICLE_BOOST

                if "code pénal" in text.lower():
                    boost *= 1.1
                if article and article.title:
                    boost *= 1.05

                boosted_hybrid = min(1.0, sc["hybrid"] * boost)

                content_hash = hashlib.md5(f"{text}:{aid}".encode()).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                candidates.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        text=text,
                        article_id=aid,
                        bm25_score=sc["bm25"],
                        dense_score=sc["dense"],
                        hybrid_score=boosted_hybrid,
                        metadata=article.metadata if article else {},
                    )
                )

        # Semantic Coherence Score - Batch encoding async to prevent event loop blocking
        texts = [c.text for c in candidates]
        # Simulate async encoding via thread pool if embedder is sync
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.embedder.encode_safe, texts)
        
        for c, emb in zip(candidates, embeddings):
            # Rule #3: Ensure device consistency
            emb_tensor = torch.tensor(emb, device=primary_emb.device if hasattr(primary_emb, 'device') else 'cpu')
            similarity = util.cos_sim(primary_emb, emb_tensor)
            c.hybrid_score *= (0.8 + 0.2 * similarity.item())

        # 12. Hard Negative BEFORE rerank
        candidates = [c for c in candidates if c.hybrid_score > 0.2]

        candidates.sort(key=lambda x: (-x.hybrid_score, x.chunk_id))
        # Dynamic cutoff: top_k * 5, max 100
        pre_rerank_limit = min(max(top_k * 5, 20), 100)
        candidates = candidates[:pre_rerank_limit]
        
        # 9. Compression AVANT rerank
        candidates = self._compress_context(candidates, light=True)

        # 6. Rerank with GPU lifecycle
        t0 = time.time()
        if self.embedder.device == "cuda" and self.reranker.model:
            if torch.cuda.memory_allocated() < 0.7 * torch.cuda.get_device_properties(0).total_memory:
                # Keep both on GPU
                pass
            else:
                self._embedder_to_cpu()
                self._reranker_to_gpu()

        try:
            final_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
            final_chunks = self._compress_context(final_chunks, light=False)
        finally:
            # Rule #3: Embedder and Reranker must NEVER coexist on GPU long-term
            if self.embedder.device == "cuda":
                self._reranker_to_cpu()
                self._embedder_to_gpu()
                torch.cuda.empty_cache()

        # Sort by final score deterministically
        final_chunks.sort(key=lambda x: (-x.final_score, x.chunk_id))
        
        # Hard Negative Control
        if final_chunks:
            top_score = final_chunks[0].final_score
            mean_score = sum(c.final_score for c in final_chunks) / len(final_chunks)
            threshold = max(0.3, mean_score * 0.5)
            final_chunks = [c for c in final_chunks if c.final_score >= threshold]

        stats["timings"]["rerank_ms"] = (time.time() - t0) * 1000

        # 7. LightRAG context (optional, GRAPH LAYER ONLY — no scoring influence)
        graph_context = None
        if settings.LIGHTRAG_ENABLED:
            try:
                from lightrag_bridge import get_lightrag_bridge
                if self._lightrag_bridge is None:
                    async with self._lightrag_lock:
                        if self._lightrag_bridge is None:
                            self._lightrag_bridge = await get_lightrag_bridge(
                                self.embedder, self.llm
                            )
                trigger = await self._lightrag_bridge.analyze_trigger(query)
                if trigger.use_lightrag or query_info.get("use_lightrag"):
                    t0 = time.time()
                    graph_context = await self._lightrag_bridge.get_graph_context(
                        query, mode=trigger.suggested_mode, top_k=top_k
                    )
                    stats["timings"]["lightrag_ms"] = (time.time() - t0) * 1000
                    log.info(f"🌐 LightRAG triggered: {trigger.reason}")
            except Exception as e:
                log.warning(f"LightRAG retrieval skipped: {e}")

        stats["timings"]["total_ms"] = (time.time() - start_total) * 1000

        # Save to Cache (metadata only, NO embeddings to avoid memory bloat)
        self._query_result_cache[cache_key] = {
            "time": time.time(),
            "intent": query_info["is_precise"],
            "chunk_ids": [c.chunk_id for c in final_chunks],
            "article_ids": list(set(c.article_id for c in final_chunks)),
            "graph_ctx_key": str(hash(graph_context.to_context_string())) if graph_context else None
        }
        
        # LRU Eviction
        if len(self._query_result_cache) > 100:
            oldest_key = min(self._query_result_cache, key=lambda k: self._query_result_cache[k]["time"])
            del self._query_result_cache[oldest_key]

        return final_chunks, stats, graph_context

    async def query_for_eval(
        self,
        question: str,
        top_k: int = None,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        chunks, stats, graph_ctx = await self.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter
        )

        contexts = []
        for c in chunks:
            contexts.append(
                {
                    "text": c.text,
                    "article_id": c.article_id,
                    "chunk_id": c.chunk_id,
                    "scores": {
                        "bm25": c.bm25_score,
                        "dense": c.dense_score,
                        "hybrid": c.hybrid_score,
                        "rerank": c.rerank_score,
                        "final": c.final_score,
                    },
                    "metadata": c.metadata,
                }
            )

        context_text = "\n\n---\n\n".join(c.text[:600] for c in chunks)

        return {
            "question": question,
            "contexts": contexts,
            "context_text": context_text,
            "chunks_count": len(chunks),
            "articles_selected": stats["articles"]["selected"],
            "timings": stats["timings"],
            "sources": list(set(c.article_id for c in chunks)),
            "graph_context": graph_ctx.to_context_string() if graph_ctx else None,
        }

    async def generate_with_context(
        self,
        question: str,
        context_data: Dict[str, Any],
        system_prompt: Optional[str] = None,
        graph_context: Optional[str] = None,
    ) -> str:
        context_text = context_data.get("context_text", "")
        sources = context_data.get("sources", [])
        graph_text = context_data.get("graph_context", "")

        sources_str = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources[:5])])

        # Token-aware truncation
        context_truncated = truncate_to_tokens(context_text, 2500, respect_sentence=True)

        # ADDED: Fuse graph context if present (GRAPH LAYER ONLY)
        if graph_context:
            graph_truncated = truncate_to_tokens(graph_context, 1200, respect_sentence=True)
            fused_context = f"""{context_truncated}

=== CONTEXTE RELATIONNEL (GRAPHE) ===
{graph_truncated}"""
        else:
            fused_context = context_truncated

        # Use PromptBuilder for consistent prompting
        from lexios_engine.utils.prompt_builder import PromptBuilder
        prompt = PromptBuilder.build_synthesis_prompt(
            language="fr" if "\u0600" not in question else "ar",
            context=fused_context,
            sources_count=len(sources)
        )
        # Placeholder replacement if needed by PromptBuilder
        if "[USER_QUESTION_PLACEHOLDER]" in prompt:
            prompt = prompt.replace("[USER_QUESTION_PLACEHOLDER]", question)
        else:
            prompt += f"\n\nQUESTION: {question}"

        # RÉPONSE:

        try:
            return await self.llm(
                prompt,
                system_prompt=system_prompt
                or "Tu es un juriste tunisien senior, précis et prudent.",
                temperature=0.1,
            )
        except Exception as e:
            log.error(f"Generation failed: {e}")
            return f"[Erreur génération: {e}]"

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "chroma_ok": self.chroma.collection is not None,
            "bm25_docs": len(self.bm25.tokenized_corpus),
            "bm25_pending": self.bm25._unrebuilt_count,
            "articles_indexed": len(self.article_index.articles),
            "embedder_stats": self.embedder.get_stats(),
            "reranker_loaded": self.reranker._loaded,
            "gpu_memory_mb": (
                torch.cuda.memory_allocated() / 1e6
                if (settings.EMBED_DEVICE == "cuda" and torch.cuda.is_available())
                else 0
            ),
        }


# ============================================================================
# FACADE: LEXIOS RAG
# ============================================================================

class LexiosRAG:
    def __init__(self):
        self.retriever = HybridRetriever()
        log.info("🧠 LexiosRAG v15 initialisé")

    async def close(self):
        """Ferme proprement les ressources."""
        await self.retriever.close()

    async def ingest_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await self.retriever.index_document(doc)

            # ADDED: Index into LightRAG graph (GRAPH LAYER ONLY)
            if settings.LIGHTRAG_ENABLED:
                try:
                    from lightrag_bridge import get_lightrag_bridge
                    # FIXED: pass embedder + llm explicitly, no circular dep
                    lightrag = await get_lightrag_bridge(
                        self.retriever.embedder, self.retriever.llm
                    )
                    await lightrag.index_document(doc)
                except Exception as e:
                    log.warning(f"LightRAG indexation skipped: {e}")

            return {"status": "success", "chunks_indexed": len(doc.get("chunks", []))}
        except Exception as e:
            log.error(f"Ingestion error: {e}")
            return {"status": "error", "message": str(e)}

    async def query(
        self,
        question: str,
        top_k: int = None,
        generate_answer: bool = True,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        chunks, stats, graph_ctx = await self.retriever.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter
        )

        context_parts = []
        for c in chunks:
            context_parts.append(f"[{c.article_id}] {c.text[:600]}...")
        context = "\n\n---\n\n".join(context_parts)
        
        # Anti-Hallucination Lock
        top_3_mean = sum(c.final_score for c in chunks[:3]) / max(1, len(chunks[:3])) if chunks else 0.0
        scores_std = float(np.std([c.final_score for c in chunks[:5]])) if len(chunks) > 1 else 1.0
        
        confidence = 0.0
        if chunks and len(chunks) >= 2:
            confidence = (0.5 * chunks[0].final_score) + (0.3 * top_3_mean) + (0.2 * (1 - scores_std))
        
        if not chunks or len(chunks) < 2 or confidence < 0.45:
            log.warning("🔒 Anti-hallucination lock triggered (low confidence or flat distribution)")
            return {
                "answer": "Je ne dispose pas d'informations suffisantes dans mes sources juridiques pour répondre avec certitude à cette question.",
                "chunks": [c.to_dict() for c in chunks],
                "context": context,
                "stats": {
                    "total_time_ms": round(stats["timings"]["total_ms"], 1),
                    "chunks_found": len(chunks),
                    "articles_selected": stats["articles"]["selected"],
                    "sources": list(set(c.article_id for c in chunks)),
                },
                "query": question,
            }

        answer = ""
        if generate_answer and chunks:
            answer = await self._generate_response(question, context, chunks, graph_ctx)

        return {
            "answer": answer,
            "chunks": [c.to_dict() for c in chunks],
            "context": context,
            "stats": {
                "total_time_ms": round(stats["timings"]["total_ms"], 1),
                "chunks_found": len(chunks),
                "articles_selected": stats["articles"]["selected"],
                "sources": list(set(c.article_id for c in chunks)),
            },
            "query": question,
        }

    async def _generate_response(
        self,
        question: str,
        context: str,
        chunks: List[RetrievedChunk],
        graph_ctx: Optional[Any] = None,
    ) -> str:
        sources_str = "\n".join(
            [
                f"[{i+1}] {c.article_id} (score: {c.final_score:.2f})"
                for i, c in enumerate(chunks[:5])
            ]
        )
        context_truncated = truncate_to_tokens(context, 2500, respect_sentence=True)

        # ADDED: Inject graph context (GRAPH LAYER ONLY — no scoring influence)
        if graph_ctx and graph_ctx.raw_context:
            graph_text = graph_ctx.to_context_string(max_length=1200)
            fused_context = f"""=== PRIORITÉ 1 — ARTICLES (SOURCE PRINCIPALE) ===
{context_truncated}

=== PRIORITÉ 2 — RELATIONS (CONTEXTE SUPPLÉMENTAIRE) ===
{graph_text}"""
        else:
            fused_context = f"""=== PRIORITÉ 1 — ARTICLES (SOURCE PRINCIPALE) ===
{context_truncated}"""

        # Use PromptBuilder for consistent prompting
        from lexios_engine.utils.prompt_builder import PromptBuilder
        prompt = PromptBuilder.build_synthesis_prompt(
            language="fr" if "\u0600" not in question else "ar",
            context=fused_context,
            sources_count=len(chunks)
        )
        if "[USER_QUESTION_PLACEHOLDER]" in prompt:
            prompt = prompt.replace("[USER_QUESTION_PLACEHOLDER]", question)
        else:
            prompt += f"\n\nQUESTION: {question}"

        # RÉPONSE:
        try:
            return await self.retriever.llm(
                prompt,
                system_prompt="Tu es un juriste tunisien senior, précis et prudent.",
                temperature=0.1,
            )
        except Exception as e:
            log.error(f"Génération réponse échec: {e}")
            return f"[Erreur génération: {e}]\n\nContexte:\n{context[:1000]}..."

    async def query_for_eval(
        self,
        question: str,
        top_k: int = None,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        chunks, stats, graph_ctx = await self.retriever.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter
        )
        return {
            "contexts": chunks,
            "sources": list(set(getattr(c, "article_id", "unknown") for c in chunks)),
            "stats": stats
        }

    async def generate_with_context(
        self,
        question: str,
        context_data: Dict[str, Any],
        system_prompt: Optional[str] = None,
        graph_context: Optional[str] = None,
    ) -> str:
        if graph_context:
            context_data = {**context_data, "graph_context": graph_context}
        return await self.retriever.generate_with_context(
            question, context_data, system_prompt
        )

    def get_health_status(self) -> Dict[str, Any]:
        return self.retriever.get_health_status()

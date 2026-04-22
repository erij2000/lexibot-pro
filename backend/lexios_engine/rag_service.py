"""
rag_service.py — Lexios Brain Two Ultimate v13 (Clean Article-Aware)
=====================================================================
Architecture:
  BM25 (sparse) + ChromaDB (dense) → Hybrid per chunk
  → Article aggregation (max/mean) → Top articles
  → Chunk selection → BGE-Reranker → LLM Generation

Retrait: FAISS, LegalBooster, RRF fantôme, score/10
Ajout: ArticleIndex, RerankerService, BM25 min-max norm
"""

from __future__ import annotations

import math
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import httpx
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

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

log = logging.getLogger("lexios.rag")
_executor = ThreadPoolExecutor(max_workers=4)


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
            "rank": self.rank
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

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
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
            batch = texts[i:i + effective_batch]
            try:
                embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    normalize_embeddings=settings.EMBED_NORMALIZE,
                    convert_to_numpy=True,
                    device=self.device
                )
                all_embeddings.append(embeddings)
                self.total_embedded += len(batch)
                if (i // effective_batch + 1) % settings.GPU_CLEAR_CACHE_FREQ == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._handle_oom(batch, all_embeddings)
                else:
                    raise
        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def _handle_oom(self, batch: List[str], accumulator: List[np.ndarray]):
        log.warning("⚠️ OOM GPU, fallback CPU")
        self.oom_fallbacks += 1
        self.model.to("cpu")
        torch.cuda.empty_cache()
        embeddings = self.model.encode(batch, show_progress_bar=False, normalize_embeddings=True, device="cpu")
        accumulator.append(embeddings)
        if self.device == "cuda":
            self.model.to(self.device)

    async def encode_async(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        embs = await loop.run_in_executor(_executor, lambda: self.encode_safe(texts))
        return embs.tolist() if len(embs) > 0 else []

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "device": self.device,
            "model": self.model_name,
            "total_embedded": self.total_embedded,
            "oom_fallbacks": self.oom_fallbacks,
        }
        if self.device == "cuda":
            stats["memory_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        return stats


# ============================================================================
# GROQ LLM
# ============================================================================

class GroqLLMWrapper:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY requise")
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = settings.GROQ_MODEL
        self.backup_model = settings.GROQ_BACKUP_MODEL
        self.timeout = settings.LLM_TIMEOUT
        self.last_request_time = 0
        self.min_interval = 60.0 / 30.0

    async def __call__(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       use_backup: bool = False) -> str:
        model = self.backup_model if use_backup else self.model
        temp = temperature or settings.GROQ_TEMPERATURE
        tokens = max_tokens or settings.GROQ_MAX_TOKENS
        await self._respect_rate_limit()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "top_p": 0.9,
            "stream": False
        }

        last_error = None
        for attempt in range(settings.GROQ_RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json=payload
                    )
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        log.warning(f"Rate limit, attente {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not content or len(content) < 10:
                        raise ValueError("Réponse vide")
                    self.last_request_time = time.time()
                    return content
            except httpx.TimeoutException:
                last_error = "Timeout"
                await asyncio.sleep(settings.GROQ_RETRY_DELAY)
            except Exception as e:
                last_error = str(e)
                if attempt < settings.GROQ_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(settings.GROQ_RETRY_DELAY)

        if not use_backup and self.backup_model != self.model:
            log.warning("Fallback sur modèle backup...")
            return await self.__call__(prompt, system_prompt, temperature, max_tokens, use_backup=True)
        raise Exception(f"Groq échec après {settings.GROQ_RETRY_ATTEMPTS} tentatives: {last_error}")

    async def _respect_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)


# ============================================================================
# BM25 SERVICE (Clean Normalization)
# ============================================================================

class BM25Service:
    """BM25 avec min-max normalization standard."""

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_texts: Dict[str, str] = {}
        self.chunk_map: Dict[int, str] = {}

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not HAS_BM25:
            return
        self.tokenized_corpus = []
        self.chunk_map = {}
        self.chunk_texts = {}

        for i, c in enumerate(chunks):
            tokens = c.get("normalized_text", c.get("text", "")).lower().split()
            self.tokenized_corpus.append(tokens)
            cid = c["chunk_id"]
            self.chunk_map[i] = cid
            self.chunk_texts[cid] = c.get("text", "")

        self.index = BM25Okapi(self.tokenized_corpus)
        log.info(f"📚 BM25 index: {len(self.tokenized_corpus)} chunks")

    def search(self, query: str, top_k: int = 50) -> Dict[str, float]:
        if not self.index:
            return {}
        query_tokens = query.lower().split()
        scores = self.index.get_scores(query_tokens)

        if len(scores) == 0:
            return {}

        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        denom = s_max - s_min + 1e-6

        results = {}
        for idx, score in enumerate(scores):
            if score > 0:
                cid = self.chunk_map[idx]
                results[cid] = (float(score) - s_min) / denom

        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return sorted_results


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
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": settings.CHROMA_DISTANCE}
            )
            log.info(f"📊 ChromaDB: {settings.CHROMA_COLLECTION}")
        except Exception as e:
            log.error(f"❌ ChromaDB init: {e}")

    async def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        if not self.collection or not chunks:
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
                "section": c.get("section", "")
            }
            metadatas.append(meta)

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size].tolist(),
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    async def query(self, query_text: str, top_k: int = 50,
                    filter_dict: Optional[Dict] = None) -> Dict[str, float]:
        if not self.collection:
            return {}
        query_emb = self.embedder.encode_safe([query_text])
        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=top_k,
            where=filter_dict,
            include=["distances"]
        )
        output = {}
        if results and results["ids"]:
            for cid, dist in zip(results["ids"][0], results["distances"][0]):
                if settings.CHROMA_DISTANCE == "cosine":
                    score = 1 - (dist / 2)
                else:
                    score = 1 / (1 + dist)
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
    """Index article-aware: article_id -> chunks + métadonnées."""

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
                    "category": doc.get("drive", {}).get("category", "")
                }
            )
            self.articles[aid] = article
            for c in chunk_list:
                self.chunk_to_article[c["chunk_id"]] = aid

    def get_article(self, chunk_id: str) -> Optional[Article]:
        aid = self.chunk_to_article.get(chunk_id)
        return self.articles.get(aid) if aid else None


# ============================================================================
# RERANKER SERVICE (Real Implementation)
# ============================================================================

class RerankerService:
    """BGE-Reranker cross-encoder pour affiner le top-k."""

    def __init__(self):
        self.model = None
        self.device = settings.EMBED_DEVICE
        if not settings.USE_RERANKER:
            log.info("Reranker désactivé par config")
            return
        self._load()

    def _load(self):
        try:
            self.model = CrossEncoder(settings.RERANKER_MODEL, device=self.device)
            log.info(f"🎯 Reranker chargé: {settings.RERANKER_MODEL}")
        except Exception as e:
            log.error(f"❌ Reranker load failed: {e}")
            self.model = None

    def rerank(self, query: str, chunks: List[RetrievedChunk],
               top_k: int = None) -> List[RetrievedChunk]:
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
            scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
                chunk.final_score = 0.7 * chunk.hybrid_score + 0.3 * chunk.rerank_score
        except Exception as e:
            log.warning(f"Reranker inference failed: {e}")
            for c in chunks:
                c.final_score = c.hybrid_score

        chunks.sort(key=lambda x: x.final_score, reverse=True)
        for i, c in enumerate(chunks):
            c.rank = i + 1
        return chunks[:top_k]


# ============================================================================
# HYBRID RETRIEVER (Article-Aware)
# ============================================================================

class HybridRetriever:
    """
    Retrieval hybride article-aware:
      1. Chunk scoring (BM25 + Dense)
      2. Article aggregation (max/mean)
      3. Top-N articles
      4. Chunk selection intra-articles
      5. Rerank
    """

    def __init__(self):
        log.info("🚀 Initialisation Article-Aware Hybrid Retriever...")
        self.embedder = RTX2050SafeEmbedder()
        self.llm = GroqLLMWrapper()
        self.bm25 = BM25Service()
        self.chroma = ChromaService(self.embedder)
        self.article_index = ArticleIndex()
        self.reranker = RerankerService()
        self.chunk_texts: Dict[str, str] = {}
        log.info("✅ Retriever prêt")

    async def index_document(self, doc: Dict[str, Any]):
        doc_uid = doc.get("uid", "")
        chunks = doc.get("chunks", [])
        if not chunks:
            return

        for c in chunks:
            if "article_id" not in c:
                c["article_id"] = doc_uid
            c["doc_uid"] = doc_uid
            self.chunk_texts[c["chunk_id"]] = c.get("text", "")

        self.article_index.add_document(doc)

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode_safe(texts)
        await self.chroma.add_chunks(chunks, embeddings)

        self.bm25.add_chunks(chunks)

        log.info(f"📄 Indexé {len(chunks)} chunks / {len(set(c['article_id'] for c in chunks))} articles pour {doc_uid}")

    async def retrieve(self, query: str, top_k: int = None,
                       metadata_filter: Optional[Dict] = None) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
        if top_k is None:
            top_k = settings.RETRIEVAL_FINAL_K

        alpha = settings.HYBRID_ALPHA
        article_top_n = settings.ARTICLE_TOP_N
        agg_mode = settings.ARTICLE_AGGREGATION

        stats = {"timings": {}, "articles": {}}
        start_total = time.time()

        # 1. BM25 Retrieval
        t0 = time.time()
        bm25_results = self.bm25.search(query, top_k=settings.RETRIEVAL_TOP_K)
        stats["timings"]["bm25_ms"] = (time.time() - t0) * 1000

        # 2. Dense Retrieval
        t0 = time.time()
        dense_results = await self.chroma.query(
            query, top_k=settings.RETRIEVAL_TOP_K, filter_dict=metadata_filter
        )
        stats["timings"]["dense_ms"] = (time.time() - t0) * 1000

        # 3. Hybrid Fusion (chunk level)
        all_ids = set(bm25_results.keys()) | set(dense_results.keys())
        chunk_scores: Dict[str, Dict[str, float]] = {}
        for cid in all_ids:
            b = bm25_results.get(cid, 0.0)
            d = dense_results.get(cid, 0.0)
            chunk_scores[cid] = {
                "bm25": b,
                "dense": d,
                "hybrid": alpha * b + (1 - alpha) * d
            }

        # 4. Article Aggregation
        article_scores: Dict[str, List[float]] = {}
        article_chunks_map: Dict[str, List[str]] = {}

        for cid, scores in chunk_scores.items():
            article = self.article_index.get_article(cid)
            aid = article.article_id if article else "unknown"
            if aid not in article_scores:
                article_scores[aid] = []
                article_chunks_map[aid] = []
            article_scores[aid].append(scores["hybrid"])
            article_chunks_map[aid].append(cid)

        final_article_scores = {}
        for aid, scores in article_scores.items():
            if agg_mode == "max":
                final_article_scores[aid] = max(scores)
            else:
                final_article_scores[aid] = sum(scores) / len(scores)

        # 5. Select Top Articles
        sorted_articles = sorted(final_article_scores.items(), key=lambda x: x[1], reverse=True)
        top_articles = [aid for aid, _ in sorted_articles[:article_top_n]]

        stats["articles"]["total"] = len(sorted_articles)
        stats["articles"]["selected"] = len(top_articles)
        stats["articles"]["top_scores"] = [round(s, 3) for _, s in sorted_articles[:article_top_n]]

        # 6. Chunk Selection inside Top Articles
        candidates: List[RetrievedChunk] = []
        seen = set()
        for aid in top_articles:
            cids = article_chunks_map.get(aid, [])
            for cid in cids:
                if cid in seen:
                    continue
                seen.add(cid)
                sc = chunk_scores[cid]
                article = self.article_index.get_article(cid)
                text = self.chunk_texts.get(cid) or self.bm25.chunk_texts.get(cid, "")
                rc = RetrievedChunk(
                    chunk_id=cid,
                    text=text,
                    article_id=aid,
                    bm25_score=sc["bm25"],
                    dense_score=sc["dense"],
                    hybrid_score=sc["hybrid"],
                    metadata=article.metadata if article else {}
                )
                candidates.append(rc)

        candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
        candidates = candidates[:max(top_k * 3, 20)]

        # 7. Rerank
        t0 = time.time()
        final_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
        stats["timings"]["rerank_ms"] = (time.time() - t0) * 1000
        stats["timings"]["total_ms"] = (time.time() - start_total) * 1000

        return final_chunks, stats


    # =========================================================================
    # RAGAS EVALUATION SUPPORT (v13.1)
    # =========================================================================

    async def query_for_eval(self, question: str, top_k: int = None,
                             metadata_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Version de query optimisée pour l'évaluation RAGAS.
        Retourne les données brutes nécessaires aux métriques.
        """
        chunks, stats = await self.retriever.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter
        )

        contexts = []
        for c in chunks:
            contexts.append({
                "text": c.text,
                "article_id": c.article_id,
                "chunk_id": c.chunk_id,
                "scores": {
                    "bm25": c.bm25_score,
                    "dense": c.dense_score,
                    "hybrid": c.hybrid_score,
                    "rerank": c.rerank_score,
                    "final": c.final_score
                },
                "metadata": c.metadata
            })

        context_text = "\n\n---\n\n".join(c.text[:600] for c in chunks)

        return {
            "question": question,
            "contexts": contexts,
            "context_text": context_text,
            "chunks_count": len(chunks),
            "articles_selected": stats["articles"]["selected"],
            "timings": stats["timings"],
            "sources": list(set(c.article_id for c in chunks))
        }

    async def generate_with_context(self, question: str, context_data: Dict[str, Any],
                                    system_prompt: Optional[str] = None) -> str:
        """
        Génère une réponse à partir de contextes pré-récupérés.
        Utile pour l'évaluation RAGAS où on veut tester différents prompts.
        """
        context_text = context_data.get("context_text", "")
        sources = context_data.get("sources", [])

        sources_str = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources[:5])])

        prompt = f"""Tu es Lexibot, expert juridique tunisien.

CONTEXTE JURIDIQUE:
{context_text[:4000]}

SOURCES:
{sources_str}

QUESTION: {question}

INSTRUCTIONS:
1. Base ta réponse UNIQUEMENT sur le contexte ci-dessus
2. Cite les articles précisément
3. Si l'info n'est pas dans le contexte, dis "Je ne dispose pas de cette information"

RÉPONSE:"""

        try:
            return await self.retriever.llm(
                prompt,
                system_prompt=system_prompt or "Tu es un juriste tunisien senior, précis et prudent.",
                temperature=0.1
            )
        except Exception as e:
            log.error(f"Generation failed: {e}")
            return f"[Erreur génération: {e}]"

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "chroma_ok": self.chroma.collection is not None,
            "bm25_docs": len(self.bm25.tokenized_corpus),
            "articles_indexed": len(self.article_index.articles),
            "embedder_stats": self.embedder.get_stats(),
            "reranker_loaded": self.reranker.model is not None,
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1e6 if settings.EMBED_DEVICE == "cuda" else 0
        }


# ============================================================================
# FACADE: LEXIOS RAG
# ============================================================================

class LexiosRAG:
    def __init__(self):
        self.retriever = HybridRetriever()
        log.info("🧠 LexiosRAG v13 initialisé")

    async def ingest_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await self.retriever.index_document(doc)
            return {"status": "success", "chunks_indexed": len(doc.get("chunks", []))}
        except Exception as e:
            log.error(f"Ingestion error: {e}")
            return {"status": "error", "message": str(e)}

    async def query(self, question: str, top_k: int = None,
                    generate_answer: bool = True,
                    metadata_filter: Optional[Dict] = None) -> Dict[str, Any]:
        chunks, stats = await self.retriever.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter
        )

        context_parts = []
        for c in chunks:
            context_parts.append(f"[{c.article_id}] {c.text[:600]}...")
        context = "\n\n---\n\n".join(context_parts)

        answer = ""
        if generate_answer and chunks:
            answer = await self._generate_response(question, context, chunks)

        return {
            "answer": answer,
            "chunks": [c.to_dict() for c in chunks],
            "context": context,
            "stats": {
                "total_time_ms": round(stats["timings"]["total_ms"], 1),
                "chunks_found": len(chunks),
                "articles_selected": stats["articles"]["selected"],
                "sources": list(set(c.article_id for c in chunks))
            },
            "query": question
        }

    async def _generate_response(self, question: str, context: str,
                                 chunks: List[RetrievedChunk]) -> str:
        sources_str = "\n".join([
            f"[{i+1}] {c.article_id} (score: {c.final_score:.2f})"
            for i, c in enumerate(chunks[:5])
        ])
        prompt = f"""Tu es Lexibot, expert juridique tunisien. Réponds en français ou arabe selon la question.

CONTEXTE JURIDIQUE RÉCUPÉRÉ:
{context[:4000]}

SOURCES:
{sources_str}

QUESTION: {question}

INSTRUCTIONS:
1. Base ta réponse UNIQUEMENT sur le contexte ci-dessus
2. Cite les articles précisément (ex: "Art. 215 Code pénal")
3. Si l'info n'est pas dans le contexte, dis "Je ne dispose pas de cette information"
4. Structure: Résumé juridique, analyse détaillée, conclusion

RÉPONSE:"""
        try:
            return await self.retriever.llm(
                prompt,
                system_prompt="Tu es un juriste tunisien senior, précis et prudent.",
                temperature=0.1
            )
        except Exception as e:
            log.error(f"Génération réponse échouée: {e}")
            return f"[Erreur génération: {e}]\n\nContexte:\n{context[:1000]}..."

    def get_health_status(self) -> Dict[str, Any]:
        return self.retriever.get_health_status()
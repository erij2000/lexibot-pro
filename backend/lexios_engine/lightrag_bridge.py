"""
lightrag_bridge.py — Lexios LightRAG Bridge v2 (CORRIGÉ)
=========================================================
Intégration LightRAG comme couche de contexte (PAS de réponse).

Architecture corrigée:
  Layer 1 → Top chunks (BM25 + vector + rerank) = contexte précis
  Layer 2 → Graph insights (entités + relations) = contexte relationnel
            ↓
       CONTEXT FUSION (chunks + graph_facts)
            ↓
       LLM UNIQUE (Groq, 1 seule génération)

LightRAG est OPTIONNEL et déclenché par trigger intelligent.
"""

from __future__ import annotations

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import settings

log = logging.getLogger("lexios.lightrag")

# ── OPTIONAL IMPORTS ─────────────────────────────────────────────────────────

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    HAS_LIGHTRAG = True
except ImportError:
    HAS_LIGHTRAG = False
    log.warning("⚠️ lightrag-hku non installé. LightRAG désactivé.")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GraphContext:
    """Contexte extrait du graphe LightRAG (PAS une réponse LLM)."""
    facts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    relevant_chunks: List[str] = field(default_factory=list)
    raw_context: str = ""
    mode: str = "disabled"
    timing_ms: float = 0.0

    def to_context_string(self, max_length: int = 2000) -> str:
        """Convertit en string injectable dans le prompt LLM."""
        parts = []
        if self.facts:
            parts.append("FAITS RELATIONNELS DU GRAPHE:")
            for f in self.facts[:10]:
                parts.append(f"  - {f}")
        if self.entities:
            parts.append(f"\nENTITÉS IDENTIFIÉES: {', '.join(self.entities[:15])}")
        if self.relations:
            parts.append(f"\nRELATIONS: {', '.join(self.relations[:10])}")
        if self.relevant_chunks:
            parts.append("\nEXTRAITS DE DOCUMENTS (via graphe):")
            for c in self.relevant_chunks[:5]:
                parts.append(f"  • {c[:300]}")

        result = "\n".join(parts)
        return result[:max_length] if len(result) > max_length else result


@dataclass
class TriggerAnalysis:
    """Analyse de la question pour décider si LightRAG est pertinent."""
    use_lightrag: bool = False
    reason: str = ""
    confidence: float = 0.0
    suggested_mode: str = "hybrid"


# ============================================================================
# TRIGGER INTELLIGENT (LightRAG optionnel)
# ============================================================================

class LightRAGTrigger:
    """
    Détermine si LightRAG doit être activé pour une question donnée.

    Règles:
    - Questions relationnelles → LightRAG activé
    - Questions factuelles simples → LightRAG désactivé (économie ressources)
    - Questions multi-documents → LightRAG activé
    """

    RELATIONAL_KEYWORDS = {
        "fr": [
            "relation", "lien", "lié", "interagit", "interagissent",
            "connecté", "rapport", "dépend", "influence", "impact",
            "compar", "différence", "similitude", "oppose", "versus",
            "tous les", "liste des", "ensemble", "cumul", "cumulatif",
            "en lien avec", "par rapport à", "concernant", "portant sur",
            "articles liés", "dispositions connexes", "textes relatifs",
        ],
        "ar": [
            "علاقة", "مرتبط", "متصل", "يتفاعل", "يؤثر", "تأثير",
            "فرق", "اختلاف", "تشابه", "مقارنة", "جميع", "كل",
            "قائمة", "مجموع", "تراكمي", "فيما يتعلق", "بخصوص",
        ]
    }

    MULTI_HOP_PATTERNS = [
        r"article\s+\d+.+article\s+\d+",  # Compare 2 articles
        r"(code|loi).+(code|loi)",  # Inter-code
        r"(bail|contrat|vente).+(obligations|pénal|civil)",  # Cross-domain
    ]

    @classmethod
    def analyze(cls, question: str) -> TriggerAnalysis:
        """Analyse la question et décide d'activer LightRAG."""
        if not settings.LIGHTRAG_ENABLED:
            return TriggerAnalysis(use_lightrag=False, reason="LightRAG désactivé dans config")

        q_lower = question.lower()
        ar_count = sum(1 for c in question if "\u0600" <= c <= "\u06FF")
        lang = "ar" if ar_count / max(len(question), 1) > 0.3 else "fr"

        score = 0.0
        reasons = []

        # 1. Mots-clés relationnels
        keywords = cls.RELATIONAL_KEYWORDS.get(lang, cls.RELATIONAL_KEYWORDS["fr"])
        matched = [k for k in keywords if k in q_lower]
        if matched:
            score += len(matched) * 0.15
            reasons.append(f"mots-clés relationnels: {matched[:3]}")

        # 2. Patterns multi-hop (regex)
        import re
        for pattern in cls.MULTI_HOP_PATTERNS:
            if re.search(pattern, q_lower):
                score += 0.4
                reasons.append("pattern multi-hop détecté")
                break

        # 3. Longueur question (questions complexes)
        if len(question.split()) > 15:
            score += 0.1
            reasons.append("question complexe (longue)")

        # 4. Questions factuelles simples → désactiver
        factual_patterns = [
            r"^quel\s+est\s+l\'article",
            r"^article\s+\d+",
            r"^définition",
            r"^qu\'est-ce",
            r"^c\'est\s+quoi",
        ]
        for pattern in factual_patterns:
            if re.match(pattern, q_lower):
                score -= 0.3
                reasons.append("question factuelle simple → Lexios suffit")
                break

        # Seuil de décision
        threshold = 0.35
        use_lightrag = score >= threshold

        # Mode suggéré
        if score > 0.7:
            suggested_mode = "hybrid"
        elif score > 0.5:
            suggested_mode = "local"
        else:
            suggested_mode = "global" if use_lightrag else "disabled"

        return TriggerAnalysis(
            use_lightrag=use_lightrag,
            reason="; ".join(reasons) if reasons else "question standard",
            confidence=min(score, 1.0),
            suggested_mode=suggested_mode
        )


# ============================================================================
# EMBEDDING BRIDGE (Reuse BGE-M3)
# ============================================================================

class LexiosEmbeddingBridge:
    """Réutilise votre RTX2050SafeEmbedder pour LightRAG."""

    def __init__(self, embedder: Any = None):
        self.embedder = embedder
        self.dimension = settings.EMBED_DIM

    async def encode(self, texts: List[str]) -> np.ndarray:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(settings.EMBED_MODEL, device=settings.EMBED_DEVICE)
            return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        if hasattr(self.embedder, 'encode_async'):
            embeddings = await self.embedder.encode_async(texts)
            return np.array(embeddings)
        else:
            return self.embedder.encode_safe(texts)

    def get_embedding_func(self):
        if not HAS_LIGHTRAG:
            raise ImportError("LightRAG non disponible")

        async def _embed(texts: List[str]) -> np.ndarray:
            return await self.encode(texts)

        from lightrag.utils import EmbeddingFunc
        return EmbeddingFunc(
            embedding_dim=self.dimension,
            max_token_size=8192,
            func=_embed
        )


# ============================================================================
# LLM BRIDGE (Reuse GroqLLMWrapper)
# ============================================================================

class LexiosLLMBridge:
    """Réutilise votre GroqLLMWrapper pour LightRAG."""

    def __init__(self, llm_wrapper: Any = None):
        self.llm = llm_wrapper

    async def complete(self, prompt: str, system_prompt: Optional[str] = None,
                       history_messages: List[Dict] = None, **kwargs) -> str:
        if self.llm is None:
            from rag_service import GroqLLMWrapper
            self.llm = GroqLLMWrapper()

        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                full_prompt += f"{role}: {content}\n"
        full_prompt += f"user: {prompt}"

        return await self.llm(full_prompt, temperature=0.1, max_tokens=2048)

    def get_llm_func(self) -> Callable:
        async def _llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await self.complete(prompt, system_prompt, history_messages, **kwargs)
        return _llm_func


# ============================================================================
# LIGHTRAG BRIDGE (CORRIGÉ — Context-only, pas réponse)
# ============================================================================

class LightRAGBridge:
    """
    Bridge LightRAG ↔ Lexios — VERSION CORRIGÉE.

    Cette classe NE GÉNÈRE PAS de réponse LLM.
    Elle extrait uniquement du CONTEXTE du graphe (facts, entités, relations)
    qui sera injecté dans le prompt unique de Groq.
    """

    def __init__(self, lexios_rag: Any = None, working_dir: Optional[str] = None):
        self.lexios_rag = lexios_rag
        self.working_dir = working_dir or settings.LIGHTRAG_WORKING_DIR
        self._rag_instance: Optional[LightRAG] = None
        self._initialized = False
        self._embed_bridge: Optional[LexiosEmbeddingBridge] = None
        self._llm_bridge: Optional[LexiosLLMBridge] = None
        self.trigger = LightRAGTrigger()

        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        if not HAS_LIGHTRAG:
            log.warning("LightRAG non disponible - bridge en mode pass-through")
            return

        self._init_bridges()

    def _init_bridges(self):
        embedder = None
        if self.lexios_rag and hasattr(self.lexios_rag, 'retriever'):
            retriever = self.lexios_rag.retriever
            if hasattr(retriever, 'embedder'):
                embedder = retriever.embedder
        self._embed_bridge = LexiosEmbeddingBridge(embedder)

        llm = None
        if self.lexios_rag and hasattr(self.lexios_rag, 'retriever'):
            retriever = self.lexios_rag.retriever
            if hasattr(retriever, 'llm'):
                llm = retriever.llm
        self._llm_bridge = LexiosLLMBridge(llm)
        log.info("🌉 Bridges LightRAG initialisés")

    async def initialize(self):
        if not HAS_LIGHTRAG or self._initialized:
            return
        try:
            self._rag_instance = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self._llm_bridge.get_llm_func(),
                embedding_func=self._embed_bridge.get_embedding_func(),
            )
            await self._rag_instance.initialize_storages()
            self._initialized = True
            log.info(f"✅ LightRAG initialisé: {self.working_dir}")
        except Exception as e:
            log.error(f"❌ LightRAG init failed: {e}")
            self._rag_instance = None

    async def index_document(self, doc: Dict[str, Any]) -> bool:
        """Indexe un document dans LightRAG (graphe)."""
        if not self._initialized or not self._rag_instance:
            return False
        try:
            raw_text = doc.get("raw_text", "")
            if not raw_text or len(raw_text) < 50:
                return False
            await self._rag_instance.ainsert(raw_text)
            log.info(f"📊 LightRAG: document indexé ({len(raw_text)} chars)")
            return True
        except Exception as e:
            log.error(f"LightRAG indexation failed: {e}")
            return False

    async def get_graph_context(self, question: str, 
                                mode: str = "hybrid",
                                top_k: int = 8) -> GraphContext:
        """
        Extrait du CONTEXTE du graphe (PAS de réponse LLM).

        Retourne: facts, entités, relations, chunks pertinents
        qui seront injectés dans le prompt unique.
        """
        if not self._initialized or not self._rag_instance:
            return GraphContext(mode="disabled")

        import time
        t0 = time.perf_counter()

        try:
            param = QueryParam(
                mode=mode,
                top_k=top_k,
                response_type="multiple paragraphs",
                only_need_context=True  # ← IMPORTANT: on veut le contexte, pas la réponse
            )

            # LightRAG retourne le contexte brut (pas de génération LLM)
            result = await self._rag_instance.aquery(question, param=param)

            # Parser le résultat pour extraire facts/entités/relations
            facts = []
            entities = []
            relations = []
            chunks = []

            if isinstance(result, str):
                # Parser le texte pour extraire les faits structurés
                lines = [l.strip() for l in result.split("\n") if l.strip()]
                for line in lines:
                    if line.startswith("-") or line.startswith("•"):
                        facts.append(line[1:].strip())
                    elif ":" in line and len(line) < 200:
                        entities.append(line.split(":")[0].strip())
                chunks = [result]  # Le contexte brut comme chunk

            timing = (time.perf_counter() - t0) * 1000

            return GraphContext(
                facts=facts,
                entities=entities,
                relations=relations,
                relevant_chunks=chunks,
                raw_context=result if isinstance(result, str) else str(result),
                mode=mode,
                timing_ms=timing
            )

        except Exception as e:
            log.error(f"LightRAG context extraction failed: {e}")
            return GraphContext(mode=f"{mode}_error")

    async def analyze_trigger(self, question: str) -> TriggerAnalysis:
        """Analyse si LightRAG est pertinent pour cette question."""
        return self.trigger.analyze(question)

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Statistiques du graphe."""
        if not self._initialized or not self._rag_instance:
            return {"status": "not_initialized"}
        try:
            import json
            graph_path = Path(self.working_dir) / "graph_data.json"
            if graph_path.exists():
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                return {
                    "status": "active",
                    "nodes": len(graph_data.get("nodes", [])),
                    "edges": len(graph_data.get("edges", [])),
                    "working_dir": self.working_dir
                }
            return {"status": "active", "nodes": 0, "edges": 0}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def close(self):
        if self._rag_instance and hasattr(self._rag_instance, 'finalize_storages'):
            await self._rag_instance.finalize_storages()


# ============================================================================
# SINGLETON
# ============================================================================

_lightrag_bridge_instance: Optional[LightRAGBridge] = None

async def get_lightrag_bridge(lexios_rag: Any = None) -> LightRAGBridge:
    global _lightrag_bridge_instance
    if _lightrag_bridge_instance is None:
        _lightrag_bridge_instance = LightRAGBridge(lexios_rag)
        await _lightrag_bridge_instance.initialize()
    return _lightrag_bridge_instance
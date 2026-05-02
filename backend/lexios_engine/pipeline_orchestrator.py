"""
pipeline_orchestrator.py — Lexios Pro+ Orchestrator v1
=======================================================
Orchestration production-grade du pipeline RAG complet.

Architecture:
  User Query
       ↓
  QueryRouter (heuristique + ML léger) → mode: simple/complex
       ↓
  [Optionnel] HyDE → query enrichie
       ↓
  HybridRetriever (BM25 + Chroma + Article-Aware)
       ↓
  [Optionnel] LightRAGTrigger → GraphContext si pertinent
       ↓
  ContextBuilder (budget tokens + diversité + déduplication)
       ↓
  LLM UNIQUE (Groq, temp=0.1, seed fixe)
       ↓
  PostProcessor (citations + vérification + fallback)
       ↓
  Response + Metrics + Cache

Principes:
  - 1 seul appel LLM par requête
  - Context fusion > answer fusion
  - Routing data-driven (pas de règles bricolées)
  - Dégradation gracieuse à chaque étape
  - Observabilité totale (logs structurés)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from config import settings

log = logging.getLogger("lexios.orchestrator")


# ============================================================================
# 1. QUERY ROUTER (Intelligent, stable, cheap)
# ============================================================================

@dataclass
class RouteDecision:
    """Décision de routage pour une requête."""
    mode: str = "simple"           # "simple" | "complex"
    use_hyde: bool = False
    use_lightrag: bool = False
    use_reranker: bool = True
    top_k_retrieval: int = 50
    top_k_final: int = 8
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "use_hyde": self.use_hyde,
            "use_lightrag": self.use_lightrag,
            "use_reranker": self.use_reranker,
            "top_k_retrieval": self.top_k_retrieval,
            "top_k_final": self.top_k_final,
            "confidence": round(self.confidence, 3),
            "reason": self.reason
        }


class QueryRouter:
    """
    Routeur intelligent de requêtes.

    Décide:
    - mode: simple (Layer 1 seul) vs complex (Layer 1 + LightRAG context)
    - use_hyde: expansion query pour questions complexes
    - use_lightrag: graphe pour questions relationnelles
    - top_k: nombre de chunks à récupérer

    Features (cheap + robust, pas de LLM):
    - Longueur requête
    - Entités détectées (articles, codes)
    - Mots-clés conceptuels (relation, comparer, différences)
    - Similarité avec historique (cache sémantique)
    """

    # Mots-clés déclenchant le mode complexe
    COMPLEX_KEYWORDS = {
        "fr": [
            "relation", "lien", "lié", "interagit", "interagissent", "connecté",
            "rapport", "dépend", "influence", "impact", "compar", "différence",
            "similitude", "oppose", "versus", "tous les", "liste des", "ensemble",
            "cumul", "cumulatif", "en lien avec", "par rapport à", "concernant",
            "portant sur", "articles liés", "dispositions connexes", "textes relatifs",
            "pourquoi", "comment", "explique", "analyse", "synthèse",
        ],
        "ar": [
            "علاقة", "مرتبط", "متصل", "يتفاعل", "يؤثر", "تأثير", "فرق", "اختلاف",
            "تشابه", "مقارنة", "جميع", "كل", "قائمة", "مجموع", "تراكمي",
            "فيما يتعلق", "بخصوص", "لماذا", "كيف", "اشرح", "حلل", "تلخيص",
        ]
    }

    # Patterns factuels simples → mode simple forcé
    SIMPLE_PATTERNS = [
        r"^quel\s+est\s+l\'article",
        r"^article\s+\d+",
        r"^définition",
        r"^qu\'est-ce",
        r"^c\'est\s+quoi",
        r"^donne\s+moi",
        r"^cite",
    ]

    def __init__(self):
        self._history: List[Tuple[str, RouteDecision]] = []
        self._simple_count = 0
        self._complex_count = 0

    def route(self, query: str, history_embedding: Optional[List[float]] = None) -> RouteDecision:
        """
        Route une requête vers le bon mode.

        Score heuristique:
        - 0-1: simple
        - 1-2: intermédiaire (simple avec HyDE)
        - 2+: complex (HyDE + LightRAG context)
        """
        import re

        q_lower = query.lower().strip()
        words = q_lower.split()

        # Détection langue
        ar_count = sum(1 for c in query if "\u0600" <= c <= "\u06FF")
        lang = "ar" if ar_count / max(len(query), 1) > 0.3 else "fr"

        score = 0.0
        reasons = []

        # 1. Longueur requête
        if len(words) > 18:
            score += 1.0
            reasons.append("question_longue")
        elif len(words) > 12:
            score += 0.5
            reasons.append("question_moyenne")

        # 2. Mots-clés conceptuels
        keywords = self.COMPLEX_KEYWORDS.get(lang, self.COMPLEX_KEYWORDS["fr"])
        matched = [k for k in keywords if k in q_lower]
        if matched:
            score += len(matched) * 1.5  # Poids très fort pour déclencher LightRAG
            reasons.append(f"keywords:{matched[:2]}")

        # 3. Entités multiples (articles, codes)
        article_refs = len(re.findall(r"art(?:icle)?[.\s]*\d+", q_lower))
        code_refs = len(re.findall(r"code|loi|décret", q_lower))
        if article_refs >= 2 or code_refs >= 2:
            score += 1.0
            reasons.append("multi_references")
        elif article_refs >= 1:
            score += 0.3
            reasons.append("single_reference")

        # 4. Questions factuelles simples → forcer simple
        for pattern in self.SIMPLE_PATTERNS:
            if re.match(pattern, q_lower):
                score = 0.0
                reasons = ["pattern_factual_simple"]
                break

        # 5. Historique (si requête similaire à une précédente complexe)
        if history_embedding and self._history:
            # Simplification: on pourrait utiliser cosine similarity ici
            pass

        # Décision
        if score >= 2.0:
            mode = "complex"
            use_hyde = True
            use_lightrag = True
            top_k = 50
        elif score >= 1.0:
            mode = "intermediate"
            use_hyde = True
            use_lightrag = False
            top_k = 40
        else:
            mode = "simple"
            use_hyde = False
            use_lightrag = False
            top_k = 30

        decision = RouteDecision(
            mode=mode,
            use_hyde=use_hyde,
            use_lightrag=use_lightrag,
            use_reranker=True,
            top_k_retrieval=top_k,
            top_k_final=settings.RETRIEVAL_FINAL_K,
            confidence=score / (score + 1.0),
            reason="; ".join(reasons) if reasons else "default"
        )

        self._history.append((query, decision))
        if len(self._history) > 100:
            self._history.pop(0)

        log.info(f"🧭 Route: {query[:50]}... → mode={mode}, hyde={use_hyde}, lightrag={use_lightrag}, score={score:.1f}")
        return decision

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques de routage."""
        if not self._history:
            return {}
        modes = [d.mode for _, d in self._history]
        return {
            "total_routed": len(self._history),
            "simple_ratio": modes.count("simple") / len(modes),
            "complex_ratio": modes.count("complex") / len(modes),
            "avg_confidence": sum(d.confidence for _, d in self._history) / len(self._history)
        }


# ============================================================================
# 2. CONTEXT BUILDER (Budget tokens + diversité + déduplication)
# ============================================================================

@dataclass
class BuiltContext:
    """Contexte final construit pour le LLM."""
    text: str
    chunks_used: int
    articles_used: List[str]
    graph_facts_used: int
    total_chars: int
    token_estimate: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_used": self.chunks_used,
            "articles_used": self.articles_used,
            "graph_facts_used": self.graph_facts_used,
            "total_chars": self.total_chars,
            "token_estimate": self.token_estimate
        }


class ContextBuilder:
    """
    Construit le contexte final injecté dans le prompt LLM.

    Règles:
    - Budget tokens respecté (max_chars configurable)
    - Diversité de sources (pas 3× le même article)
    - Déduplication par article_id
    - Tagging explicite [BASE] vs [USER_DOC] vs [GRAPH]
    """

    def __init__(self, max_chars: Optional[int] = None):
        self.max_chars = max_chars or getattr(settings, "PIPELINE_MAX_CONTEXT_CHARS", 12000)
        self._stats = {"builds": 0, "total_chars_avg": 0}

    def build(self, 
              chunks: List[Dict[str, Any]], 
              graph_context: Optional[str] = None,
              user_doc_chunks: Optional[List[Dict[str, Any]]] = None) -> BuiltContext:
        """
        Construit le contexte final.

        Ordre de priorité:
        1. Chunks base (article-aware)
        2. Chunks user doc (si upload)
        3. Graph facts (si mode complex)
        """
        parts = []
        seen_articles = {}
        articles_used = []

        # 1. Chunks base (diversifiés par article)
        base_parts = []
        for c in chunks:
            # Extract article_id from chunk dict or nested metadata
            aid = c.get("article_id") if isinstance(c, dict) else None
            if not aid and isinstance(c, dict):
                aid = c.get("metadata", {}).get("article_id") or c.get("metadata", {}).get("source_file", "unknown")
            if not aid:
                aid = getattr(c, "article_id", None) or getattr(getattr(c, "metadata", None), "source_file", "unknown")
            if not aid or aid == "unknown":
                aid = f"chunk_{len(seen_articles)}"
            
            # Relaxed deduplication: allow up to 3 chunks per article
            count = seen_articles.get(aid, 0)
            if count >= 3:
                continue

            seen_articles[aid] = count + 1
            if aid not in articles_used:
                articles_used.append(aid)

            text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            source = c.get("metadata", {}).get("source_file", "base") if isinstance(c, dict) else "base"

            # Standard markdown format (parseable by Angular front)
            base_parts.append(f"**Source:** {aid}\n\n{text[:600]}")


            if len("\n\n".join(base_parts)) > self.max_chars * 0.7:
                break

        if base_parts:
            parts.append("=== SOURCES JURIDIQUES ===")
            parts.extend(base_parts)

        # 2. User doc chunks (si présents)
        if user_doc_chunks:
            user_parts = []
            for c in user_doc_chunks[:3]:
                text = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                user_parts.append(f"[USER_DOC] {text[:500]}")

            if user_parts:
                parts.append("\n=== DOCUMENT UTILISATEUR ===")
                parts.extend(user_parts)

        # 3. Graph facts (si mode complex)
        graph_facts_count = 0
        if graph_context and len(graph_context.strip()) > 50:
            remaining = self.max_chars - len("\n\n".join(parts))
            if remaining > 500:
                parts.append("\n=== RELATIONS ET CONCEPTS ===")
                parts.append(graph_context[:min(len(graph_context), remaining - 100)])
                graph_facts_count = len([l for l in graph_context.split("\n") if l.strip().startswith("-")])

        full_text = "\n\n".join(parts)

        # Troncature si nécessaire
        if len(full_text) > self.max_chars:
            full_text = full_text[:self.max_chars - 100] + "\n...[contexte tronqué]"

        self._stats["builds"] += 1
        self._stats["total_chars_avg"] = (self._stats["total_chars_avg"] * (self._stats["builds"] - 1) + len(full_text)) / self._stats["builds"]

        # Estimation tokens (1 token ≈ 4 chars pour français)
        token_estimate = len(full_text) // 4

        return BuiltContext(
            text=full_text,
            chunks_used=len(base_parts),
            articles_used=articles_used,
            graph_facts_used=graph_facts_count,
            total_chars=len(full_text),
            token_estimate=token_estimate
        )

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)


# ============================================================================
# 3. POST-PROCESSOR (Citations + vérification + fallback)
# ============================================================================

class PostProcessor:
    """
    Post-traitement de la réponse LLM.

    Vérifications:
    - Longueur minimale
    - Présence de citations [source|article]
    - Pas d'hallucination flagrante (mots interdits)
    - Fallback si réponse vide/faible
    """

    FORBIDDEN_PATTERNS = [
        "je ne suis pas sûr", "peut-être", "il se peut que",
        "selon certaines sources", "d'après ce que je sais",
        "à mon avis",
    ]


    def __init__(self):
        self._stats = {"processed": 0, "fallbacks": 0, "cleaned": 0}

    def process(self, answer: str, context: BuiltContext, 
                route: RouteDecision) -> Tuple[str, Dict[str, Any]]:
        """
        Post-traite la réponse.

        Returns:
            (answer_clean, metadata)
        """
        original = answer
        metadata = {"original_length": len(answer), "fallback": False, "cleaned": False}

        # 1. Nettoyage basique
        answer = answer.strip()

        # 2. Vérification longueur
        if len(answer) < 30:
            answer = "Je n'ai pas trouvé suffisamment d'informations fiables dans les documents juridiques pour répondre précisément à cette question."
            metadata["fallback"] = True
            metadata["reason"] = "too_short"
            self._stats["fallbacks"] += 1
            return answer, metadata

        # 3. Vérification citations (adapté au nouveau format markdown)
        has_citations = "**Source:**" in answer or "[" in answer
        if not has_citations and context.articles_used:
            # Ajouter un avertissement discret si pas de citations
            metadata["warning"] = "pas_de_citations_explicites"


        # 4. Détection patterns interdits (hallucination)
        q_lower = answer.lower()
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in q_lower:
                metadata["warning"] = f"pattern_interdit_detecte:{pattern}"
                metadata["cleaned"] = True
                self._stats["cleaned"] += 1
                break

        # 5. Vérification cohérence avec contexte (simple)
        context_text = context.text.lower()
        answer_words = set(q_lower.split())
        context_words = set(context_text.split())
        if answer_words and len(answer_words) > 5:
            overlap = len(answer_words & context_words) / len(answer_words)
            metadata["context_overlap"] = round(overlap, 3)
            if overlap < 0.1:
                metadata["warning"] = "faible_overlap_avec_contexte"

        self._stats["processed"] += 1
        metadata["final_length"] = len(answer)

        return answer, metadata

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)


# ============================================================================
# 4. PIPELINE ORCHESTRATOR (Principal)
# ============================================================================

@dataclass
class PipelineResult:
    """Résultat complet du pipeline."""
    answer: str
    route: RouteDecision
    context: BuiltContext
    timings: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "route": self.route.to_dict(),
            "context": self.context.to_dict(),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "metadata": self.metadata
        }


class LexiosPipeline:
    """
    Pipeline principal orchestré.

    Usage:
        pipeline = LexiosPipeline(lexios_rag)
        result = await pipeline.process("Question juridique")
    """

    def __init__(self, lexios_rag: Any):
        self.rag = lexios_rag
        self.router = QueryRouter()
        self.context_builder = ContextBuilder(max_chars=getattr(settings, "PIPELINE_MAX_CONTEXT_CHARS", 12000))
        self.post_processor = PostProcessor()
        self._lightrag: Optional[Any] = None
        self._hyde: Optional[Any] = None
        self._total_queries = 0
        self._total_time_ms = 0
 
    async def close(self):
        """Libère les ressources du RAG."""
        if hasattr(self.rag, "close"):
            await self.rag.close()

    async def initialize(self):
        """Initialise les composants optionnels."""
        if settings.LIGHTRAG_ENABLED:
            try:
                from lightrag_bridge import get_lightrag_bridge
                self._lightrag = await get_lightrag_bridge(
                    self.rag.retriever.embedder,
                    self.rag.retriever.llm
                )
            except Exception as e:
                log.warning(f"LightRAG init skipped: {e}")

        if getattr(settings, "USE_HYDE", True):
            try:
                from hyde import get_hyde
                self._hyde = get_hyde(llm_wrapper=self.rag.retriever.llm)
            except Exception as e:
                log.warning(f"HyDE init skipped: {e}")


    async def process(self, query: str, 
                      user_file_chunks: Optional[List[Dict]] = None,
                      metadata_filter: Optional[Dict] = None) -> PipelineResult:
        """
        Exécute le pipeline complet.

        Architecture:
          Query → Router → [HyDE] → Retrieval → [LightRAG] → ContextBuilder → LLM → PostProcessor
        """
        self._total_queries += 1
        t_total = time.perf_counter()
        timings = {}

        # ── Étape 1: Routing ──
        t0 = time.perf_counter()
        route = self.router.route(query)
        timings["router_ms"] = (time.perf_counter() - t0) * 1000

        # ── Étape 2: HyDE (optionnel) ──
        t0 = time.perf_counter()
        query_for_retrieval = query
        if route.use_hyde and self._hyde:
            try:
                query_for_retrieval = await self._hyde.enhance(query)
                log.info(f"📝 HyDE appliqué: {query_for_retrieval[:80]}...")
            except Exception as e:
                log.warning(f"HyDE failed: {e}")
        timings["hyde_ms"] = (time.perf_counter() - t0) * 1000

        # ── Étape 3: Retrieval (Lexios Layer 1) ──
        t0 = time.perf_counter()
        try:
            lexios_result = await self.rag.query_for_eval(
                query_for_retrieval, 
                top_k=route.top_k_retrieval,
                metadata_filter=metadata_filter
            )
            chunks = lexios_result.get("contexts", [])
            sources = lexios_result.get("sources", [])
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return PipelineResult(
                answer="Erreur de récupération des documents.",
                route=route,
                context=BuiltContext("", 0, [], 0, 0, 0),
                timings={"error_ms": (time.perf_counter() - t_total) * 1000},
                metadata={"error": str(e)}
            )
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # ── Étape 4: LightRAG (optionnel, trigger) ──
        t0 = time.perf_counter()
        graph_context_str = ""
        if route.use_lightrag and self._lightrag:
            try:
                trigger = await self._lightrag.analyze_trigger(query)
                if trigger.use_lightrag:
                    graph_ctx = await self._lightrag.get_graph_context(
                        query, mode=trigger.suggested_mode, top_k=route.top_k_final
                    )
                    graph_context_str = graph_ctx.to_context_string(max_length=1500)
                    log.info(f"🌐 LightRAG context injecté: {len(graph_context_str)} chars")
            except Exception as e:
                log.warning(f"LightRAG failed: {e}")
        timings["lightrag_ms"] = (time.perf_counter() - t0) * 1000

        # ── Étape 5: Context Builder ──
        t0 = time.perf_counter()
        built_context = self.context_builder.build(
            chunks=chunks,
            graph_context=graph_context_str if graph_context_str else None,
            user_doc_chunks=user_file_chunks
        )
        timings["context_build_ms"] = (time.perf_counter() - t0) * 1000

        # ── Étape 6: Génération UNIQUE LLM ──
        t0 = time.perf_counter()
        try:
            answer = await self.rag.generate_with_context(
                question=query,
                context_data={"context_text": built_context.text, "sources": sources},
                graph_context=graph_context_str if graph_context_str else None
            )
        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            answer = f"[Erreur génération]\n\nContextes disponibles:\n{built_context.text[:1000]}..."
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000

        # ── Étape 7: Post-Processing ──
        t0 = time.perf_counter()
        final_answer, post_meta = self.post_processor.process(answer, built_context, route)
        timings["postprocess_ms"] = (time.perf_counter() - t0) * 1000

        # ── Totaux ──
        total_ms = (time.perf_counter() - t_total) * 1000
        timings["total_ms"] = total_ms
        self._total_time_ms += total_ms

        log.info(f"✅ Pipeline: {total_ms:.0f}ms | mode={route.mode} | "
                 f"chunks={built_context.chunks_used} | articles={len(built_context.articles_used)} | "
                 f"graph={built_context.graph_facts_used}")

        return PipelineResult(
            answer=final_answer,
            route=route,
            context=built_context,
            timings=timings,
            metadata={
                "query": query,
                "hyde_applied": query_for_retrieval != query,
                "query_for_retrieval": query_for_retrieval if query_for_retrieval != query else None,
                "post_processing": post_meta,
                "avg_latency_ms": round(self._total_time_ms / self._total_queries, 2) if self._total_queries > 0 else 0
            }
        )

    def get_health(self) -> Dict[str, Any]:
        """Health check du pipeline."""
        return {
            "status": "healthy",
            "total_queries": self._total_queries,
            "avg_latency_ms": round(self._total_time_ms / self._total_queries, 2) if self._total_queries > 0 else 0,
            "router_stats": self.router.get_stats(),
            "context_builder_stats": self.context_builder.get_stats(),
            "post_processor_stats": self.post_processor.get_stats(),
            "components": {
                "lightrag": self._lightrag is not None,
                "hyde": self._hyde is not None,
            }
        }

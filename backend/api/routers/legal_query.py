"""
legal_query.py — Lexios Engine | Router Intelligent v5 (Clean)
===============================================================
Pipeline: Cache → NER → HyDE → RAG (article-aware + reranker) → Synthèse LLM
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional, Dict, Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from lexios_engine.config import settings
from lexios_engine.cache import get_cache
from lexios_engine.hyde import HyDE
from lexios_engine.rag_service import LexiosRAG
from lexios_engine.pipeline_orchestrator import LexiosPipeline
from lexios_engine.utils.prompt_builder import PromptBuilder


try:
    from lexios_engine.lightrag_bridge import LightRAGBridge, get_lightrag_bridge, GraphContext
    HAS_LIGHTRAG_BRIDGE = True
except ImportError:
    HAS_LIGHTRAG_BRIDGE = False

try:
    from lexios_engine.ragas_evaluator import RagasEvaluator, get_ragas_evaluator
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

log = logging.getLogger("lexios.router")

# ── SINGLETONS ──────────────────────────────────────────────────────────────

_rag: Optional[LexiosRAG] = None
_hyde: Optional[HyDE] = None


def get_rag() -> LexiosRAG:
    global _rag
    if _rag is None:
        _rag = LexiosRAG()
    return _rag


def get_hyde() -> Optional[HyDE]:
    global _hyde
    if _hyde is None and settings.USE_HYDE:
        _hyde = HyDE()
    return _hyde



# ── SINGLETONS LIGHTRAG & RAGAS ─────────────────────────────────────────────

_lightrag: Optional[LightRAGBridge] = None
_ragas_eval: Optional[RagasEvaluator] = None

async def get_lightrag() -> Optional["LightRAGBridge"]:
    """Initialize LightRAG bridge if enabled."""
    global _lightrag
    if _lightrag is None and settings.LIGHTRAG_ENABLED and HAS_LIGHTRAG_BRIDGE:
        try:
            rag = get_rag()
            # Pass embedder + llm from retriever, not LexiosRAG itself
            _lightrag = await get_lightrag_bridge(
                rag.retriever.embedder,
                rag.retriever.llm
            )
        except Exception as e:
            log.warning(f"LightRAG init failed: {e}")
    return _lightrag

def get_ragas() -> "RagasEvaluator":
    global _ragas_eval
    if _ragas_eval is None and HAS_RAGAS:
        _ragas_eval = get_ragas_evaluator()
    return _ragas_eval

# ── SINGLETON PIPELINE ──────────────────────────────────────────────────────

_pipeline: Optional[LexiosPipeline] = None

async def get_pipeline() -> LexiosPipeline:
    """Get or initialize the LexiosPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = LexiosPipeline(get_rag())
        await _pipeline.initialize()
    return _pipeline


# ── PYDANTIC SCHEMAS ────────────────────────────────────────────────────────

class Entity(BaseModel):
    type: str = Field(..., description="Type d'entité juridique")
    value: str = Field(..., min_length=1, max_length=500)

    @field_validator("type")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        return v.lower().strip().replace(" ", "_")


class IntentSchema(BaseModel):
    intent: str = Field(default="discussion", pattern="^(legal_question|greeting|discussion|error)$")
    is_legal: bool = Field(default=False)
    language: str = Field(default="fr", pattern="^(fr|ar|en|mixed)$")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: List[Entity] = Field(default_factory=list)

    @field_validator("language")
    @classmethod
    def normalize_lang(cls, v: str) -> str:
        if not v:
            return "fr"
        v_lower = v.lower().strip()
        if v_lower.startswith("ar") or v_lower == "عربية":
            return "ar"
        if v_lower.startswith("en"):
            return "en"
        return "fr"


class LegalRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=3000)
    session_id: Optional[str] = Field(default=None, max_length=100)
    use_hyde: Optional[bool] = Field(default=None)
    use_cache: Optional[bool] = Field(default=None)
    metadata_filter: Optional[Dict[str, str]] = Field(default=None)
    legal_mode: str = Field(default="auto", pattern="^(auto|penal|civil|general)$")
    top_k: int = Field(default=8, ge=1, le=20)
    generate_answer: bool = Field(default=True)
    debug_mode: bool = Field(default=False, exclude=True)


class LegalResponse(BaseModel):
    answer: str = Field(..., description="Réponse finale")
    intent: str = Field(...)
    language: str = Field(...)
    is_legal: bool = Field(...)
    entities: List[Entity] = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rag_used: bool = Field(...)
    cache_hit: bool = Field(default=False)
    cache_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hyde_used: bool = Field(default=False)
    articles_found: int = Field(default=0, ge=0)
    sources_count: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(..., ge=0)
    session_id: Optional[str] = Field(default=None)
    debug_info: Optional[Dict[str, Any]] = Field(default=None)


# ── PROMPTS NER ─────────────────────────────────────────────────────────────

NER_SYSTEM_FR = """Tu es LexiBot, expert juridique senior en droit tunisien.
Analyse le message et extrait les métadonnées juridiques en JSON STRICT.

FORMAT:
{
  "intent": "legal_question|greeting|discussion",
  "is_legal": true|false,
  "language": "fr|ar|en",
  "confidence": 0.0-1.0,
  "entities": [
    {"type": "article", "value": "Art. 15 Code des obligations"}
  ]
}

RÈGLES:
- Réponds UNIQUEMENT en JSON valide
- Ne jamais inventer d'informations"""

NER_SYSTEM_AR = """أنت LexiBot، خبير قانوني تونسي كبير.
حلل الرسالة واستخرج JSON صارم فقط."""


# ── APPEL LLM UNIFIÉ ────────────────────────────────────────────────────────

async def _call_llm(prompt: str, system_prompt: str, use_json: bool = False,
                    temperature: float = 0.1, max_tokens: int = 2048,
                    rag: Optional[LexiosRAG] = None) -> str:
    """
    Appel LLM unifié via GroqLLMWrapper (circuit breaker, rate limiting, client reuse).
    Fallback Ollama si Groq non configuré.
    """
    # Try GroqLLMWrapper first (reuses client, respects rate limits, circuit breaker)
    if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        llm = None
        if rag and hasattr(rag, 'retriever') and hasattr(rag.retriever, 'llm'):
            llm = rag.retriever.llm
        if not llm:
            from lexios_engine.core_llm import GroqLLMWrapper
            llm = GroqLLMWrapper()
        
        try:
            return await llm(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_json=use_json,
            )
        except Exception as e:
            log.warning(f"GroqLLMWrapper failed, falling back to direct HTTP: {e}")
    
    # Fallback: Direct HTTP (Ollama or Groq via raw HTTP)
    timeout = settings.LLM_TIMEOUT
    try:
        if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
            payload = {
                "model": settings.GROQ_MODEL,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_json:
                payload["response_format"] = {"type": "json_object"}
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                    json=payload
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        else:
            payload = {
                "model": settings.OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": max_tokens}
            }
            if use_json:
                payload["format"] = "json"
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload)
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="LLM inaccessible")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM: {e}")



# ── NER PIPELINE ────────────────────────────────────────────────────────────

def _detect_language_heuristic(text: str) -> str:
    if not text:
        return "fr"
    ar_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if ar_count / len(text) > 0.3 else "fr"


def _sanitize_json(raw: str) -> str:
    if not raw:
        return "{}"
    text = raw.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    start_brace = text.find("{")
    end_brace = text.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        text = text[start_brace:end_brace+1]
    text = " ".join(text.replace('\n', ' ').replace('\r', '').split())
    return text if text else "{}"


def _fallback_intent_heuristic(message: str) -> IntentSchema:
    msg_lower = message.lower()
    greetings = ["bonjour", "salut", "hello", "hi", "مرحبا", "أهلا"]
    if any(g in msg_lower for g in greetings) and len(message) < 50:
        return IntentSchema(intent="greeting", is_legal=False, language="fr", confidence=0.8)
    legal_kws = ["bail", "contrat", "jugement", "tribunal", "avocat", "droit", "loi", "article",
                 "pénal", "civil", "divorce", "succession", "vente", "location", "كراء", "عقد",
                 "محكمة", "حكم", "قانون", "جزاء", "جنائي"]
    is_legal = any(kw in msg_lower for kw in legal_kws)
    ar_count = sum(1 for c in message if '\u0600' <= c <= '\u06FF')
    lang = "ar" if ar_count / len(message) > 0.3 else "fr"
    return IntentSchema(
        intent="legal_question" if is_legal else "discussion",
        is_legal=is_legal, language=lang,
        confidence=0.6 if is_legal else 0.3
    )


async def _extract_intent(message: str) -> IntentSchema:
    lang_hint = _detect_language_heuristic(message)
    system = NER_SYSTEM_AR if lang_hint == "ar" else NER_SYSTEM_FR

    try:
        raw = await _call_llm(message, system, use_json=True, temperature=0.0, max_tokens=1024)
        clean = _sanitize_json(raw)
        parsed = json.loads(clean)
        return IntentSchema(**parsed)
    except Exception as e:
        log.warning(f"NER LLM failed: {e}")
        return _fallback_intent_heuristic(message)


# ── RÉPONSES STATIQUES ──────────────────────────────────────────────────────

GREETINGS = {
    "ar": "مرحباً! أنا Lexibot، مساعدك القانوني المتخصص في القانون التونسي. كيف يمكنني مساعدتك؟",
    "fr": "Bonjour ! Je suis Lexibot, votre assistant juridique spécialisé en droit tunisien. Comment puis-je vous aider ?",
}

OFF_TOPIC = {
    "ar": "أنا متخصص حصرياً في القانون التونسي. هل لديك سؤال قانوني محدد؟",
    "fr": "Je suis exclusivement spécialisé en droit tunisien. Avez-vous une question juridique spécifique ?",
}

NO_CONTEXT = {
    "ar": "عذراً، لم أجد معلومات كافية في قاعدة البيانات. يمكنك محاولة صياغة السؤال بشكل مختلف.",
    "fr": "Désolé, je n'ai pas trouvé suffisamment d'informations. Vous pouvez reformuler votre question.",
}


# ── ROUTEUR FASTAPI ─────────────────────────────────────────────────────────

from lexios_engine.utils.prompt_builder import PromptBuilder
router = APIRouter(tags=["Legal Query v5"])


@router.post("/ask-legal", response_model=LegalResponse)
async def ask_legal(request: LegalRequest) -> LegalResponse:
    start_time = time.perf_counter()
    session_id = request.session_id or f"anon_{int(start_time * 1000)}"
    log.info(f"[{session_id}] Query: {request.message[:80]}...")

    # ── Étape 0: Cache ──
    use_cache = request.use_cache if request.use_cache is not None else settings.USE_CACHE
    if use_cache:
        cache = get_cache()
        if cache:
            cached = cache.get(request.message)
            if cached:
                answer, score = cached
                return LegalResponse(
                    answer=answer, intent="legal_question", language="fr",
                    is_legal=True, entities=[], confidence=float(score),
                    rag_used=True, cache_hit=True, cache_score=float(score),
                    articles_found=0, sources_count=0,
                    processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                    session_id=session_id
                )

    # ── Étape 1: NER ──
    try:
        intent = await _extract_intent(request.message)
    except Exception as e:
        log.error(f"[{session_id}] NER error: {e}")
        intent = IntentSchema(intent="discussion", is_legal=False, confidence=0.0)

    log.info(f"[{session_id}] NER: intent={intent.intent}, legal={intent.is_legal}")

    # ── Étape 2: Routage Intent ──
    if intent.intent == "greeting":
        return LegalResponse(
            answer=GREETINGS.get(intent.language, GREETINGS["fr"]),
            intent="greeting", language=intent.language, is_legal=False,
            entities=[], confidence=intent.confidence, rag_used=False,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            session_id=session_id
        )

    if not intent.is_legal:
        return LegalResponse(
            answer=OFF_TOPIC.get(intent.language, OFF_TOPIC["fr"]),
            intent=intent.intent, language=intent.language, is_legal=False,
            entities=intent.entities, confidence=intent.confidence, rag_used=False,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            session_id=session_id
        )

    # ── Étape 3: HyDE ──
    use_hyde = request.use_hyde if request.use_hyde is not None else settings.USE_HYDE
    enhanced_query = request.message
    hyde_used_flag = False
    if use_hyde and intent.confidence > 0.4:
        try:
            hyde = get_hyde()
            if hyde:
                enhanced_query = await hyde.enhance(request.message)
                hyde_used_flag = True
        except Exception as e:
            log.warning(f"[{session_id}] HyDE failed: {e}")

    # ── Étape 4: RAG Retrieval (Article-Aware + Reranker) ──
    rag = get_rag()
    try:
        rag_result = await rag.query(
            question=enhanced_query,
            top_k=request.top_k,
            generate_answer=False,
            metadata_filter=request.metadata_filter
        )
        context = rag_result.get("context", "")
        chunks = rag_result.get("chunks", [])
        articles_selected = rag_result.get("stats", {}).get("articles_selected", 0)
    except Exception as e:
        log.error(f"[{session_id}] RAG retrieval error: {e}")
        try:
            fallback = await _call_llm(
                request.message,
                f"Tu es Lexibot, expert en droit tunisien. Réponds en {intent.language}.",
                temperature=0.3,
                rag=rag
            )

            return LegalResponse(
                answer=f"[Mode fallback - base documentaire indisponible]\n\n{fallback}",
                intent=intent.intent, language=intent.language, is_legal=True,
                entities=intent.entities, confidence=intent.confidence * 0.8,
                rag_used=False,
                processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                session_id=session_id,
                debug_info={"error": str(e)} if request.debug_mode else None
            )
        except Exception:
            raise HTTPException(status_code=503, detail="Service temporairement indisponible")

    # ── Étape 5: Vérification Contexte ──
    if not context or len(context.strip()) < 50:
        return LegalResponse(
            answer=NO_CONTEXT.get(intent.language, NO_CONTEXT["fr"]),
            intent=intent.intent, language=intent.language, is_legal=True,
            entities=intent.entities, confidence=intent.confidence, rag_used=True,
            articles_found=articles_selected, sources_count=len(chunks),
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            session_id=session_id
        )

    # ── Étape 6: Synthèse LLM ──
    legal_mode = request.legal_mode
    if legal_mode == "auto":
        legal_mode = _infer_legal_mode(intent)

    entity_dict = {e.type: e.value for e in intent.entities[:8]}
    system_prompt = PromptBuilder.build_synthesis_prompt(
        language=intent.language,
        context=context[:8000],
        legal_mode=legal_mode,
        entities=entity_dict,
        sources_count=len(chunks)
    )

    try:
        final_answer = await _call_llm(
            request.message,
            system_prompt,
            temperature=0.1,
            max_tokens=2048,
            rag=rag
        )

    except Exception as e:
        log.error(f"[{session_id}] Synthèse failed: {e}")
        final_answer = f"[Synthèse indisponible]\n\nContextes trouvés:\n{context[:2000]}..."

    # ── Étape 7: Cache Write ──
    if use_cache and final_answer and len(final_answer) > 20:
        try:
            cache = get_cache()
            if cache:
                cache.set(request.message, final_answer)
        except Exception as e:
            log.warning(f"Cache write failed: {e}")

    processing_time = int((time.perf_counter() - start_time) * 1000)
    log.info(f"[{session_id}] Complete: {processing_time}ms | Articles={articles_selected} | Sources={len(chunks)}")

    return LegalResponse(
        answer=final_answer,
        intent=intent.intent,
        language=intent.language,
        is_legal=True,
        entities=intent.entities,
        confidence=intent.confidence,
        rag_used=True,
        cache_hit=False,
        hyde_used=hyde_used_flag,
        articles_found=articles_selected,
        sources_count=len(chunks),
        processing_time_ms=processing_time,
        session_id=session_id,
        debug_info={
            "legal_mode": legal_mode,
            "query_enhanced": enhanced_query[:200] if hyde_used_flag else None,
            "context_length": len(context),
            "scores_preview": [c.get("scores", {}) for c in chunks[:3]]
        } if request.debug_mode else None
    )


def _infer_legal_mode(intent: IntentSchema) -> str:
    for e in intent.entities:
        if e.type in {"articles_cp", "articles_cpp", "gravite", "peine"}:
            return "penal"
        if e.type in {"articles_coc", "parties", "montants"}:
            return "civil"
    text = " ".join(e.value for e in intent.entities).lower()
    if any(k in text for k in ["crime", "délit", "peine", "prison", "pénal", "جنائي", "جزاء"]):
        return "penal"
    if any(k in text for k in ["bail", "contrat", "vente", "société", "coc", "civil", "كراء", "عقد"]):
        return "civil"
    return "general"


# ── UTILITAIRES ─────────────────────────────────────────────────────────────

@router.get("/ask-legal/health")
async def health_check() -> Dict[str, Any]:
    rag = get_rag()
    cache = get_cache()
    return {
        "status": "healthy",
        "version": "5.0.0",
        "components": {
            "rag": rag.get_health_status(),
            "cache": cache.stats if cache else {"enabled": False},
            "llm_provider": settings.LLM_PROVIDER,
            "groq_configured": bool(settings.GROQ_API_KEY),
        },
        "features": {
            "hyde": settings.USE_HYDE,
            "cache": settings.USE_CACHE,
            "reranker": settings.USE_RERANKER,
        }
    }


@router.post("/ask-legal/ingest")
async def ingest_documents(markdown_dir: str = "./data/markdown") -> Dict[str, Any]:
    from lexios_engine.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    try:
        stats = await pipeline.run(markdown_dir)
        if settings.USE_CACHE:
            try:
                cache = get_cache()
                if cache:
                    cache.clear(confirm=True)
            except Exception as e:
                log.warning(f"Cache clear failed: {e}")
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")


@router.delete("/ask-legal/cache")
async def clear_cache_endpoint(confirm: bool = False) -> Dict[str, Any]:
    if not settings.USE_CACHE:
        return {"status": "disabled"}
    cache = get_cache()
    if not cache:
        return {"status": "error", "message": "Cache non initialisé"}
    if not confirm:
        return {"status": "pending", "message": "Ajoutez ?confirm=true"}
    cache.clear(confirm=True)
    return {"status": "success", "action": "cleared"}

# ============================================================================
# NOUVEAUX ENDPOINTS: LIGHTRAG (v2 corrigé) & RAGAS
# ============================================================================

@router.post("/ask-legal/graph-context")
async def get_graph_context_endpoint(request: LegalRequest) -> Dict[str, Any]:
    """
    Endpoint LightRAG: Extrait du CONTEXTE du graphe (pas de réponse LLM).
    Retourne facts, entités, relations pour injection dans prompt unique.
    """
    start_time = time.perf_counter()
    session_id = request.session_id or f"graphctx_{int(start_time * 1000)}"

    if not settings.LIGHTRAG_ENABLED:
        return {"status": "disabled", "context": "", "entities": [], "relations": [], "facts": []}

    try:
        lightrag = await get_lightrag()
        if not lightrag:
            return {"status": "not_initialized", "context": ""}

        mode = request.metadata_filter.get("lightrag_mode", settings.LIGHTRAG_DEFAULT_MODE) if request.metadata_filter else settings.LIGHTRAG_DEFAULT_MODE

        graph_ctx = await lightrag.get_graph_context(request.message, mode=mode, top_k=request.top_k)

        processing_time = int((time.perf_counter() - start_time) * 1000)

        return {
            "status": "success",
            "context": graph_ctx.to_context_string(),
            "facts": graph_ctx.facts[:10],
            "entities": graph_ctx.entities[:15],
            "relations": graph_ctx.relations[:10],
            "mode": graph_ctx.mode,
            "timing_ms": graph_ctx.timing_ms,
            "processing_time_ms": processing_time,
            "session_id": session_id
        }
    except Exception as e:
        log.error(f"[{session_id}] LightRAG context error: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/ask-legal/pipeline", response_model=LegalResponse)
async def ask_legal_pipeline(request: LegalRequest) -> LegalResponse:
    """
    Endpoint PIPELINE: Uses LexiosPipeline orchestrator.
    Production-grade routing with QueryRouter, HyDE, LightRAG trigger,
    ContextBuilder, and PostProcessor.
    """
    start_time = time.perf_counter()
    session_id = request.session_id or f"pipeline_{int(start_time * 1000)}"
    
    try:
        pipeline = await get_pipeline()
        result = await pipeline.process(
            query=request.message,
            metadata_filter=request.metadata_filter
        )
        
        processing_time = int((time.perf_counter() - start_time) * 1000)
        
        return LegalResponse(
            answer=result.answer,
            intent="legal_question",
            language="fr",
            is_legal=True,
            entities=[],
            confidence=result.route.confidence,
            rag_used=True,
            hyde_used=result.metadata.get("hyde_applied", False),
            articles_found=len(result.context.articles_used),
            sources_count=result.context.chunks_used,
            processing_time_ms=processing_time,
            session_id=session_id,
            debug_info=result.to_dict() if request.debug_mode else None
        )
    except Exception as e:
        log.error(f"[{session_id}] Pipeline error: {e}")
        # Fallback to standard endpoint
        return await ask_legal(request)


@router.post("/ask-legal/mix", response_model=LegalResponse)
async def ask_legal_mix(request: LegalRequest) -> LegalResponse:

    """
    Endpoint MIX: Combine Lexios + LightRAG via CONTEXT FUSION.
    Architecture corrigée (v2):
      1. Lexios récupère les chunks pertinents
      2. LightRAGTrigger analyse si le graphe est pertinent
      3. Si oui: LightRAG extrait facts/entités/relations (PAS de réponse LLM)
      4. FUSION DES CONTEXTES: chunks + graph_facts
      5. UN SEUL appel LLM (Groq) avec les deux contextes
    """
    start_time = time.perf_counter()
    session_id = request.session_id or f"mix_{int(start_time * 1000)}"

    # Étape 1: Récupération contexte Lexios
    rag = get_rag()
    try:
        lexios_result = await rag.query_for_eval(request.message, top_k=request.top_k)
        lexios_context = lexios_result.get("context_text", "")
        lexios_chunks = lexios_result.get("contexts", [])
        sources = lexios_result.get("sources", [])
    except Exception as e:
        log.error(f"[{session_id}] Lexios retrieval error: {e}")
        return await ask_legal(request)

    # Étape 2: Analyse trigger LightRAG
    graph_context_str = ""
    trigger_info = None

    if settings.LIGHTRAG_ENABLED:
        try:
            lightrag = await get_lightrag()
            if lightrag:
                trigger = await lightrag.analyze_trigger(request.message)
                trigger_info = trigger
                if trigger.use_lightrag:
                    log.info(f"[{session_id}] LightRAG TRIGGERED: {trigger.reason} (conf={trigger.confidence:.2f})")
                    graph_ctx = await lightrag.get_graph_context(request.message, mode=trigger.suggested_mode, top_k=request.top_k)
                    graph_context_str = graph_ctx.to_context_string(max_length=1500)
                else:
                    log.info(f"[{session_id}] LightRAG SKIPPED: {trigger.reason}")
        except Exception as e:
            log.warning(f"[{session_id}] LightRAG trigger failed: {e}")

    # Étape 3: Vérification contexte
    if not lexios_context or len(lexios_context.strip()) < 50:
        return LegalResponse(
            answer="Désolé, je n'ai pas trouvé suffisamment d'informations.",
            intent="legal_question", language="fr", is_legal=True,
            entities=[], confidence=0.0, rag_used=True,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            session_id=session_id
        )

    # Étape 4: GÉNÉRATION UNIQUE avec contextes fusionnés
    try:
        final_answer = await rag.generate_with_context(
            question=request.message,
            context_data={"context_text": lexios_context, "sources": sources},
            graph_context=graph_context_str if graph_context_str else None
        )
    except Exception as e:
        log.error(f"[{session_id}] Generation failed: {e}")
        final_answer = f"[Erreur génération]\n\nContextes trouvés:\n{lexios_context[:1000]}..."

    processing_time = int((time.perf_counter() - start_time) * 1000)

    log.info(f"[{session_id}] Mix complete: {processing_time}ms | LightRAG={'ON' if graph_context_str else 'OFF'} | Sources={len(sources)}")

    return LegalResponse(
        answer=final_answer,
        intent="legal_question",
        language="fr",
        is_legal=True,
        entities=[],
        confidence=0.9,
        rag_used=True,
        articles_found=lexios_result.get("articles_selected", 0),
        sources_count=len(lexios_chunks),
        processing_time_ms=processing_time,
        session_id=session_id,
        debug_info={
            "mode": "mix_v2",
            "lightrag_triggered": bool(graph_context_str),
            "trigger_reason": trigger_info.reason if trigger_info else None,
            "trigger_confidence": trigger_info.confidence if trigger_info else None,
            "trigger_suggested_mode": trigger_info.suggested_mode if trigger_info else None,
            "graph_context_length": len(graph_context_str) if graph_context_str else 0,
            "lexios_context_length": len(lexios_context),
        } if request.debug_mode else None
    )


@router.post("/ask-legal/eval")
async def evaluate_response(request: LegalRequest) -> Dict[str, Any]:
    """Endpoint d'évaluation RAGAS d'une requête."""
    if not settings.RAGAS_ENABLED:
        return {"status": "disabled", "message": "RAGAS désactivé"}

    try:
        response = await ask_legal(request)
        evaluator = get_ragas()

        contexts = []
        if response.debug_info and "context_length" in response.debug_info:
            rag = get_rag()
            raw = await rag.query_for_eval(request.message, top_k=request.top_k)
            contexts = [c["text"] for c in raw.get("contexts", [])]

        eval_report = await evaluator.evaluate_single(
            question=request.message,
            answer=response.answer,
            contexts=contexts if contexts else ["Contexte non disponible"]
        )

        return {
            "status": "success",
            "ragas_metrics": eval_report.metrics.to_dict(),
            "acceptable": eval_report.metrics.is_acceptable(settings.RAGAS_THRESHOLD),
            "threshold": settings.RAGAS_THRESHOLD,
            "duration_ms": eval_report.duration_ms,
            "original_response": {
                "answer": response.answer[:500],
                "sources_count": response.sources_count,
                "processing_time_ms": response.processing_time_ms
            }
        }
    except Exception as e:
        log.error(f"RAGAS evaluation error: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/ask-legal/lightrag/stats")
async def lightrag_stats() -> Dict[str, Any]:
    """Statistiques du graphe LightRAG."""
    if not settings.LIGHTRAG_ENABLED:
        return {"status": "disabled"}
    try:
        lightrag = await get_lightrag()
        if not lightrag:
            return {"status": "not_initialized"}
        stats = await lightrag.get_graph_stats()
        return {"status": "success", "lightrag": stats, "enabled": True}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/ask-legal/eval/batch")
async def evaluate_batch(file_path: Optional[str] = None) -> Dict[str, Any]:
    """Évaluation batch RAGAS sur jeu de test."""
    if not settings.RAGAS_ENABLED:
        return {"status": "disabled"}
    try:
        from lexios_engine.evaluation import RagasEvaluationSuite
        suite = RagasEvaluationSuite(get_rag())
        report = await suite.run_full_evaluation(save_report=True)
        return {
            "status": "success",
            "overall_score": report.get("overall_score", 0),
            "standard_tests": report.get("standard", {}),
            "adversarial_tests": report.get("adversarial", {}),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

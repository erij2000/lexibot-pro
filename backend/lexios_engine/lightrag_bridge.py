"""
lightrag_bridge.py — Lexios LightRAG Bridge v5 (Clean Architecture)
====================================================================
GRAPH CONTEXT LAYER ONLY.

Rules:
- NO scoring influence on BM25 / Chroma / Reranker
- NO LLM calls inside graph logic (only via GroqLLMWrapper for LightRAG internals)
- NO string parsing ambiguity — structured fallback with JSON priority
- NO embedding duplication — uses global RTX2050SafeEmbedder

Architecture:
  Query → Trigger Analysis → (optional) Graph Context Extraction
  → returns GraphContext dataclass
  → consumed ONLY in final context fusion (rag_service.py)
"""

from __future__ import annotations

import re
import json
import time
import logging
import unicodedata
from pathlib import Path
import hashlib
import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

import numpy as np

from config import settings

# FIXED: break circular dependency — import from core layers
from core_llm import GroqLLMWrapper
from core_embedder import RTX2050SafeEmbedder

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
    """Structured graph context — NO raw LLM output here."""
    facts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    relevant_chunks: List[str] = field(default_factory=list)
    raw_context: str = ""
    mode: str = "disabled"
    timing_ms: float = 0.0

    def to_context_string(self, max_length: int = 1500) -> str:
        parts = []
        if self.facts:
            parts.append("FAITS RELATIONNELS:")
            for f in self.facts[:10]:
                parts.append(f"  - {f}")
        if self.entities:
            parts.append(f"\nENTITÉS: {', '.join(self.entities[:15])}")
        if self.relations:
            parts.append(f"\nRELATIONS: {', '.join(self.relations[:10])}")
        if self.relevant_chunks:
            parts.append("\nEXTRAITS:")
            for c in self.relevant_chunks[:5]:
                parts.append(f"  • {c[:300]}")

        result = "\n".join(parts)
        return result[:max_length] if len(result) > max_length else result


@dataclass
class TriggerAnalysis:
    use_lightrag: bool = False
    reason: str = ""
    confidence: float = 0.0
    suggested_mode: str = "hybrid"


# ============================================================================
# TRIGGER INTELLIGENT
# ============================================================================

class LightRAGTrigger:
    """Determines if LightRAG should activate. Threshold = 0.5."""

    RELATIONAL_KEYWORDS = {
        "fr": [
            "relation", "lien", "lié", "interagit", "connecté", "rapport",
            "dépend", "influence", "impact", "compar", "différence",
            "similitude", "tous les", "liste des", "ensemble", "cumul",
            "articles liés", "dispositions connexes", "pourquoi", "comment",
            "analyse", "synthèse",
        ],
        "ar": [
            "علاقة", "مرتبط", "متصل", "يتفاعل", "يؤثر", "فرق", "اختلاف",
            "تشابه", "مقارنة", "جميع", "قائمة", "تراكمي", "لماذا", "كيف",
            "حلل", "تلخيص",
        ]
    }

    MULTI_HOP_PATTERNS = [
        re.compile(r"article\s+\d+.+article\s+\d+"),
        re.compile(r"(code|loi).+(code|loi)"),
        re.compile(r"(bail|contrat|vente).+(obligations|pénal|civil)"),
    ]

    FACTUAL_PATTERNS = [
        re.compile(r"^quel\s+est\s+l\'article"),
        re.compile(r"^article\s+\d+"),
        re.compile(r"^définition"),
        re.compile(r"^qu\'est-ce"),
        re.compile(r"^c\'est\s+quoi"),
        re.compile(r"^donne\s+moi"),
        re.compile(r"^cite"),
    ]

    @classmethod
    def analyze(cls, question: str) -> TriggerAnalysis:
        if not settings.LIGHTRAG_ENABLED:
            return TriggerAnalysis(use_lightrag=False, reason="LightRAG disabled")

        q_lower = question.lower()
        ar_count = sum(1 for c in question if "\u0600" <= c <= "\u06FF")
        lang = "ar" if ar_count / max(len(question), 1) > 0.3 else "fr"

        score = 0.0
        reasons = []

        keywords = cls.RELATIONAL_KEYWORDS.get(lang, cls.RELATIONAL_KEYWORDS["fr"])
        matched = [k for k in keywords if k in q_lower]
        if matched:
            score += len(matched) * 0.15
            reasons.append(f"relational_keywords: {matched[:3]}")

        if "article" in q_lower or "chapitre" in q_lower:
            score += 0.3
            reasons.append("multi-articles detected")
            
        if any(w in q_lower for w in [" et ", " ou ", "comparé", " contre "]):
            score += 0.25
            reasons.append("comparative logic")

        for pattern in cls.MULTI_HOP_PATTERNS:
            if pattern.search(q_lower):
                score += 0.4
                reasons.append("multi-hop detected")
                break

        word_count = len(question.split())
        if word_count > 18:
            score += 1.0
            reasons.append("long query")
        elif word_count > 12:
            score += 0.5
            reasons.append("medium query")

        for pattern in cls.FACTUAL_PATTERNS:
            if pattern.match(q_lower):
                score -= 0.3
                reasons.append("factual pattern")
                break

        threshold = 0.5
        use_lightrag = score >= threshold

        if score > 0.7:
            mode = "hybrid"
        elif score > 0.5:
            mode = "local"
        else:
            mode = "global" if use_lightrag else "disabled"

        return TriggerAnalysis(
            use_lightrag=use_lightrag,
            reason="; ".join(reasons) if reasons else "standard",
            confidence=min(score / 1.5, 1.0),
            suggested_mode=mode,
        )


# ============================================================================
# SAFE GRAPH PARSER
# ============================================================================

class GraphContextParser:
    """
    Robust parser for LightRAG raw context strings.
    NO LLM calls — pure regex + structured fallback.
    Tries JSON first, then structured text parsing.
    """

    @classmethod
    def parse(cls, raw: str) -> Dict[str, List[str]]:
        if not raw or not raw.strip():
            return {"facts": [], "entities": [], "relations": [], "chunks": []}

        # Try JSON first (if LightRAG returns JSON or structured output)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {
                    "facts": cls._ensure_list(data.get("facts", [])),
                    "entities": cls._ensure_list(data.get("entities", [])),
                    "relations": cls._ensure_list(data.get("relations", [])),
                    "chunks": cls._ensure_list(data.get("chunks", data.get("context", []))),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Structured text parsing with section awareness
        facts: List[str] = []
        entities: List[str] = []
        relations: List[str] = []
        chunks: List[str] = []
        
        lines = raw.splitlines()
        current_section: Optional[str] = None
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            lower = line_stripped.lower()
            # Detect section headers
            if any(h in lower for h in ["fact", "fait", "facts:", "faits:", "faits relationnels"]):
                current_section = "facts"
                continue
            elif any(h in lower for h in ["entit", "entity", "entities:", "entités:", "entités identifiées"]):
                current_section = "entities"
                continue
            elif any(h in lower for h in ["relation", "relations:", "liens", "relations identifiées"]):
                current_section = "relations"
                continue
            elif any(h in lower for h in ["chunk", "extrait", "text", "source", "contexte", "passage"]):
                current_section = "chunks"
                continue
            
            # Parse bullet points
            if line_stripped.startswith(("-", "•", "*", "→", "->", "▸", "›")):
                content = line_stripped[1:].strip()
                if current_section == "facts":
                    facts.append(content)
                elif current_section == "entities":
                    entities.append(content)
                elif current_section == "relations":
                    relations.append(content)
                elif current_section == "chunks":
                    chunks.append(content)
                else:
                    # Default to facts if no section detected
                    if len(content) > 10:
                        facts.append(content)
            elif current_section == "relations" and ("->" in line_stripped or "→" in line_stripped or "--" in line_stripped):
                relations.append(line_stripped)
            elif current_section == "entities" and 3 < len(line_stripped) < 100:
                entities.append(line_stripped)
            elif current_section == "chunks":
                chunks.append(line_stripped)
            else:
                # Treat as fact if it's a substantial sentence
                if len(line_stripped) > 20:
                    facts.append(line_stripped)
        
        # Fallback: if still mostly empty, extract sentences as facts and raw as chunk
        if not any([facts, entities, relations]):
            sentences = re.split(r"(?<=[.!?])\s+", raw[:2000])
            facts = [s.strip() for s in sentences if len(s.strip()) > 20][:10]
            chunks = [raw[:500]] if not chunks else chunks
        
        # Deduplicate with Unicode-aware normalization
        seen = set()
        unique_facts = []
        for f in facts:
            # Normalize Unicode (handles accents, etc.) and lowercase
            norm = unicodedata.normalize("NFKD", f).lower().strip()
            # Remove only punctuation, preserve word characters (including Arabic)
            norm = re.sub(r"[^\w\s\u0600-\u06FF]", "", norm)
            if norm and norm not in seen:
                seen.add(norm)
                unique_facts.append(f)

        seen_ent = set()
        unique_entities = []
        for e in entities:
            # Same Unicode-aware normalization
            norm = unicodedata.normalize("NFKD", e).lower().strip()
            norm = re.sub(r"[^\w\s]", "", norm, flags=re.UNICODE)
            if norm and norm not in seen_ent:
                seen_ent.add(norm)
                unique_entities.append(e)

        return {
            "facts": unique_facts[:15],
            "entities": unique_entities[:15],
            "relations": list(dict.fromkeys(relations))[:10],
            "chunks": list(dict.fromkeys(chunks))[:5] or [raw[:500]],
        }
    
    @staticmethod
    def _ensure_list(val):
        if val is None:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [val]
        return []


# ============================================================================
# EMBEDDING ADAPTER (thin wrapper around global embedder)
# ============================================================================

class LightRAGEmbedAdapter:
    """
    Thin adapter — NO model duplication.
    Uses global RTX2050SafeEmbedder directly.
    """

    def __init__(self, embedder: RTX2050SafeEmbedder):
        self.embedder = embedder
        self.dimension = settings.EMBED_DIM

    async def encode(self, texts: List[str]) -> np.ndarray:
        embs = await self.embedder.encode_async(texts)
        return np.array(embs)

    def get_embedding_func(self):
        if not HAS_LIGHTRAG:
            raise ImportError("LightRAG not available")

        async def _embed(texts: List[str]) -> np.ndarray:
            return await self.encode(texts)

        return EmbeddingFunc(
            embedding_dim=self.dimension,
            max_token_size=8192,
            func=_embed,
        )


# ============================================================================
# LIGHTRAG BRIDGE (Context Layer Only)
# ============================================================================

class LightRAGBridge:
    """
    LightRAG Bridge — GRAPH CONTEXT LAYER ONLY.

    Responsibilities:
    - index_document(): add docs to graph
    - analyze_trigger(): decide if graph is relevant
    - get_graph_context(): extract structured facts/entities/relations

    Forbidden:
    - NO scoring influence
    - NO retrieval ranking
    - NO LLM generation for answers
    """

    def __init__(
        self,
        embedder: RTX2050SafeEmbedder,
        llm: GroqLLMWrapper,
        working_dir: Optional[str] = None,
    ):
        self.embedder = embedder
        self.llm = llm
        self.working_dir = working_dir or settings.LIGHTRAG_WORKING_DIR
        self._rag_instance: Optional[LightRAG] = None
        self._initialized = False
        self._cache: Dict[str, GraphContext] = {}
        self.trigger = LightRAGTrigger()

        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        if not HAS_LIGHTRAG:
            log.warning("LightRAG unavailable — bridge in pass-through mode")
            return

        self._embed_adapter = LightRAGEmbedAdapter(embedder)
        self._init_llm_bridge()

    def _init_llm_bridge(self):
        """Wrap GroqLLMWrapper for LightRAG internal use."""
        self._llm_func = self._make_llm_func()

    def _make_llm_func(self) -> Callable:
        async def _llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            # Build proper message list for Groq API (not raw text concatenation)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                for msg in history_messages:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            messages.append({"role": "user", "content": prompt})

            # Call LLM with structured messages
            return await self.llm(
                prompt="",  # Fix missing positional arg
                messages=messages,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 2048),
            )
        return _llm_func


    async def initialize(self):
        if not HAS_LIGHTRAG or self._initialized:
            return
        try:
            self._rag_instance = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self._llm_func,
                embedding_func=self._embed_adapter.get_embedding_func(),
            )
            await self._rag_instance.initialize_storages()
            self._initialized = True
            log.info(f"✅ LightRAG initialized: {self.working_dir}")
        except Exception as e:
            log.error(f"❌ LightRAG init failed: {e}")
            self._rag_instance = None

    async def index_document(self, doc: Dict[str, Any]) -> bool:
        """Index raw text into LightRAG graph."""
        if not self._initialized or not self._rag_instance:
            return False
        try:
            raw_text = doc.get("raw_text", "")
            if not raw_text or len(raw_text) < 50:
                return False
            await self._rag_instance.ainsert(raw_text)
            log.info(f"📊 LightRAG indexed: {len(raw_text)} chars")
            return True
        except Exception as e:
            log.error(f"LightRAG index failed: {e}")
            return False

    async def get_graph_context(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 8,
    ) -> GraphContext:
        """
        Extract structured graph context.
        NO LLM answer generation — context only.
        Returns structured GraphContext with facts/entities/relations.
        """
        if not self._initialized or not self._rag_instance:
            return GraphContext(mode="disabled")

        key = hashlib.md5(question.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]

        t0 = time.perf_counter()

        try:
            param = QueryParam(
                mode=mode,
                top_k=top_k,
                response_type="multiple paragraphs",
                only_need_context=True,
            )

            result = None
            for _ in range(2):
                try:
                    result = await self._rag_instance.aquery(question, param=param)
                    break
                except Exception as e:
                    log.warning(f"LightRAG aquery failed, retrying: {e}")
                    await asyncio.sleep(1)
                    
            if result is None:
                raise Exception("LightRAG aquery failed after 2 attempts")
                
            raw = result if isinstance(result, str) else str(result)

            # FIXED: structured parsing via GraphContextParser with JSON + fallback
            parsed = GraphContextParser.parse(raw)

            timing = (time.perf_counter() - t0) * 1000

            graph_ctx = GraphContext(
                facts=parsed["facts"],
                entities=parsed["entities"],
                relations=parsed["relations"],
                relevant_chunks=parsed["chunks"],
                raw_context=raw,
                mode=mode,
                timing_ms=timing,
            )
            
            self._cache[key] = graph_ctx
            return graph_ctx

        except Exception as e:
            log.error(f"LightRAG context extraction failed: {e}")
            return GraphContext(mode=f"{mode}_error")

    async def analyze_trigger(self, question: str) -> TriggerAnalysis:
        return self.trigger.analyze(question)

    async def get_graph_stats(self) -> Dict[str, Any]:
        if not self._initialized or not self._rag_instance:
            return {"status": "not_initialized"}
        try:
            graph_path = Path(self.working_dir) / "graph_data.json"
            if graph_path.exists():
                with open(graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "status": "active",
                    "nodes": len(data.get("nodes", [])),
                    "edges": len(data.get("edges", [])),
                    "working_dir": str(self.working_dir),
                }
            return {"status": "active", "nodes": 0, "edges": 0}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def close(self):
        if self._rag_instance and hasattr(self._rag_instance, "finalize_storages"):
            await self._rag_instance.finalize_storages()


# ============================================================================
# SINGLETON
# ============================================================================

_lightrag_bridge_instance: Optional[LightRAGBridge] = None
_lightrag_lock = asyncio.Lock()

async def get_lightrag_bridge(
    embedder: RTX2050SafeEmbedder,
    llm: GroqLLMWrapper,
) -> LightRAGBridge:
    """Thread-safe singleton getter for LightRAG bridge."""
    global _lightrag_bridge_instance
    if _lightrag_bridge_instance is None:
        async with _lightrag_lock:
            if _lightrag_bridge_instance is None:
                _lightrag_bridge_instance = LightRAGBridge(embedder, llm)
                await _lightrag_bridge_instance.initialize()
    return _lightrag_bridge_instance

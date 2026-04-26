"""
hyde.py — Hypothetical Document Embeddings v6
==============================================
Query expansion avec respect strict de settings.HYDE_MODE.
DRY: Réutilise GroqLLMWrapper (rate limiting, circuit breaker, client reuse)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional, Literal

from config import settings
from core_llm import GroqLLMWrapper

log = logging.getLogger("lexios.hyde")


class HyDE:
    """
    Générateur de documents hypothétiques pour améliorer le retrieval.
    Mode contrôlé par settings.HYDE_MODE:
    - "hypo_only": Retourne uniquement le doc hypothétique
    - "combined": Question + doc hypothétique (recommandé)
    - "query_only": Désactivé (pass-through)
    """
    
    SYSTEM_PROMPT_FR = """Tu es un juriste tunisien senior rédigeant un extrait authentique de document juridique.
Génère un passage de 100-150 mots qui répondrait à la question posée.
Style : Précis, technique, citant articles de loi tunisiens réels (COC, Code pénal, Constitution 2022).
Format : Paragraphe unique sans introduction ni conclusion métalinguistique."""

    SYSTEM_PROMPT_AR = """أنت قانوني تونسي كبير تكتب مستنداً قانونياً أصيلاً.
اكتب مقطعاً من 100-150 كلمة يجيب على السؤال المطروح.
الأسلوب: دقيق، تقني، يستشهد بنصوص قانونية تونسية حقيقية."""

    def __init__(self, llm_wrapper: Optional[GroqLLMWrapper] = None):
        # Réutilise le LLM wrapper partagé (circuit breaker, rate limiting, client pooling)
        self.llm = llm_wrapper or GroqLLMWrapper()

    def _detect_language(self, text: str) -> Literal["fr", "ar"]:
        if not text:
            return "fr"
        ar_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if ar_count / max(len(text), 1) > 0.3 else "fr"

    def _is_factual_pattern(self, query: str) -> bool:
        """
        Rule #8: HyDE must only be disabled when a query is both short AND matches a factual pattern.
        """
        # Rule #6: Normalize (lowercase + strip punctuation)
        clean = re.sub(r'[^\w\s]', '', query.lower()).strip()
        
        patterns = [
            r"^quel(le)?s?\s+est\s+(l[' ])?article",
            r"^article\s+\d+",
            r"^définition\s+de",
            r"^qu[' ]est[- ]ce\s+que",
            r"^المادة\s+\d+",
            r"^ما\s+هو\s+تعريف",
            r"^من\s+هو",
            r"^cite(z)?\s+",
            r"^donne(z)?\s+moi"
        ]
        return any(re.search(p, clean) for p in patterns)

    async def _generate(self, question: str, language: Literal["fr", "ar"]) -> str:
        """Génère le document hypothétique via le LLM wrapper partagé."""
        system = self.SYSTEM_PROMPT_AR if language == "ar" else self.SYSTEM_PROMPT_FR
        
        try:
            return await self.llm(
                f"Question : {question}",
                system_prompt=system,
                temperature=0.3,
                max_tokens=300
            )
        except Exception as e:
            log.warning(f"HyDE generation failed: {e}")
            return ""

    async def enhance(self, question: str) -> str:
        """
        Améliore la requête selon settings.HYDE_MODE.
        Mode lu dynamiquement (si changé à runtime, l'instance s'en aperçoit).
        """
        mode = getattr(settings, "HYDE_MODE", "combined")
        
        # Mode pass-through
        if mode == "query_only" or not settings.USE_HYDE:
            return question
            
        # Rule #8: Short AND Factual -> Disable HyDE
        if len(question.split()) < 6 and self._is_factual_pattern(question):
            log.debug(f"HyDE bypassed (short factual query): {question}")
            return question
            
        if not question or len(question.strip()) < 5:
            return question
            
        lang = self._detect_language(question)
        hypo = await self._generate(question, lang)
        
        if not hypo or len(hypo) < 50:
            log.debug("HyDE empty, returning original")
            return question
            
        # Mode hypo_only: retourne uniquement le document hypothétique
        if mode == "hypo_only":
            return hypo
            
        # Mode combined (par défaut): question + contexte hypothétique
        return f"{question}\n\n[Contexte juridique hypothétique]\n{hypo}"

    async def enhance_batch(self, questions: list[str]) -> list[str]:
        """Traite plusieurs questions en parallèle."""
        return await asyncio.gather(*[self.enhance(q) for q in questions])


# Singleton
_hyde_instance: Optional[HyDE] = None

def get_hyde(llm_wrapper: Optional[GroqLLMWrapper] = None) -> HyDE:
    global _hyde_instance
    if _hyde_instance is None:
        _hyde_instance = HyDE(llm_wrapper=llm_wrapper)
    return _hyde_instance

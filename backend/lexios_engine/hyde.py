"""
hyde.py — Hypothetical Document Embeddings v5
==============================================
Query expansion avec respect strict de settings.HYDE_MODE
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Literal

import httpx

from config import settings

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

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._session: Optional[httpx.AsyncClient] = None
        # Utilise le mode de config
        self.mode = settings.HYDE_MODE
        
        if self.mode == "query_only":
            log.info("HyDE disabled (query_only mode)")

    async def _get_session(self) -> httpx.AsyncClient:
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    def _detect_language(self, text: str) -> Literal["fr", "ar"]:
        if not text:
            return "fr"
        ar_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if ar_count / max(len(text), 1) > 0.3 else "fr"

    async def _generate(self, question: str, language: Literal["fr", "ar"]) -> str:
        """Génère le document hypothétique."""
        system = self.SYSTEM_PROMPT_AR if language == "ar" else self.SYSTEM_PROMPT_FR
        
        try:
            if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
                return await self._call_groq(question, system)
            else:
                return await self._call_ollama(question, system)
        except Exception as e:
            log.warning(f"HyDE generation failed: {e}")
            return ""

    async def _call_ollama(self, prompt: str, system: str) -> str:
        session = await self._get_session()
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question : {prompt}\n\nRédige l'extrait juridique répondant à cette question :"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 2048,
                "num_predict": 300
            }
        }
        
        response = await session.post(f"{settings.OLLAMA_HOST}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    async def _call_groq(self, prompt: str, system: str) -> str:
        session = await self._get_session()
        payload = {
            "model": settings.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question : {prompt}"}
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = await session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    async def enhance(self, question: str) -> str:
        """
        Améliore la requête selon settings.HYDE_MODE.
        """
        # Mode pass-through
        if self.mode == "query_only" or not settings.USE_HYDE:
            return question
            
        if not question or len(question.strip()) < 5:
            return question
            
        lang = self._detect_language(question)
        hypo = await self._generate(question, lang)
        
        if not hypo or len(hypo) < 50:
            log.debug("HyDE empty, returning original")
            return question
            
        # Mode hypo_only: retourne uniquement le document hypothétique
        if self.mode == "hypo_only":
            return hypo
            
        # Mode combined (par défaut): question + contexte hypothétique
        return f"{question}\n\n[Contexte juridique hypothétique]\n{hypo}"

    async def enhance_batch(self, questions: list[str]) -> list[str]:
        """Traite plusieurs questions en parallèle."""
        return await asyncio.gather(*[self.enhance(q) for q in questions])


# Singleton
_hyde_instance: Optional[HyDE] = None

def get_hyde() -> HyDE:
    global _hyde_instance
    if _hyde_instance is None:
        _hyde_instance = HyDE()
    return _hyde_instance
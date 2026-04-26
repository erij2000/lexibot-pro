"""
core_llm.py — Lexios Core LLM Wrapper (Corrected & Optimized)
=============================================================
Shared LLM layer to prevent circular imports.
Used by: rag_service.py, lightrag_bridge.py
"""

from __future__ import annotations

import time
import asyncio
import logging
import random
import copy
from typing import Optional, List, Dict, Any

import httpx

from config import settings

log = logging.getLogger("lexios.core_llm")

class GroqLLMWrapper:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY requise dans les variables d'environnement")
        
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = settings.GROQ_MODEL
        self.backup_model = settings.GROQ_BACKUP_MODEL
        self.timeout = settings.LLM_TIMEOUT
        
        self.last_request_time = 0
        self.min_interval = getattr(settings, "GROQ_MIN_INTERVAL", 60.0 / 20.0)

        # HTTPX Persistent Client
        self.client: Optional[httpx.AsyncClient] = None
        self._client_closed = True

        # Circuit Breaker + Thread Safety
        self._lock = asyncio.Lock()
        self.fail_count = 0
        self.max_fail = 5
        self.block_until = 0

    async def close(self):
        """Ferme proprement le client HTTP."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
        self.client = None
        self._client_closed = True

    async def _get_client(self) -> httpx.AsyncClient:
        """Récupère ou recrée le client HTTP s'il est fermé."""
        if self._client_closed or self.client is None or self.client.is_closed:
            # Optimisation des limites de connexion pour FastAPI
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
            self.client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
            self._client_closed = False
        return self.client

    async def __call__(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_backup: bool = False,
        use_json: bool = False,
    ) -> str:

        temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else settings.GROQ_MAX_TOKENS
        
        await self._respect_rate_limit()

        # Check circuit breaker sous lock pour thread-safety
        async with self._lock:
            if time.time() < self.block_until:
                raise Exception(f"LLM temporairement bloqué (circuit breaker). Réessayez dans {int(self.block_until - time.time())}s")

        models_to_try = [self.model]
        if not use_backup and self.backup_model and self.backup_model != self.model:
            models_to_try.append(self.backup_model)

        last_error = None
        
        # Préparation des messages (Deepcopy pour éviter les side-effects sur l'historique)
        if messages is not None:
            payload_messages = copy.deepcopy(messages)
        else:
            payload_messages = []
            if system_prompt:
                payload_messages.append({"role": "system", "content": system_prompt})
            payload_messages.append({"role": "user", "content": prompt})

        # Groq EXIGE le mot "json" dans le prompt si response_format est json_object
        if use_json:
            json_word_present = any("json" in str(m.get("content", "")).lower() for m in payload_messages)
            if not json_word_present:
                if payload_messages and payload_messages[0]["role"] == "system":
                    payload_messages[0]["content"] += "\nReturn your response strictly in JSON format."
                else:
                    payload_messages.insert(0, {"role": "system", "content": "You must output strictly in JSON format."})

        for model in models_to_try:
            for attempt in range(settings.GROQ_RETRY_ATTEMPTS):
                try:
                    client = await self._get_client()
                    
                    payload = {
                        "model": model,
                        "messages": payload_messages,
                        "temperature": temp,
                        "max_tokens": tokens,
                        "top_p": 0.9,
                        "stream": False,
                    }

                    if use_json:
                        payload["response_format"] = {"type": "json_object"}

                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    
                    # Gestion stricte du Rate Limit Groq (Code 429)
                    if response.status_code == 429:
                        wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                        log.warning(f"Rate limit Groq, attente {wait:.2f}s...")
                        await asyncio.sleep(wait)
                        continue
                        
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    if not content or len(content.strip()) < 2:
                        raise ValueError("Réponse de l'API vide ou invalide")
                        
                    self.last_request_time = time.time()
                    
                    # Reset du circuit breaker en cas de succès
                    async with self._lock:
                        self.fail_count = 0
                        
                    return content
                    
                except httpx.TimeoutException:
                    last_error = f"Timeout sur le modèle {model}"
                    await asyncio.sleep(settings.GROQ_RETRY_DELAY + random.uniform(0, 0.5))
                except httpx.HTTPStatusError as e:
                    last_error = f"Erreur HTTP {e.response.status_code} : {e.response.text}"
                    # Si erreur 400 (ex: prompt mal formaté), on arrête de boucler inutilement
                    if e.response.status_code == 400:
                        break 
                    await asyncio.sleep(settings.GROQ_RETRY_DELAY + random.uniform(0, 0.5))
                except Exception as e:
                    last_error = str(e)
                    if attempt < settings.GROQ_RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(settings.GROQ_RETRY_DELAY + random.uniform(0, 0.5))

        # En cas d'échec total de tous les modèles et tentatives
        async with self._lock:
            self.fail_count += 1
            if self.fail_count >= self.max_fail:
                self.block_until = time.time() + 60
                self.fail_count = 0

        raise Exception(f"Groq échec: tous les modèles/tentatives ont échoué. Dernière erreur: {last_error}")

    async def _respect_rate_limit(self):
        """Garantit qu'on respecte le délai minimum entre requêtes."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
# Lexios Engine Critical Fixes - TODO

## Phase 1: Core Infrastructure Fixes
- [x] 1. config.py - Lazy torch import (CI-safe via try/except)
- [x] 2. config.py - validate_config() returns errors, never calls sys.exit(1)
- [x] 3. config.py - Added CACHE_MAX_SIZE and ARTICLE_BOOST settings
- [x] 4. core_llm.py - Fix rate limiting (20 req/min) + messages support + use_json
- [x] 5. core_llm.py - FIXED payload bug: messages -> payload_messages
- [x] 6. core_llm.py - Thread-safe client lifecycle (_client_closed, _get_client, circuit breaker lock)
- [x] 7. core_embedder.py - Improve Arabic token fallback (// 3 instead of // 4)
- [x] 8. cache.py - Link max_size to settings.CACHE_MAX_SIZE
- [x] 9. cache.py - _embed() uses truncate_to_tokens() (token-aware, not char truncation)
- [x] 10. cache.py - set() embeds once and reuses q_emb (eliminates triple embedding)

## Phase 2: Service Layer Fixes
- [x] 11. hyde.py - Delegate to GroqLLMWrapper (remove custom HTTP client)
- [x] 12. hyde.py - Constructor accepts optional llm_wrapper for DRY
- [x] 13. lightrag_bridge.py - Fix _llm_func message format (proper Groq API message list)
- [x] 14. lightrag_bridge.py - Thread-safe singleton with asyncio.Lock
- [x] 15. rag_service.py - Make boost configurable via settings.ARTICLE_BOOST

## Phase 3: API Integration
- [x] 16. legal_query.py - _call_llm delegates to GroqLLMWrapper (reuses client, circuit breaker, rate limit)
- [x] 17. legal_query.py - Added /ask-legal/pipeline endpoint with LexiosPipeline singleton
- [x] 18. pipeline_orchestrator.py - Integrate with legal_query.py via /ask-legal/pipeline
- [x] 19. pipeline_orchestrator.py - ContextBuilder uses markdown source tags for Angular
- [x] 20. pipeline_orchestrator.py - PostProcessor.FORBIDDEN_PATTERNS trimmed (removed "je pense que")

## Phase 4: Validation
- [ ] 21. Final testing & validation (run imports, check for circular deps)

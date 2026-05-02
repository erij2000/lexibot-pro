
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

# Add engine to path
ENGINE_PATH = Path("backend/lexios_engine").resolve()
sys.path.append(str(ENGINE_PATH.parent))
sys.path.append(str(ENGINE_PATH))

import platform
if platform.system() == "Windows":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class TestRAGPipelineLogic(unittest.IsolatedAsyncioTestCase):
    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    @patch('core_llm.GroqLLMWrapper')
    async def test_full_rag_logic(self, mock_llm_class, mock_chroma_client, mock_st_class):
        """Simulate Full RAG Pipeline: Ingestion -> Embedding -> Retrieval -> Query"""
        print("\n🧪 SIMULATING FULL RAG PIPELINE (LOGIC VALIDATION)")
        print("-" * 50)

        # 1. Setup Mocks
        mock_st = MagicMock()
        mock_st_class.return_value = mock_st
        # Return dummy embeddings
        mock_st.encode.return_value = np.random.rand(2, 1024).astype(np.float32)

        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.generate.return_value = "C'est une réponse simulée basée sur l'article 1."
        mock_llm.close = MagicMock()

        # 2. Initialize Retriever
        from rag_service import HybridRetriever
        retriever = HybridRetriever()
        retriever.llm = mock_llm # Force use of mock
        
        # 3. Ingest a Document
        print("📥 Step 1: Ingesting mock document...")
        mock_doc = {
            "uid": "test_doc_001",
            "source_file": "test.pdf",
            "raw_text": "ARTICLE 1: DEFINITIONS. Le Service désigne l'application Lexios. ARTICLE 2: OBJET.",
            "chunks": [] # Chunker will run
        }
        await retriever.index_document(mock_doc)
        print("✅ Document indexed (logic pass)")

        # 4. Perform Retrieval
        print("🔍 Step 2: Running retrieval...")
        query = "C'est quoi le Service ?"
        # Mock Chroma query result
        retriever.chroma.collection.query = MagicMock(return_value={
            "ids": [["chunk_1"]],
            "distances": [[0.1]],
            "documents": [["ARTICLE 1: DEFINITIONS. Le Service désigne l'application Lexios."]]
        })
        
        chunks, stats, _ = await retriever.retrieve(query, top_k=1)
        self.assertTrue(len(chunks) > 0)
        print(f"✅ Retrieved {len(chunks)} chunks. Top score: {chunks[0].final_score:.2f}")

        # 5. Full Pipeline Process
        print("🤖 Step 3: Running LLM generation...")
        from pipeline_orchestrator import LexiosPipeline
        pipeline = LexiosPipeline(retriever)
        await pipeline.initialize()
        
        result = await pipeline.process(query)
        self.assertIsNotNone(result.answer)
        print(f"✅ Final Answer: {result.answer}")
        
        print("-" * 50)
        print("✨ RAG PIPELINE LOGIC VERIFIED")

if __name__ == "__main__":
    unittest.main()

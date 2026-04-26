"""
evaluation.py — Lexios Test Suite v5 (Clean)
=============================================
Tests pour architecture article-aware v13.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from lexios_engine.config import settings

from ragas_evaluator import (
    RagasEvaluator, TunisianLegalTestSuite, 
    EvaluationCase, BatchEvaluationReport
)
from rag_service import LexiosRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("lexios.eval")


@dataclass
class TestResult:
    component: str
    test_name: str
    passed: bool
    score: float
    details: str = ""
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self):
        status = "✅" if self.passed else "❌"
        return f"{status} [{self.component}] {self.test_name}: {self.score*100:.0f}%"


@dataclass
class EvalReport:
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)
        print(str(result))

    @property
    def global_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0.0

    def summary(self) -> str:
        lines = ["\n" + "=" * 60, "  LEXIOS ENGINE — RAPPORT v5", "=" * 60]
        for r in self.results:
            bar = "█" * int(r.score * 20) + "░" * (20 - int(r.score * 20))
            err = f" [ERR: {r.error}]" if r.error else ""
            lines.append(f"  {bar} {r.score*100:5.1f}%  {r.component}/{r.test_name}{err}")
        lines.extend([
            "─" * 60,
            f"  Score Global   : {self.global_score*100:.1f}%",
            f"  Taux Réussite  : {self.pass_rate*100:.1f}%",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "summary": {
                "global_score": round(self.global_score, 3),
                "pass_rate": round(self.pass_rate, 3),
                "total_tests": len(self.results)
            },
            "tests": [r.to_dict() for r in self.results]
        }, ensure_ascii=False, indent=2)


class FastTestSuite:
    @staticmethod
    def test_config_loading() -> TestResult:
        t0 = time.time()
        try:
            assert settings.EMBED_MODEL
            assert settings.LLM_PROVIDER in ["ollama", "groq"]
            assert 0 < settings.HYBRID_ALPHA < 1
            assert settings.ARTICLE_TOP_N >= 1
            assert settings.RERANKER_MODEL
            return TestResult("Config", "loading", True, 1.0,
                f"OK alpha={settings.HYBRID_ALPHA}", (time.time()-t0)*1000)
        except Exception as e:
            return TestResult("Config", "loading", False, 0.0, "", (time.time()-t0)*1000, str(e))

    @staticmethod
    def test_article_aware() -> TestResult:
        t0 = time.time()
        try:
            from rag_service import ArticleIndex
            idx = ArticleIndex()
            doc = {
                "uid": "DOC_001",
                "chunks": [
                    {"chunk_id": "c1", "text": "Art. 1", "article_id": "ART_001"},
                    {"chunk_id": "c2", "text": "suite", "article_id": "ART_001"},
                    {"chunk_id": "c3", "text": "Art. 2", "article_id": "ART_002"},
                ]
            }
            idx.add_document(doc)
            assert len(idx.articles) == 2
            assert idx.get_article("c1").article_id == "ART_001"
            return TestResult("ArticleAware", "structure", True, 1.0,
                f"{len(idx.articles)} articles", (time.time()-t0)*1000)
        except Exception as e:
            return TestResult("ArticleAware", "structure", False, 0.0, "", (time.time()-t0)*1000, str(e))

    @staticmethod
    def test_bm25_norm() -> TestResult:
        t0 = time.time()
        try:
            from rag_service import BM25Service
            bm25 = BM25Service()
            chunks = [
                {"chunk_id": "c1", "normalized_text": "contrat bail", "text": "Contrat"},
                {"chunk_id": "c2", "normalized_text": "code penal", "text": "Code"},
            ]
            bm25.add_chunks(chunks)
            scores = bm25.search("bail")
            for sc in scores.values():
                assert 0 <= sc <= 1
            return TestResult("BM25", "norm", True, 1.0,
                f"scores={list(scores.values())}", (time.time()-t0)*1000)
        except Exception as e:
            return TestResult("BM25", "norm", False, 0.0, "", (time.time()-t0)*1000, str(e))

    @staticmethod
    def test_cache_singleton() -> TestResult:
        t0 = time.time()
        try:
            from cache import get_cache
            c1 = get_cache()
            c2 = get_cache()
            assert c1 is c2
            return TestResult("Cache", "singleton", True, 1.0, "OK", (time.time()-t0)*1000)
        except Exception as e:
            return TestResult("Cache", "singleton", False, 0.0, "", (time.time()-t0)*1000, str(e))

    @staticmethod
    def test_extractors() -> List[TestResult]:
        results = []
        try:
            from extractors.factory import ExtractorFactory
            ex = ExtractorFactory.get_extractor("pénal")
            r = ex.extract("Art. 198 Code pénal. Vol.")
            assert len(r.get("articles_cp", [])) > 0
            results.append(TestResult("Extractors", "penal", True, 1.0, str(r), 0))
        except Exception as e:
            results.append(TestResult("Extractors", "penal", False, 0.0, "", 0, str(e)))
        return results


async def run_evaluation(mode: str = "fast") -> EvalReport:
    report = EvalReport()
    fast = FastTestSuite()
    report.add(fast.test_config_loading())
    report.add(fast.test_article_aware())
    report.add(fast.test_bm25_norm())
    report.add(fast.test_cache_singleton())
    for r in fast.test_extractors():
        report.add(r)
    print(report.summary())
    return report





# ============================================================================
# RAGAS EVALUATION SUITE (v5.1)
# ============================================================================

class RagasEvaluationSuite:
    """Suite d'évaluation RAGAS complète pour Lexios."""

    def __init__(self, rag: Optional[LexiosRAG] = None):
        self.rag = rag or LexiosRAG()
        self.evaluator = RagasEvaluator()
        self.test_suite = TunisianLegalTestSuite()

    async def run_standard_tests(self) -> BatchEvaluationReport:
        """Exécute les tests standards du droit tunisien."""
        log.info("=" * 60)
        log.info("RAGAS STANDARD EVALUATION START")
        log.info("=" * 60)

        cases = self.test_suite.get_test_cases()
        evaluated_cases = []

        for case in cases:
            # Récupérer la réponse Lexios
            response = await self.rag.query(
                case.question, 
                top_k=8, 
                generate_answer=True
            )

            # Mettre à jour le cas avec la réponse réelle
            case.answer = response.get("answer", "")
            case.contexts = [
                c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                for c in response.get("chunks", [])
            ]

            evaluated_cases.append(case)

        # Évaluer avec RAGAS
        report = await self.evaluator.evaluate_batch(evaluated_cases)

        log.info("=" * 60)
        log.info("RAGAS STANDARD EVALUATION COMPLETE")
        log.info(report.summary())
        log.info("=" * 60)

        return report

    async def run_adversarial_tests(self) -> BatchEvaluationReport:
        """Exécute les tests adversariaux."""
        log.info("RAGAS ADVERSARIAL EVALUATION START")

        cases = self.test_suite.get_adversarial_cases()
        evaluated_cases = []

        for case in cases:
            response = await self.rag.query(case.question, generate_answer=True)
            case.answer = response.get("answer", "")
            case.contexts = [
                c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                for c in response.get("chunks", [])
            ]
            evaluated_cases.append(case)

        report = await self.evaluator.evaluate_batch(evaluated_cases)
        log.info(report.summary())
        return report

    async def run_full_evaluation(self, save_report: bool = True) -> Dict[str, Any]:
        """Exécute l'évaluation complète (standard + adversarial)."""
        standard_report = await self.run_standard_tests()
        adversarial_report = await self.run_adversarial_tests()

        combined = {
            "standard": standard_report.to_dict(),
            "adversarial": adversarial_report.to_dict(),
            "overall_score": round(
                (standard_report.aggregate_metrics.get("composite_score", 0) * 0.8 +
                 adversarial_report.aggregate_metrics.get("composite_score", 0) * 0.2), 3
            ),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if save_report:
            from pathlib import Path
            report_dir = Path("./data/ragas_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = report_dir / f"full_evaluation_{timestamp}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(combined, f, ensure_ascii=False, indent=2)
            log.info(f"Full report saved: {path}")

        return combined


# ============================================================================
# UPDATED MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lexios Evaluation v5.1 (with RAGAS)")
    parser.add_argument("--fast", action="store_true", help="Tests rapides")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--adversarial", action="store_true", help="Run adversarial tests")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    if args.ragas:
        async def run_ragas():
            suite = RagasEvaluationSuite()
            if args.adversarial:
                report = await suite.run_adversarial_tests()
            else:
                report = await suite.run_full_evaluation()
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
        asyncio.run(run_ragas())
    else:
        report = asyncio.run(run_evaluation("fast"))
        if args.json:
            print(report.to_json())
        else:
            sys.exit(0 if report.pass_rate >= 0.8 else 1)
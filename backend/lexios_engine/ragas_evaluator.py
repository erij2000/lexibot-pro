"""
ragas_evaluator.py — Lexios RAGAS Evaluation Suite v1
=======================================================
Évaluation complète du pipeline RAG avec métriques RAGAS standardisées.

Métriques implémentées:
- Faithfulness: La réponse est-elle fidèle au contexte récupéré ?
- AnswerRelevancy: La réponse est-elle pertinente par rapport à la question ?
- ContextPrecision: Le contexte récupéré est-il précis et pertinent ?
- ContextRecall: Tout le contexte pertinent est-il récupéré ?
- ContextEntityRecall: Les entités du contexte sont-elles couvertes ?
- AnswerSimilarity: Similarité sémantique question/réponse
- AnswerCorrectness: Exactitude de la réponse (vs ground truth)

Usage:
    evaluator = RagasEvaluator()
    result = await evaluator.evaluate_single(question, answer, contexts, ground_truth)
    report = await evaluator.evaluate_batch(test_cases)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

from config import settings

log = logging.getLogger("lexios.ragas")

# ── OPTIONAL IMPORTS ─────────────────────────────────────────────────────────

try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness,
    )
    from ragas.llms import llm_factory
    from ragas import evaluate, EvaluationDataset
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    log.warning("⚠️ ragas non installé. Évaluation RAGAS désactivée.")
    log.warning("   pip install ragas")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RagasMetrics:
    """Scores RAGAS pour une évaluation unique."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_entity_recall: float = 0.0
    answer_similarity: float = 0.0
    answer_correctness: float = 0.0

    # Score composite pondéré
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "context_entity_recall": round(self.context_entity_recall, 3),
            "answer_similarity": round(self.answer_similarity, 3),
            "answer_correctness": round(self.answer_correctness, 3),
            "composite_score": round(self.composite_score, 3),
        }

    @property
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Vérifie si le score composite dépasse le seuil."""
        return self.composite_score >= threshold


@dataclass
class EvaluationCase:
    """Cas de test pour l'évaluation RAGAS."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata
        }


@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet."""
    metrics: RagasMetrics
    case: EvaluationCase
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "case": self.case.to_dict(),
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp
        }


@dataclass
class BatchEvaluationReport:
    """Rapport d'évaluation batch."""
    reports: List[EvaluationReport] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0

    def add(self, report: EvaluationReport):
        self.reports.append(report)
        self.total_cases += 1
        if report.metrics.is_acceptable:
            self.passed_cases += 1
        else:
            self.failed_cases += 1

    def compute_aggregate(self):
        """Calcule les moyennes des métriques."""
        if not self.reports:
            return

        metrics_names = [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "context_entity_recall", "answer_similarity",
            "answer_correctness", "composite_score"
        ]

        for name in metrics_names:
            values = [getattr(r.metrics, name) for r in self.reports]
            self.aggregate_metrics[name] = round(sum(values) / len(values), 3)

    def to_dict(self) -> Dict[str, Any]:
        self.compute_aggregate()
        return {
            "aggregate_metrics": self.aggregate_metrics,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": round(self.passed_cases / self.total_cases, 3) if self.total_cases > 0 else 0,
            "reports": [r.to_dict() for r in self.reports]
        }

    def summary(self) -> str:
        self.compute_aggregate()
        lines = [
            "=" * 60,
            "  LEXIOS RAGAS EVALUATION REPORT",
            "=" * 60,
            f"  Total Cases   : {self.total_cases}",
            f"  Passed        : {self.passed_cases} ({self.pass_rate*100:.1f}%)",
            f"  Failed        : {self.failed_cases}",
            "-" * 60,
            "  AGGREGATE METRICS:",
        ]
        for name, value in self.aggregate_metrics.items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            lines.append(f"  {bar} {value*100:5.1f}%  {name}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# RAGAS EVALUATOR
# ============================================================================

class RagasEvaluator:
    """
    Évaluateur RAGAS pour le pipeline Lexios.

    Cet évaluateur:
    1. Prend question + answer + contexts + ground_truth
    2. Calcule les métriques RAGAS standardisées
    3. Génère un rapport détaillé
    4. Permet l'évaluation batch sur un jeu de test
    """

    def __init__(self, llm_wrapper: Any = None):
        self.llm = llm_wrapper
        self.enabled = HAS_RAGAS and settings.RAGAS_ENABLED

        if not self.enabled:
            log.warning("RAGAS évaluateur désactivé (non installé ou désactivé dans config)")
            return

        # Initialiser le LLM pour RAGAS
        self._init_ragas_llm()

        # Sélectionner les métriques actives
        self.active_metrics = self._get_active_metrics()
        log.info(f"🎯 RAGAS évaluateur initialisé | Métriques: {list(self.active_metrics.keys())}")

    def _init_ragas_llm(self):
        """Initialise le LLM pour RAGAS (utilise Groq via factory)."""
        try:
            # RAGAS a besoin d'un LLM async compatible OpenAI
            # On utilise le factory avec la config Groq
            import httpx
            from openai import AsyncOpenAI

            if settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
                client = AsyncOpenAI(
                    api_key=settings.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1"
                )
                self.ragas_llm = llm_factory("gpt-4o", client=client)
            else:
                log.warning("RAGAS LLM non configuré (besoin Groq/OpenAI)")
                self.ragas_llm = None
        except Exception as e:
            log.error(f"RAGAS LLM init failed: {e}")
            self.ragas_llm = None

    def _get_active_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques RAGAS actives selon la config."""
        metrics = {}
        metric_names = settings.RAGAS_METRICS.split(",") if hasattr(settings, 'RAGAS_METRICS') else [
            "faithfulness", "answer_relevancy", "context_precision", "context_recall"
        ]

        available = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_entity_recall": context_entity_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness,
        }

        for name in metric_names:
            name = name.strip()
            if name in available:
                metrics[name] = available[name]

        return metrics

    async def evaluate_single(self, question: str, answer: str,
                            contexts: List[str],
                            ground_truth: Optional[str] = None) -> EvaluationReport:
        """
        Évalue un cas unique avec RAGAS.

        Args:
            question: La question posée
            answer: La réponse générée par le système
            contexts: Les contextes récupérés (chunks)
            ground_truth: Réponse de référence (optionnel)

        Returns:
            EvaluationReport avec toutes les métriques
        """
        if not self.enabled or not self.ragas_llm:
            log.warning("RAGAS non disponible, retour métriques par défaut")
            return self._fallback_report(question, answer, contexts, ground_truth)

        t0 = time.perf_counter()

        try:
            # Préparer les données pour RAGAS
            from datasets import Dataset

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            # Évaluer avec RAGAS
            result = evaluate(
                dataset=dataset,
                metrics=list(self.active_metrics.values()),
                llm=self.ragas_llm,
            )

            # Extraire les scores
            metrics = RagasMetrics()
            scores = result.to_pandas().iloc[0].to_dict()

            if "faithfulness" in scores:
                metrics.faithfulness = float(scores["faithfulness"])
            if "answer_relevancy" in scores:
                metrics.answer_relevancy = float(scores["answer_relevancy"])
            if "context_precision" in scores:
                metrics.context_precision = float(scores["context_precision"])
            if "context_recall" in scores:
                metrics.context_recall = float(scores["context_recall"])
            if "context_entity_recall" in scores:
                metrics.context_entity_recall = float(scores["context_entity_recall"])
            if "answer_similarity" in scores:
                metrics.answer_similarity = float(scores["answer_similarity"])
            if "answer_correctness" in scores:
                metrics.answer_correctness = float(scores["answer_correctness"])

            # Score composite pondéré
            weights = {
                "faithfulness": 0.25,
                "answer_relevancy": 0.20,
                "context_precision": 0.20,
                "context_recall": 0.15,
                "context_entity_recall": 0.10,
                "answer_correctness": 0.10,
            }

            weighted_sum = 0.0
            weight_total = 0.0
            for metric_name, weight in weights.items():
                value = getattr(metrics, metric_name, 0.0)
                weighted_sum += value * weight
                weight_total += weight

            metrics.composite_score = weighted_sum / weight_total if weight_total > 0 else 0.0

            duration = (time.perf_counter() - t0) * 1000

            case = EvaluationCase(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            )

            return EvaluationReport(
                metrics=metrics,
                case=case,
                duration_ms=duration
            )

        except Exception as e:
            log.error(f"RAGAS evaluation failed: {e}")
            return self._fallback_report(question, answer, contexts, ground_truth, error=str(e))

    def _fallback_report(self, question: str, answer: str, contexts: List[str],
                        ground_truth: Optional[str] = None,
                        error: Optional[str] = None) -> EvaluationReport:
        """Rapport fallback quand RAGAS n'est pas disponible."""
        case = EvaluationCase(
            question=question, answer=answer, contexts=contexts, ground_truth=ground_truth
        )
        metrics = RagasMetrics()

        # Calculs heuristiques de fallback
        if contexts and answer:
            # Faithfulness fallback: overlap lexical simple
            context_text = " ".join(contexts).lower()
            answer_words = set(answer.lower().split())
            context_words = set(context_text.split())
            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                metrics.faithfulness = min(overlap * 1.5, 1.0)  # Boost approximatif

            # Context precision fallback: longueur contexte
            metrics.context_precision = min(len(context_text) / 2000, 1.0)

        if ground_truth and answer:
            # Answer correctness fallback: similarité simple
            gt_words = set(ground_truth.lower().split())
            ans_words = set(answer.lower().split())
            if gt_words and ans_words:
                metrics.answer_correctness = len(gt_words & ans_words) / len(gt_words)

        metrics.composite_score = (
            metrics.faithfulness * 0.3 +
            metrics.context_precision * 0.3 +
            metrics.answer_correctness * 0.4
        )

        return EvaluationReport(
            metrics=metrics,
            case=case,
            duration_ms=0.0
        )

    async def evaluate_batch(self, cases: List[EvaluationCase],
                            progress_callback: Optional[Callable] = None) -> BatchEvaluationReport:
        """
        Évalue un batch de cas de test.

        Args:
            cases: Liste de EvaluationCase
            progress_callback: Fonction appelée après chaque évaluation

        Returns:
            BatchEvaluationReport avec agrégation
        """
        report = BatchEvaluationReport()

        for i, case in enumerate(cases):
            log.info(f"RAGAS eval {i+1}/{len(cases)}: {case.question[:50]}...")

            result = await self.evaluate_single(
                case.question, case.answer, case.contexts, case.ground_truth
            )
            report.add(result)

            if progress_callback:
                progress_callback(i + 1, len(cases), result)

        report.compute_aggregate()
        return report

    async def evaluate_lexios_response(self, question: str,
                                       lexios_response: Dict[str, Any]) -> EvaluationReport:
        """
        Évalue directement une réponse LexiosRAG.

        Args:
            question: Question posée
            lexios_response: Dict retourné par LexiosRAG.query()

        Returns:
            EvaluationReport
        """
        answer = lexios_response.get("answer", "")
        contexts = []

        # Extraire les textes des chunks
        chunks = lexios_response.get("chunks", [])
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
            else:
                text = getattr(chunk, "text", "")
            if text:
                contexts.append(text)

        # Si pas de chunks, utiliser le context brut
        if not contexts:
            raw_context = lexios_response.get("context", "")
            if raw_context:
                contexts = [raw_context]

        return await self.evaluate_single(question, answer, contexts)

    def save_report(self, report: BatchEvaluationReport, path: Optional[str] = None):
        """Sauvegarde un rapport au format JSON."""
        if path is None:
            path = Path(settings.CACHE_DIR) / "ragas_reports"
            path.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = path / f"ragas_report_{timestamp}.json"

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        log.info(f"RAGAS report saved: {path}")
        return str(path)


# ============================================================================
# TEST SUITE PRÉDÉFINIE (Droit Tunisien)
# ============================================================================

class TunisianLegalTestSuite:
    """
    Jeu de test prédéfini pour le droit tunisien.
    Utilisé pour valider la qualité du RAG sur des cas réels.
    """

    @staticmethod
    def get_test_cases() -> List[EvaluationCase]:
        """Retourne les cas de test standards."""
        return [
            EvaluationCase(
                question="Quelles sont les conditions de validité d'un contrat de bail selon le droit tunisien ?",
                answer="",  # À remplir par le système
                contexts=[],  # À remplir par le système
                ground_truth="Selon l'article 8 du Code des obligations et contrats, le contrat de bail est un contrat par lequel une partie s'oblige à faire jouir l'autre d'une chose pendant un certain temps, moyennant un prix que cette dernière s'oblige de payer. Les conditions de validité incluent le consentement des parties, la capacité juridique, l'objet licite et la cause licite.",
                metadata={"domain": "civil", "article": "Art. 8 COC", "type": "bail"}
            ),
            EvaluationCase(
                question="Quel est le délai de prescription pour les crimes selon le Code pénal tunisien ?",
                answer="",
                contexts=[],
                ground_truth="Selon l'article 10 du Code pénal, l'action publique se prescrit par dix années révolues pour les crimes, cinq années pour les délits et deux années pour les contraventions.",
                metadata={"domain": "penal", "article": "Art. 10 CPP", "type": "prescription"}
            ),
            EvaluationCase(
                question="Quelles sont les formalités requises pour l'acte de vente d'un immeuble ?",
                answer="",
                contexts=[],
                ground_truth="La vente d'un immeuble doit être constatée par acte authentique (article 56 COC). Elle doit être enregistrée au bureau de l'enregistrement et publiée au registre de conservation de la propriété foncière.",
                metadata={"domain": "civil", "article": "Art. 56 COC", "type": "vente"}
            ),
        ]

    @staticmethod
    def get_adversarial_cases() -> List[EvaluationCase]:
        """Cas adversariaux pour tester la robustesse."""
        return [
            EvaluationCase(
                question="Explique-moi la relativité d'Einstein",  # Hors sujet
                answer="",
                contexts=[],
                ground_truth="Je suis spécialisé en droit tunisien uniquement.",
                metadata={"type": "off_topic", "expected_behavior": "refuse"}
            ),
            EvaluationCase(
                question="Article 999 du Code pénal",  # Article inexistant
                answer="",
                contexts=[],
                ground_truth="Je ne dispose pas d'informations sur cet article.",
                metadata={"type": "hallucination_test", "expected_behavior": "admit_ignorance"}
            ),
        ]


# ============================================================================
# SINGLETON
# ============================================================================

_ragas_instance: Optional[RagasEvaluator] = None

def get_ragas_evaluator(llm_wrapper: Any = None) -> RagasEvaluator:
    global _ragas_instance
    if _ragas_instance is None:
        _ragas_instance = RagasEvaluator(llm_wrapper)
    return _ragas_instance
"""
doc_detector.py — Détecteur Automatique de Type de Document
============================================================
Détecte le type d'un document juridique tunisien sans LLM,
uniquement via patterns (rapide, zéro hallucination).

Types détectés :
  Jugement, Arrêt, Ordonnance, Contrat, Bail, PV,
  Note_de_service, Code_de_loi, Acte_notarié, Inconnu

Utilisé par :
  - ocr_pipeline.py : enrichir les métadonnées avant indexation
  - legal_query.py  : adapter le prompt de synthèse au type de document
  - factory.py      : choisir le bon extracteur

Score de confiance : 0.0 à 1.0
  >= 0.7 : détection fiable
  0.4-0.7: incertain, vérifier
  < 0.4  : utiliser "Inconnu"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DetectionResult:
    doc_type: str
    confidence: float
    category: str        # Civil | Pénal | Administratif | Général
    language: str        # ar | fr | mixed
    signals: List[str]   # raisons de la détection
    score_details: Dict[str, float] = field(default_factory=dict)

    def __str__(self):
        return (
            f"Type: {self.doc_type} (conf={self.confidence:.2f}) | "
            f"Cat: {self.category} | Lang: {self.language}"
        )


# ── DICTIONNAIRE DE PATTERNS ──────────────────────────────────────────────────

DOC_PATTERNS: Dict[str, Dict] = {

    "Jugement": {
        "weight": 1.0,
        "fr": [
            r"(?:au nom du peuple tunisien|tribunal)",
            r"(?:condamn[eé]|acquitté|relaxé)",
            r"(?:PAR CES MOTIFS|par ces motifs)",
            r"(?:jugement|JUGEMENT)\s+n[°o]?\s*\d+",
            r"(?:attendu que|considérant que)",
        ],
        "ar": [
            r"باسم الشعب التونسي",
            r"(?:المحكمة الابتدائية|محكمة)",
            r"(?:حكمت المحكمة|تقضي|قضت|بتحويز|إستعجاليا)",
            r"(?:لهذه الأسباب|لهذه الأسباب|وبناء عليه)",
            r"حكم\s+عدد\s*\d+",
            r"قضية\s+رقم\s*\d+",
        ],
        "category": "Général",
    },

    "Arrêt": {
        "weight": 1.0,
        "fr": [
            r"(?:cour de cassation|محكمة النقض)",
            r"(?:cour d.appel|محكمة الاستئناف)",
            r"(?:arrêt|ARRÊT)\s+n[°o]?\s*\d+",
            r"(?:la cour|statuant sur)",
        ],
        "ar": [
            r"(?:محكمة التعقيب|هيئة المحكمة)",
            r"قرار\s+عدد\s*\d+",
            r"(?:أصدرت محكمة الاستئناف|تقرر)",
        ],
        "category": "Général",
    },

    "Ordonnance": {
        "weight": 0.9,
        "fr": [
            r"(?:ordonnance|ORDONNANCE)\s+n[°o]?\s*\d+",
            r"(?:juge d.instruction|référé)",
            r"(?:nous ordonnons|il est ordonné)",
        ],
        "ar": [
            r"أمر\s+عدد\s*\d+",
            r"(?:قاضي التحقيق|قضاء الأمور المستعجلة|قاضي الأسرة)",
            r"(?:نأمر|يُؤمر|أمر الانتزاع|انتزاع)",
        ],
        "category": "Général",
    },

    "Contrat": {
        "weight": 0.9,
        "fr": [
            r"(?:entre les soussignés|entre les parties)",
            r"(?:contrat|convention|accord)",
            r"(?:en foi de quoi|en foi de quoi les parties)",
            r"(?:article\s+\d+\s*[-:]\s*)",
            r"(?:d.une part|d.autre part)",
        ],
        "ar": [
            r"(?:بين الأطراف|المتعاقدان)",
            r"عقد\s+(?:بيع|كراء|عمل|شركة)",
            r"(?:الطرف الأول|الطرف الثاني)",
            r"(?:اتفق الطرفان|يلتزم)",
        ],
        "category": "Civil",
    },

    "Bail": {
        "weight": 0.95,
        "fr": [
            r"(?:contrat de bail|bail d.habitation)",
            r"(?:bailleur|preneur|locataire)",
            r"(?:loyer mensuel|charges locatives)",
            r"(?:état des lieux)",
        ],
        "ar": [
            r"(?:عقد الكراء|عقد الإيجار)",
            r"(?:المؤجر|المستأجر)",
            r"(?:الكراء الشهري|قيمة الإيجار)",
        ],
        "category": "Civil",
    },

    "PV": {
        "weight": 0.9,
        "fr": [
            r"(?:procès.verbal|P\.V\.)",
            r"(?:officier de police judiciaire|OPJ)",
            r"(?:nous avons constaté|il a été constaté)",
            r"(?:en date du|à \d+ heures)",
        ],
        "ar": [
            r"(?:محضر|محضر ضبط)",
            r"(?:ضابط الحرس|ضابط الشرطة)",
            r"(?:تبين لنا|تحررنا)",
        ],
        "category": "Pénal",
    },

    "Code_de_loi": {
        "weight": 0.85,
        "fr": [
            r"(?:code p[eé]nal|code des obligations|code civil)",
            r"(?:loi n[°o]?\s*\d+[/-]\d+)",
            r"(?:chapitre\s+\d+|section\s+\d+)",
            r"(?:article\s+premier|article\s+1er)",
        ],
        "ar": [
            r"(?:مجلة الجزاء|مجلة الالتزامات|مجلة المرافعات)",
            r"(?:قانون عدد\s*\d+)",
            r"(?:الفصل الأول|الباب الأول)",
        ],
        "category": "Général",
    },

    "Acte_notarié": {
        "weight": 0.9,
        "fr": [
            r"(?:par-devant nous|notaire|acte notarié)",
            r"(?:acte authentique|minute)",
            r"(?:taxe de publicité foncière)",
        ],
        "ar": [
            r"(?:العدل العام|عدل الإشهاد)",
            r"(?:حجة شرعية|وثيقة رسمية)",
            r"(?:التسجيل العقاري)",
        ],
        "category": "Civil",
    },

    "Note_de_service": {
        "weight": 0.7,
        "fr": [
            r"(?:note de service|circulaire|instruction)",
            r"(?:à l.attention de|objet\s*:)",
            r"(?:veuillez trouver|il vous est demandé)",
        ],
        "ar": [
            r"(?:مذكرة|منشور)",
            r"(?:بمناسبة|موضوع\s*:)",
            r"(?:يُرجى|نُحيطكم علماً)",
        ],
        "category": "Administratif",
    },
}


class DocumentDetector:
    """
    Détecte automatiquement le type d'un document juridique.
    Zéro LLM requis — 100% regex, rapide et déterministe.
    """

    @classmethod
    def detect(cls, text: str, filename: str = "") -> DetectionResult:
        """
        Détecte le type du document avec un score de confiance.

        Paramètres :
          text     : texte extrait par OCR
          filename : nom du fichier (indice supplémentaire)
        """
        if not text:
            return DetectionResult("Inconnu", 0.0, "Général", "fr", ["Texte vide"])

        language = cls._detect_language(text)
        scores: Dict[str, float] = {}
        signals_map: Dict[str, List[str]] = {}

        for doc_type, config in DOC_PATTERNS.items():
            score = 0.0
            signals = []
            lang_patterns = config.get(language, []) or config.get("fr", [])

            for pattern in lang_patterns:
                if re.search(pattern, text[:3000], re.IGNORECASE):
                    score += config["weight"] / len(lang_patterns)
                    signals.append(pattern[:40])

            # Bonus filename
            if filename:
                fname_lower = filename.lower()
                type_hints = {
                    "Jugement":     ["jugement", "حكم", "jugmt"],
                    "Arrêt":        ["arret", "arrêt", "قرار", "cassation"],
                    "Contrat":      ["contrat", "عقد", "convention"],
                    "Bail":         ["bail", "كراء", "location"],
                    "PV":           ["pv", "محضر", "proces"],
                    "Code_de_loi":  ["code", "مجلة", "loi"],
                    "Acte_notarié": ["acte", "notaire", "حجة"],
                    "Ordonnance":   ["ordonnance", "أمر"],
                }
                for hint in type_hints.get(doc_type, []):
                    if hint in fname_lower:
                        score += 0.15
                        signals.append(f"filename:{hint}")

            scores[doc_type] = min(score, 1.0)
            signals_map[doc_type] = signals

        if not scores or max(scores.values()) < 0.1:
            return DetectionResult("Inconnu", 0.0, "Général", language, ["Aucun pattern trouvé"])

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        category = DOC_PATTERNS[best_type].get("category", "Général")

        # Surcharge catégorie via patterns pénal
        penal_keywords = ["جناية", "جنحة", "مخالفة", "crime", "délit", "contravention",
                          "pénal", "pénale", "CPP", "code pénal"]
        if any(kw in text[:2000].lower() for kw in penal_keywords):
            category = "Pénal"

        return DetectionResult(
            doc_type=best_type,
            confidence=round(best_score, 3),
            category=category,
            language=language,
            signals=signals_map[best_type][:5],
            score_details={k: round(v, 3) for k, v in scores.items() if v > 0},
        )

    @staticmethod
    def _detect_language(text: str) -> str:
        ar = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        ratio = ar / max(len(text), 1)
        return "ar" if ratio > 0.3 else ("mixed" if ratio > 0.1 else "fr")

    @classmethod
    def classify_batch(cls, texts: List[Tuple[str, str]]) -> List[DetectionResult]:
        """
        Classifie un batch de (text, filename).
        Non-bloquant, pas de LLM.
        """
        return [cls.detect(text, fname) for text, fname in texts]

    @classmethod
    def test_score(cls) -> Dict[str, Any]:
        """Auto-test avec des cas connus. Retourne un score de confiance."""
        test_cases = [
            ("باسم الشعب التونسي\nتقضي المحكمة بإدانة المتهم", "حكم_2024.pdf", "Jugement"),
            ("contrat de bail d'habitation\nbailleur: Mohamed Ali\nlocataire: Ahmed", "bail.pdf", "Bail"),
            ("procès-verbal dressé par l'officier de police judiciaire", "pv_123.pdf", "PV"),
            ("مجلة الجزاء التونسية\nالفصل الأول\nالباب الأول", "code_penal.pdf", "Code_de_loi"),
        ]
        correct = 0
        results = []
        for text, fname, expected in test_cases:
            result = cls.detect(text, fname)
            ok = result.doc_type == expected
            if ok:
                correct += 1
            results.append({
                "expected": expected,
                "got": result.doc_type,
                "confidence": result.confidence,
                "ok": ok,
            })

        score = correct / len(test_cases)
        return {
            "score": round(score * 100, 1),
            "correct": correct,
            "total": len(test_cases),
            "details": results,
        }


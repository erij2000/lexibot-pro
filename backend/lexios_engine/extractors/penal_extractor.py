"""
penal_extractor.py — Extracteur Droit Pénal v2
===============================================
Spécialisé pour: الجنائي, الجنحي, المخالفات, حوادث المرور, etc.
"""

import re
from typing import Dict, Any, List
from .base_extractor import BaseExtractor

class PenalExtractor(BaseExtractor):
    """
    Extracteur spécialisé pour les documents pénaux tunisiens
    """
    
    PATTERNS = {
        "code_penal_art": r"Art\.?\s*(\d+)\s*du\s*Code\s*pénal",
        "cpp_art": r"Art\.?\s*(\d+)\s*du\s*Code\s*de\s*procédure\s*pénale",
        "tribunal": r"(Tribunal\s+(?:de\s+)?(?:Première\s+Instance|d'Instance|de\s+Sfax|Tunis|Ariana))",
        "infraction": r"(délit|crime|contravention|infraction)",
        "peine": r"((?:amende|emprisonnement|prison)\s+(?:de\s+)?[\w\s]+)",
        "date_audience": r"(\d{1,2})\s*(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*(\d{4})"
    }

    def extract(self, text: str) -> Dict[str, Any]:
        """Extraction spécifique droit pénal"""
        entities = {
            "articles_cp": self._extract_all(text, self.PATTERNS["code_penal_art"]),
            "articles_cpp": self._extract_all(text, self.PATTERNS["cpp_art"]),
            "tribunaux": self._extract_all(text, self.PATTERNS["tribunal"]),
            "type_incident": self._detect_incident_type(text),
            "gravite": self._evaluer_gravite(text)
        }
        return entities

    def _extract_all(self, text: str, pattern: str) -> List[str]:
        return list(set(re.findall(pattern, text, re.IGNORECASE)))

    def _detect_incident_type(self, text: str) -> str:
        """Détecte le type d'incident selon les mots clés"""
        text_lower = text.lower()
        if any(w in text_lower for w in ["accident", "حادث", "دهس", "اصطدام"]):
            return "Accident de la route"
        elif any(w in text_lower for w in ["vol", "سرقة", "اختلاس"]):
            return "Vol"
        elif any(w in text_lower for w in ["agression", "عنف", "ضرب"]):
            return "Agression"
        elif any(w in text_lower for w in ["drogue", "مخدرات", "stupéfiant", "zatla"]):
            return "Stupéfiants"
        return "Non spécifié"

    def _evaluer_gravite(self, text: str) -> str:
        """Évalue la gravité selon les termes utilisés"""
        if any(w in text.lower() for w in ["crime", "جناية", " homicide", "قتل"]):
            return "Crime (Grave)"
        elif any(w in text.lower() for w in ["délit", "جنحة"]):
            return "Délit (Moyen)"
        elif any(w in text.lower() for w in ["contravention", "مخالفة"]):
            return "Contravention (Léger)"
        return "Indéterminée"

    # CORRECTION: Indentation fixée
    def validate(self, entities: Dict[str, Any]) -> bool:
        """Valide que c'est bien un document pénal."""
        has_articles = len(entities.get("articles_cp", [])) > 0 or len(entities.get("articles_cpp", [])) > 0
        has_gravite = entities.get("gravite") != "Indéterminée"
        return has_articles or has_gravite
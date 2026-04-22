"""
jurisprudence_extractor.py — Extracteur Jurisprudence Tunisienne
================================================================
Gère: Arrêts de la Cour de Cassation, jugements d'appel, décisions de tribunaux
Focus sur la structuration du raisonnement juridique
"""

import re
from typing import Dict, Any, List, Optional
from .base_extractor import BaseExtractor

class JurisprudenceExtractor(BaseExtractor):
    """
    Extracteur pour la jurisprudence tunisienne.
    Identifie la structure classique : Faits -> Droit -> Motivation -> Dispositif
    """
    
    STRUCTURE_PATTERNS = {
        "faits": r'(?:Faits\s*et\s*procédure|وقائع|الوقائع|الحيثيات)(.*?)(?:Droit|En\s*droit|القانون|المنطوق|$)',
        "moyens": r'(?:Moyens|Means|الوسائل)(.*?)(?:Réponse|Réponses|الرد|$)',
        "dispositif": r'(?:Dispositif|Par\s*ces\s*motifs|المنطوق|قررنا)(.*?)(?:|)',
        "articles_loi": r'(Articles?|Art\.)\s*(\d+[\s-]*\d*)\s*(?:du\s+)?(Code\s+\w+|loi\s+\w+)',
        "num_arret": r'(?:Arrêt|Decision|قرار)\s*(?:n°|numéro)?\s*(\d+[/\-]\d+)'
    }

    def extract(self, text: str) -> Dict[str, Any]:
        """Extraction structurée de la jurisprudence"""
        entities = {
            "type_decision": self._detect_type_decision(text),
            "juridiction": self._extract_juridiction(text),
            "numero_affaire": self._extract_numero(text),
            "date_decision": self._extract_date(text),
            "parties": self._extract_parties_judiciaires(text),
            "structure": {
                "faits": self._extract_section("faits", text),
                "moyens": self._extract_section("moyens", text),
                "dispositif": self._extract_section("dispositif", text)
            },
            "principes_juridiques": self._extract_principes(text),
            "articles_appliques": self._extract_articles(text)
        }
        return entities

    def validate(self, entities: Dict[str, Any]) -> bool:
        """Valide si c'est bien une décision de justice"""
        has_juridiction = entities.get("juridiction") is not None
        has_structure = any(entities.get("structure", {}).values())
        return has_juridiction or has_structure

    def _detect_type_decision(self, text: str) -> str:
        """Détecte le type de décision"""
        text = text[:1000].lower()  # Cherche dans l'en-tête
        
        if any(w in text for w in ["cassation", "نقض", "قرار", "تقضي"]):
            return "Arret_Cassation"
        elif any(w in text for w in ["jugement", "حكم", "محكمة"]):
            return "Jugement_TPI"
        elif any(w in text for w in ["appel", "استئناف"]):
            return "Arret_Cour_Appel"
        elif any(w in text for w in ["ordonnance", "أمر"]):
            return "Ordonnance"
        return "Decision_Judiciaire"

    def _extract_juridiction(self, text: str) -> Optional[str]:
        """Extrait la juridiction"""
        patterns = [
            r'(Cour de Cassation|محكمة النقض)',
            r'(Cour d\'Appel de\s+\w+|محكمة الاستئناف)',
            r'(Tribunal de Première Instance de\s+\w+|المحكمة الابتدائية)',
            r'(Tribunal Administratif|المحكمة الإدارية)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_numero(self, text: str) -> Optional[str]:
        """Numéro d'affaire ou d'arrêt"""
        # Pattern tunisien typique : 12345/2024 ou 123/2023
        match = re.search(r'(\d{1,5}[/\\-]\d{4})', text)
        if match:
            return match.group(1)
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Date de la décision"""
        # Cherche date proche du début (entête)
        header = text[:500]
        match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', header)
        if match:
            return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
        return None

    def _extract_parties_judiciaires(self, text: str) -> Dict[str, List[str]]:
        """Extrait demandeur/défendeur ou appelant/intimé"""
        parties = {"demandeur": [], "defendeur": [], "autres": []}
        
        # Pattern demandeur
        patterns_dem = [
            r'(?: demandeur|مطالب|المدعي)[sS]?\s*:?\s*([^\n,]+)',
            r'(?:appelant|طاعن)[sS]?\s*:?\s*([^\n,]+)'
        ]
        
        for pattern in patterns_dem:
            matches = re.findall(pattern, text, re.IGNORECASE)
            parties["demandeur"].extend([m.strip() for m in matches])
        
        # Pattern défendeur
        patterns_def = [
            r'(?:défendeur|مدعى\s+عليه|مطلوب\s+ضده)[sS]?\s*:?\s*([^\n,]+)',
            r'(?:intimé|معني|المحكوم\s+عليه)[sS]?\s*:?\s*([^\n,]+)'
        ]
        
        for pattern in patterns_def:
            matches = re.findall(pattern, text, re.IGNORECASE)
            parties["defendeur"].extend([m.strip() for m in matches])
            
        return parties

    def _extract_section(self, section_type: str, text: str) -> Optional[str]:
        """Extrait une section structurée (faits, moyens, etc.)"""
        pattern = self.STRUCTURE_PATTERNS.get(section_type)
        if not pattern:
            return None
            
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(1).strip()
            # Limite la taille
            return section[:2000] + "..." if len(section) > 2000 else section
        return None

    def _extract_principes(self, text: str) -> List[str]:
        """Extrait les principes juridiques énoncés"""
        principes = []
        
        # Phrases types : "Attendu que", "Considérant que", "حيث أن"
        patterns = [
            r'(?:Attendu\s+que|Considérant\s+que|حيث\s+أن)([^.]+)',
            r'(?:alors\s+que|بما\s+أن)([^.]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            principes.extend([m.strip() for m in matches if len(m) > 20])
            
        return principes[:10]  # Top 10 principes

    def _extract_articles(self, text: str) -> List[Dict[str, str]]:
        """Extrait les articles avec le code associé"""
        articles = []
        matches = re.finditer(self.STRUCTURE_PATTERNS["articles_loi"], text, re.IGNORECASE)
        
        for match in matches:
            articles.append({
                "reference": match.group(0),
                "numero": match.group(2),
                "code": match.group(3) if match.group(3) else "Non précisé"
            })
            
        return articles
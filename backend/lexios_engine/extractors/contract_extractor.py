"""
contract_extractor.py — Extracteur Droit Civil & Commercial
===========================================================
Gère: code de société, contrats, baux, obligations, successions
Flexible avec documents semi-structurés ou manuscrits
"""

import re
from typing import Dict, Any, List, Optional
from .base_extractor import BaseExtractor

class ContractExtractor(BaseExtractor):
    """
    Extracteur adaptable pour le droit civil tunisien.
    Fonctionne aussi bien sur des contrats types que des jugements civils.
    """
    
    # Patterns souples (tolerant aux OCR errors)
    PATTERNS = {
        "montant": r'(\d[\d\s.,]*)\s*(DT|TND|دينار|dinars?)',
        "date_contrat": r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})',
        "article_coc": r'Art\.?\s*(\d+)\s*(?:du\s+)?Code\s+(?:des\s+obligations|des\s+contrats|civil)',
        "partie_civile": r'(?:M\.|Mme|السيد|السيدة)\s*([A-Z][a-zA-Z\s]+|[\u0600-\u06FF\s]+)',
        "nature_bien": r'(appartement|terrain|immeuble|voiture|véhicule|معدات|معدات|bien\s+immobilier)',
        "duree": r'(\d+)\s*(an|mois|jour|année|سنة|شهر|يوم)s?'
    }

    def extract(self, text: str) -> Dict[str, Any]:
        """Extraction souple avec fallback"""
        text = self._normalize(text)
        
        entities = {
            "montants": self._extract_montants(text),
            "parties": self._extract_parties(text),
            "dates": self._extract_dates(text),
            "articles_coc": self._extract_articles(text),
            "nature_transaction": self._detect_transaction_type(text),
            "lieu": self._extract_lieu(text),
            "obligations": self._extract_obligations_key_phrases(text)
        }
        
        # Flexibilité : on garde aussi le texte brut pertinent
        entities["extraits_cles"] = self._extract_key_snippets(text)
        
        return entities

    def validate(self, entities: Dict[str, Any]) -> bool:
        """Validation souple : au moins 2 parties OU un montant"""
        has_parties = len(entities.get("parties", [])) >= 2
        has_montant = len(entities.get("montants", [])) > 0
        return has_parties or has_montant

    def _normalize(self, text: str) -> str:
        """Normalise le texte OCR (erreurs communes)"""
        # Fix espaces dans les montants : 1 000 DT -> 1000 DT
        text = re.sub(r'(\d)\s+(\d{3})', r'\1\2', text)
        # Fix dates arabes mal OCRisées
        text = text.replace('٠', '0').replace('١', '1').replace('٢', '2')
        return text

    def _extract_montants(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les montants avec contexte"""
        montants = []
        for match in re.finditer(self.PATTERNS["montant"], text, re.IGNORECASE):
            # Cherche le contexte (10 caractères avant/après)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            
            montants.append({
                "valeur": match.group(1).replace(" ", ""),
                "devise": match.group(2),
                "contexte": context.strip()
            })
        return montants

    def _extract_parties(self, text: str) -> List[str]:
        """Extraction flexible des parties (peut être améliorée par LLM ensuite)"""
        parties = set()
        
        # Pattern : "Entre [Nom]" ou "Clause : [Nom]"
        patterns = [
            r'Entre\s*:?\s*([\w\s]+?)(?:,|et|;)',
            r'(?:Promettant|Promettante)\s*:?\s*([^\n]+)',
            r'(?:Vendeur|Acheteur|Locator|Preneur)\s*:?\s*([^\n,]+)',
            r'الطرف\s+(?:الأول|الثاني)\s*:?\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            parties.update([m.strip() for m in matches if len(m.strip()) > 3])
            
        return list(parties)[:10]  # Limite pour éviter le bruit

    def _detect_transaction_type(self, text: str) -> str:
        """Détection souple du type de transaction"""
        indicators = {
            "vente_immobiliere": ["vente", "بيع", "شراء", "acte authentique"],
            "bail": ["bail", "location", "إيجار", "كراء"],
            "societe": ["société", "capital", "أجل", "شركة", "مساهمة"],
            "pret": ["prêt", "قرض", "emprunt", "ضمان"],
            "travail": ["contrat de travail", "عقد عمل", "salarié"]
        }
        
        text_lower = text.lower()
        scores = {}
        for trans_type, keywords in indicators.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[trans_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "non_identifie"

    def _extract_lieu(self, text: str) -> Optional[str]:
        """Extrait le lieu (Tunis, Sfax, etc.)"""
        villes = ["Tunis", "Sfax", "Sousse", "Ariana", "Ezzahra", "Ariana", "تونس", "صفاقس", "سوسة"]
        for ville in villes:
            if ville in text:
                return ville
        return None

    def _extract_obligations_key_phrases(self, text: str) -> List[str]:
        """Extrait les phrases clés sur les obligations"""
        phrases = []
        # Cherche les "doit", "s'engage", "obligé de"
        pattern = r'(?:doit|s\'engage|obligé| obliged|ملزم|يلتزم)\s+([^.]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [m.strip() for m in matches[:5]]  # Top 5 obligations

    def _extract_key_snippets(self, text: str) -> List[str]:
        """Extrait les paragraphes les plus denses en informations"""
        # Split en paragraphes et prend ceux avec des chiffres ou dates
        paragraphs = text.split('\n\n')
        importants = [p for p in paragraphs if re.search(r'\d|DT|دينار|Article', p)]
        return importants[:3]  # Top 3 paragraphes clés
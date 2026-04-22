"""factory.py — Route vers le bon extracteur selon catégorie Drive."""

from typing import Dict, Any
import logging

log = logging.getLogger("lexios.extractors")


class ExtractorFactory:

    @staticmethod
    def get_extractor(category: str, nature: str = ""):
        cat = (category or "").lower().strip()
        if cat in ("pénal", "penal", "pénale"):
            from .penal_extractor import PenalExtractor
            return PenalExtractor()
        elif cat == "civil":
            from .contract_extractor import ContractExtractor
            return ContractExtractor()
        else:
            from .jurisprudence_extractor import JurisprudenceExtractor
            return JurisprudenceExtractor()

    @staticmethod
    def enrich_entities(base: Dict[str, Any], drive_ctx: Dict[str, Any]) -> Dict[str, Any]:
        enriched = base.copy()
        if not drive_ctx:
            return enriched
        enriched["drive_category"]    = drive_ctx.get("category")
        enriched["drive_subcategory"] = drive_ctx.get("subcategory")
        enriched["drive_path"]        = drive_ctx.get("context_path", "")
        path = drive_ctx.get("context_path", "")
        if any(x in path for x in ["حوادث", "accident", "مرور"]):
            enriched["type_affaire"] = "Accident de circulation"
        elif any(x in path for x in ["مخدرات", "stupéfiant", "drogue"]):
            enriched["type_affaire"] = "Affaire de stupéfiants"
        return enriched
"""
base_extractor.py — Classe de base
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def validate(self, entities: Dict[str, Any]) -> bool:
        pass
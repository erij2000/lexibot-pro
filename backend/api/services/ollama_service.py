# backend/services/ollama_service.py
from typing import List, Dict, Any, Optional
import requests
from fastapi import HTTPException, status
import json

# Imports de la configuration et du logging
from backend.core.config import settings
from backend.core.logging_config import ollama_logger, logger

class OllamaService:
    """
    Gère toutes les interactions avec l'API Ollama.
    """
    
    def __init__(self):
        self.base_url = settings.OLLAMA_API_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.endpoint = f"{self.base_url}/api/generate"
        self.timeout = 30 # Timeout en secondes pour les requêtes LLM

        # Vérification rapide de l'accessibilité de l'API au démarrage
        self.check_api_status()

    def check_api_status(self):
        """Vérifie si l'API Ollama est accessible."""
        health_endpoint = f"{self.base_url}/api/tags"
        try:
            response = requests.get(health_endpoint, timeout=5)
            if response.status_code == 200:
                logger.info(f"Ollama API accessible à {self.base_url}.")
                return True
            else:
                logger.error(f"Ollama API répond avec le statut {response.status_code}.")
                return False
        except requests.exceptions.ConnectionError:
            error_msg = f"Impossible de se connecter à Ollama à l'URL : {self.base_url}. Le service est-il démarré ?"
            logger.critical(error_msg)
            # Ne pas lever d'exception ici pour permettre à l'API de démarrer
            return False
        except Exception as e:
            logger.error(f"Erreur inconnue lors de la vérification Ollama : {e}")
            return False

    def generate_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Envoie une requête de génération de texte à Ollama.

        :param prompt: Le nouveau message de l'utilisateur.
        :param history: L'historique des messages (optionnel).
        :return: La réponse générée par le modèle.
        """
        # Construction des messages pour l'API Ollama
        messages = history if history else []
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False # Nous voulons la réponse complète
        }
        
        ollama_logger.info(f"Requête Ollama (prompt): {prompt[:50]}...")

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status() # Lève une erreur pour les statuts 4xx/5xx

            # L'API Ollama /api/generate pour les requêtes non-streamées retourne un objet JSON
            data = response.json()
            
            # Extraction du message du modèle
            full_response = data.get('message', {}).get('content', 'Erreur de réponse du modèle.')
            
            ollama_logger.info(f"Réponse Ollama (longueur): {len(full_response)} chars.")
            
            return full_response

        except requests.exceptions.RequestException as e:
            error_detail = f"Erreur de communication avec Ollama: {e}"
            ollama_logger.error(error_detail)
            # Renvoyer une erreur 503 si le service Ollama est inaccessible ou échoue
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Le service LLM (Ollama) est indisponible ou a échoué. Veuillez vérifier la console."
            )
        except json.JSONDecodeError:
            error_detail = f"Erreur de décodage JSON de la réponse Ollama. Réponse brute: {response.text[:100]}"
            ollama_logger.error(error_detail)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Réponse invalide reçue du modèle LLM."
            )
        except Exception as e:
            error_detail = f"Erreur inattendue dans OllamaService: {e}"
            ollama_logger.error(error_detail)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur interne serveur lors de la génération de la réponse LLM."
            )

# Instance du service LLM pour l'injection de dépendances
ollama_service = OllamaService()

# Dépendance pour FastAPI
def get_ollama_service():
    """Dépendance qui fournit l'instance du service Ollama."""
    return ollama_service
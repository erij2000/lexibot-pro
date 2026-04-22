# backend/services/rasa_service.py - Service de gestion des interactions avec Rasa
from typing import List, Dict, Any, Optional
import requests
from fastapi import HTTPException, status
import json

# Imports du projet
from core.config import settings
from core.logging_config import logger

class RasaService:
    """
    Service pour gérer les interactions avec l'API Rasa.
    Utilisé pour le NLU et potentiellement la réponse si le modèle est activé.
    """
    
    def __init__(self):
        self.endpoint = settings.RASA_API_BASE_URL
        self.timeout = settings.RASA_TIMEOUT
        
        # Vérification rapide de l'accessibilité de l'API Rasa au démarrage
        self.check_api_status()

    def check_api_status(self):
        """Vérifie si l'API Rasa est accessible (en testant le health check standard)."""
        health_endpoint = self.endpoint.replace('/webhooks/rest/webhook', '/status')
        try:
            response = requests.get(health_endpoint, timeout=5)
            if response.status_code == 200 and 'version' in response.json():
                logger.info(f"Rasa API accessible à {self.endpoint}.")
                return True
            else:
                logger.warning(f"Rasa API répond avec le statut {response.status_code} à {health_endpoint}.")
                return False
        except requests.exceptions.ConnectionError:
            logger.critical(f"Impossible de se connecter à Rasa à l'URL : {self.endpoint}. Le service est-il démarré ?")
            return False
        except Exception as e:
            logger.error(f"Erreur inconnue lors de la vérification Rasa : {e}")
            return False

    def get_response(self, message: str, sender: str = "fastapi_user") -> List[Dict[str, Any]]:
        """
        Envoie un message à l'API Rasa et retourne la liste des réponses.

        :param message: Le message de l'utilisateur.
        :param sender: L'ID de l'expéditeur pour maintenir la session.
        :return: Liste des objets de réponse de Rasa (peut contenir texte, boutons, etc.).
        """
        payload = {
            "sender": sender, 
            "message": message
        }
        
        logger.info(f"Requête Rasa (sender: {sender}, message: {message[:30]}...")

        try:
            response = requests.post(
                self.endpoint, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status() # Lève une erreur pour les statuts 4xx/5xx

            # Rasa retourne toujours une liste, même si elle est vide
            rasa_response: List[Dict[str, Any]] = response.json()
            
            logger.info(f"Réponse Rasa reçue: {len(rasa_response)} objets.")
            
            return rasa_response

        except requests.exceptions.RequestException as e:
            error_detail = f"Erreur de communication avec Rasa: {e}"
            logger.error(error_detail)
            # Renvoyer une erreur 503 si le service Rasa est inaccessible ou échoue
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Le service de NLU (Rasa) est indisponible ou a échoué. Veuillez vérifier la console."
            )
        except json.JSONDecodeError:
            error_detail = f"Erreur de décodage JSON de la réponse Rasa. Réponse brute: {response.text[:100]}"
            logger.error(error_detail)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Réponse invalide reçue de l'API Rasa."
            )
        except Exception as e:
            error_detail = f"Erreur inattendue dans RasaService: {e}"
            logger.error(error_detail)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur interne serveur lors de l'interaction Rasa."
            )

# Instance du service Rasa pour l'injection de dépendances
rasa_service = RasaService()

# Dépendance pour FastAPI
def get_rasa_service():
    """Dépendance qui fournit l'instance du service Rasa."""
    return rasa_service

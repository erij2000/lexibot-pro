from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import requests
import datetime
import json
import re
from spellchecker import SpellChecker
from langdetect import detect
from typing import Optional

# Imports de la sécurité et de la DB
from database.config import get_async_session
from auth.auth_config import current_active_user
from models.user_models import User, Conversation, ConversationRead, UserRole

# ----------------------------------------------------------------------
# Configuration de base
# ----------------------------------------------------------------------

router = APIRouter(
    tags=["chatbot"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3" # Assurez-vous que ce modèle est installé sur votre serveur Ollama

# Initialisation des outils de traitement du langage
spell = SpellChecker(['fr', 'en']) # Dictionnaire pour Français et Anglais

# Schémas Pydantic pour les requêtes et réponses
class ChatRequest(BaseModel):
    """Schéma de la requête utilisateur."""
    prompt: str
    
class ChatResponse(BaseModel):
    """Schéma de la réponse du Chatbot."""
    response: str
    is_appointment_request: bool = False
    is_vip: bool
    conversation_id: int
    
# ----------------------------------------------------------------------
# Logique de Traitement du Prompt
# ----------------------------------------------------------------------

def preprocess_prompt(prompt: str) -> str:
    """Corrige l'orthographe et normalise le texte."""
    try:
        lang = detect(prompt)
    except:
        lang = 'fr'
        
    words = prompt.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return " ".join(corrected_words)

def detect_appointment_intent(prompt: str) -> bool:
    """Détecte si l'utilisateur demande un rendez-vous."""
    # Mots-clés en français et anglais
    patterns = r"(rdv|rendez-vous|appointment|date|réserver|book|fixer)"
    return bool(re.search(patterns, prompt, re.IGNORECASE))

# ----------------------------------------------------------------------
# Route principale du Chatbot
# ----------------------------------------------------------------------

@router.post("/ask", response_model=ChatResponse, summary="Envoyer une question au Chatbot (Ollama)")
async def ask_chatbot(
    request: ChatRequest,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Traite la question de l'utilisateur, vérifie l'intention de RDV, 
    appelle Ollama, enregistre la conversation, et renvoie la réponse.
    """
    original_prompt = request.prompt
    
    # 1. Vérification de l'intention de RDV
    is_appointment_request = detect_appointment_intent(original_prompt)
    
    if is_appointment_request:
        response_text = "Je vois que vous souhaitez prendre rendez-vous. Pourriez-vous me fournir vos disponibilités ou me confirmer que je dois en faire la demande formelle ?"
        is_vip = user.role in [UserRole.ADMIN, UserRole.PRACTITIONER]
        
    else:
        # 2. Pré-traitement du prompt
        processed_prompt = preprocess_prompt(original_prompt)
        
        # 3. Appel à Ollama
        system_prompt = (
            f"Vous êtes Lexibot, un assistant juridique spécialisé en droit. "
            f"Répondez avec précision et clarté. L'utilisateur est: {user.email}, rôle: {user.role.value}. "
            f"S'il demande des informations sensibles ou des conseils légaux personnalisés, "
            f"conseillez-lui de prendre un rendez-vous (utiliser la fonction RDV) car vous êtes un bot."
        )
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": processed_prompt,
            "system": system_prompt,
            "stream": False
        }
        
        try:
            ollama_response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            ollama_response.raise_for_status() # Lève une exception si le statut est une erreur (4xx ou 5xx)
            
            # Extraction de la réponse Ollama
            data = ollama_response.json()
            response_text = data.get("response", "Erreur: Aucune réponse d'Ollama.")
            is_vip = user.role in [UserRole.ADMIN, UserRole.PRACTITIONER]
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion/timeout Ollama: {e}")
            response_text = "Désolé, je ne parviens pas à contacter mon cerveau (Ollama) pour le moment. Veuillez réessayer plus tard."
            is_vip = False
            
        except json.JSONDecodeError:
            print("Erreur de décodage JSON de la réponse Ollama.")
            response_text = "J'ai reçu une réponse illisible de mon système."
            is_vip = False
            
    # 4. Enregistrement de la conversation (même en cas d'erreur Ollama, on logue l'interaction)
    try:
        new_conversation = Conversation(
            user_id=user.id,
            timestamp=datetime.datetime.now(),
            user_prompt=original_prompt,
            ai_response=response_text,
            is_appointment_request=is_appointment_request,
            user_role=user.role.value # Enregistre le rôle de l'utilisateur au moment de l'échange
        )
        session.add(new_conversation)
        await session.commit()
        await session.refresh(new_conversation)
        
        conversation_id = new_conversation.id
        
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de la conversation: {e}")
        # On ne bloque pas l'utilisateur, on continue avec un ID temporaire ou 0
        conversation_id = 0
        
    return ChatResponse(
        response=response_text,
        is_appointment_request=is_appointment_request,
        is_vip=is_vip,
        conversation_id=conversation_id
    )

@router.get("/history", response_model=List[ConversationRead], summary="Voir l'historique de conversation de l'utilisateur")
async def get_conversation_history(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Récupère l'historique complet des conversations pour l'utilisateur actuel."""
    try:
        # Filtrer par user_id
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.timestamp.desc()) # Du plus récent au plus ancien
        )
        history = result.scalars().all()
        return [ConversationRead.from_orm(conv) for conv in history]
    except Exception as e:
        print(f"Erreur lors de la récupération de l'historique: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne serveur.")
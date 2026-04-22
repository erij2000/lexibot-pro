#chatbot_router.py
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import re
from spellchecker import SpellChecker
from langdetect import detect

# Imports de la sécurité, DB et Services
from database.session import get_async_session
from auth.auth_config import current_active_user
from models.models import User, UserRole
from api.services.ollama_service import get_ollama_service, OllamaService
from api.services.conversation_service import ConversationService

# ----------------------------------------------------------------------
# Configuration de base
# ----------------------------------------------------------------------

router = APIRouter(tags=["chatbot"])
spell = SpellChecker(language=['fr', 'en']) 

# ----------------------------------------------------------------------
# Schémas Pydantic
# ----------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[uuid.UUID] = None

class ChatResponse(BaseModel):
    response: str
    is_appointment_request: bool = False
    is_vip: bool
    conversation_id: uuid.UUID 

# ----------------------------------------------------------------------
# Logique de Traitement
# ----------------------------------------------------------------------

def preprocess_prompt(prompt: str) -> str:
    """Corrige l'orthographe et normalise le texte."""
    try:
        lang = detect(prompt)
    except:
        lang = 'fr'
        
    words = prompt.split()
    corrected_words = [spell.correction(word) or word for word in words]
    return " ".join(corrected_words)

def detect_appointment_intent(prompt: str) -> bool:
    """Détecte si l'utilisateur demande un rendez-vous."""
    patterns = r"(rdv|rendez-vous|appointment|date|réserver|book|fixer)"
    return bool(re.search(patterns, prompt, re.IGNORECASE))

# ----------------------------------------------------------------------
# Route principale
# ----------------------------------------------------------------------

@router.post("/ask", response_model=ChatResponse, summary="Envoyer une question au Chatbot")
async def ask_chatbot(
    request: ChatRequest,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
    ollama: OllamaService = Depends(get_ollama_service)
):
    original_prompt = request.prompt
    conv_service = ConversationService(session)
    
    # 1. Analyse de l'intention et du statut
    is_appointment_request = detect_appointment_intent(original_prompt)
    is_vip = user.role in [UserRole.ADMIN, UserRole.LAWYER, UserRole.PREMIUM]
    
    # 2. Récupérer ou créer la conversation en base de données
    conversation = await conv_service.get_or_create_conversation(user.id, request.conversation_id)
    
    # 3. Sauvegarder la question de l'utilisateur
    await conv_service.add_message(conversation.id, "user", original_prompt)

    # 4. Générer la réponse
    if is_appointment_request:
        response_text = "Je vois que vous souhaitez prendre rendez-vous avec le cabinet. Pourriez-vous utiliser l'interface dédiée aux rendez-vous pour me fournir vos disponibilités ?"
    else:
        processed_prompt = preprocess_prompt(original_prompt)
        
        # Récupérer l'historique depuis la DB
        history = await conv_service.get_history_for_llm(conversation.id)
        
        # Contexte professionnel injecté
        system_context = (
            f"Vous êtes Lexibot, l'assistant juridique intelligent du cabinet de Maître Hila Ben Arbia. "
            f"Répondez avec précision et professionnalisme. L'utilisateur est {user.first_name} {user.last_name}."
        )
        full_prompt = f"{system_context}\n\nQuestion du client: {processed_prompt}"
        
        # Utilisation de ton service Ollama
        response_text = ollama.generate_response(prompt=full_prompt, history=history)
        
    # 5. Sauvegarder la réponse de l'IA
    await conv_service.add_message(conversation.id, "assistant", response_text)
    
    return ChatResponse(
        response=response_text,
        is_appointment_request=is_appointment_request,
        is_vip=is_vip,
        conversation_id=conversation.id
    )
# backend/schemas/chatbot_schemas.py
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    """Schéma pour la requête de chat (POST /chatbot/ask)."""
    message: str
    model: str = "phi3:mini"
    stream: bool = False
    category: str = "Général"
    chosen_language: str = "fr"

class ChatResponse(BaseModel):
    """Schéma pour la réponse de chat."""
    response: str
    model: str = "phi3:mini"
    language: str = "fr"
    status: str = "success"
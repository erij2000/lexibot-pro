#chatbot_schema.py
import uuid
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Schéma pour l'entrée utilisateur (POST /chatbot/ask)."""
    prompt: str  # Utilise 'prompt' pour matcher ton router.py

class ChatResponse(BaseModel):
    """Schéma pour la sortie envoyée au frontend."""
    response: str
    is_appointment_request: bool = False
    is_vip: bool
    conversation_id: Optional[uuid.UUID] = None
    model: str = "llama3:latest"
    status: str = "success"

# ADD THESE MISSING CLASSES:
class MessageCreate(BaseModel):
    """Schéma pour créer un message."""
    content: str
    metadata_: Optional[dict] = None

    class Config:
        from_attributes = True

class MessageRead(BaseModel):
    """Schéma pour lire un message."""
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str  # "user" or "assistant"
    content: str
    metadata_: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True
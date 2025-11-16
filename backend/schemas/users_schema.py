# backend/schemas/user_schemas.py
from typing import Optional, List
from datetime import datetime
import uuid
from pydantic import BaseModel, ConfigDict
from fastapi_users import schemas
from backend.models.models import UserRole, UserStatus, AppointmentStatus # Import des Enums et modèles

# --- Pydantic 2 Base Config ---
class AppBaseSchema(BaseModel):
    """Configuration Pydantic 2 pour la lecture depuis les modèles SQLAlchemy."""
    model_config = ConfigDict(from_attributes=True) # Remplace orm_mode = True

# -----------------------------
# SCHEMAS D'UTILISATEUR (FastAPI-Users)
# -----------------------------
class UserRead(schemas.BaseUser[uuid.UUID], AppBaseSchema):
    """Schéma de lecture des données utilisateur."""
    first_name: str
    last_name: str
    phone: Optional[str] = None
    role: UserRole
    status: UserStatus
    preferred_language: str = "fr"
    notifications_enabled: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None

class UserCreate(schemas.BaseUserCreate, AppBaseSchema):
    """Schéma de création d'utilisateur."""
    first_name: str
    last_name: str
    phone: Optional[str] = None
    preferred_language: str = "fr"
    role: UserRole = UserRole.CLIENT

class UserUpdate(schemas.BaseUserUpdate, AppBaseSchema):
    """Schéma de mise à jour des données utilisateur."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    preferred_language: Optional[str] = None
    notifications_enabled: Optional[bool] = None

# -----------------------------
# SCHEMAS DE CONVERSATION
# -----------------------------
class ConversationBase(BaseModel):
    title: str
    category: str = "Général"
    language: str = "fr"
    model_used: str = "phi3:mini"

class ConversationCreate(ConversationBase):
    messages: str # Le contenu du message

class ConversationRead(ConversationBase, AppBaseSchema):
    id: uuid.UUID
    user_id: uuid.UUID
    messages: str
    created_at: datetime
    updated_at: datetime

# -----------------------------
# SCHEMAS DE RENDEZ-VOUS
# -----------------------------
class AppointmentBase(BaseModel):
    subject: str
    description: Optional[str] = None
    preferred_date: datetime # La date demandée

class AppointmentCreate(AppointmentBase):
    duration_minutes: int = 60 # Ajouté ici pour la création

class AppointmentRead(AppointmentBase, AppBaseSchema):
    id: uuid.UUID
    client_id: uuid.UUID
    status: AppointmentStatus
    confirmed_date: Optional[datetime] = None
    duration_minutes: int = 60
    location: str
    approved_by_lawyer: Optional[bool] = None
    lawyer_notes: Optional[str] = None
    google_event_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
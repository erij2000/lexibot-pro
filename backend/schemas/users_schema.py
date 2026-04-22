from typing import Optional, List
from datetime import datetime
import uuid
from pydantic import BaseModel, ConfigDict
from fastapi_users import schemas
from models.models import UserRole, UserStatus, AppointmentStatus

class AppBaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

# --- SCHEMAS UTILISATEUR ---
class UserRead(schemas.BaseUser[uuid.UUID], AppBaseSchema):
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
    first_name: str
    last_name: str
    phone: Optional[str] = None
    preferred_language: str = "fr"
    role: UserRole = UserRole.CLIENT

class UserUpdate(schemas.BaseUserUpdate, AppBaseSchema):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    preferred_language: Optional[str] = None
    notifications_enabled: Optional[bool] = None

# --- NOUVEAU : SCHEMAS MESSAGE (Option 1) ---
class MessageBase(BaseModel):
    role: str
    content: str

class MessageRead(MessageBase, AppBaseSchema):
    id: uuid.UUID
    conversation_id: uuid.UUID
    created_at: datetime

# --- SCHEMAS CONVERSATION (Centralisés) ---
class ConversationBase(BaseModel):
    title: str
    category: str = "Général"
    language: str = "fr"
    model_used: str = "phi3:mini"

class ConversationCreate(ConversationBase):
    pass # Plus besoin de 'messages: str' car on utilise la table Message

class ConversationRead(ConversationBase, AppBaseSchema):
    id: uuid.UUID
    user_id: uuid.UUID
    messages: List[MessageRead] = [] # Liste des objets Message
    created_at: datetime
    updated_at: datetime

# --- SCHEMAS RENDEZ-VOUS ---
class AppointmentBase(BaseModel):
    subject: str
    description: Optional[str] = None
    preferred_date: datetime

class AppointmentCreate(AppointmentBase):
    duration_minutes: int = 60

class AppointmentRequest(AppointmentBase):
    reason: Optional[str] = None # Alias utilisé dans ton router

class AppointmentRead(AppointmentBase, AppBaseSchema):
    id: uuid.UUID
    client_id: uuid.UUID
    status: AppointmentStatus
    confirmed_date: Optional[datetime] = None
    duration_minutes: int = 60
    location: str
    approved_by_lawyer: Optional[bool] = None
    lawyer_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class AppointmentUpdate(BaseModel):
    subject: Optional[str] = None
    status: Optional[AppointmentStatus] = None
    confirmed_date: Optional[datetime] = None
    lawyer_notes: Optional[str] = None
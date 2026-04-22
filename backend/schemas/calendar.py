from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from uuid import UUID

class AppointmentRequest(BaseModel):
    subject: str
    description: Optional[str] = None
    preferred_date: datetime

class AppointmentRead(BaseModel):
    model_config = {"from_attributes": True}
    id: UUID
    subject: str
    description: Optional[str] = None
    preferred_date: datetime
    confirmed_date: Optional[datetime] = None
    status: str
    created_at: datetime

class AppointmentUpdateStatus(BaseModel):
    status: str
    confirmed_date: Optional[datetime] = None
    lawyer_notes: Optional[str] = None
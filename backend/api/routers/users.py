# api/routers/users.py - Routes d'administration des utilisateurs
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from api.dependencies.auth import current_active_user
from models.user_models import User, Permission, Appointment, AppointmentStatus
from schemas.calendar import AppointmentRead
from database.session import get_async_session

router = APIRouter()


@router.get("/appointments/pending", response_model=List[AppointmentRead])
async def get_pending_appointments(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Voir RDV en attente (pour votre mère)"""
    
    if not Permission.has_permission(user, "manage_appointments"):
         raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Accès refusé. Vous n'avez pas la permission 'manage_appointments'. Rôle: {user.role.value}"
        )
    
    # Logique implémentée pour récupérer les RDV
    try:
        result = await session.execute(
            select(Appointment)
            .where(Appointment.status == AppointmentStatus.PENDING)
            .order_by(Appointment.preferred_date.asc())
        )
        appointments = result.scalars().all()
        return [AppointmentRead.from_orm(appt) for appt in appointments]
    except Exception as e:
        print(f"Erreur admin get appointments: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne serveur.")
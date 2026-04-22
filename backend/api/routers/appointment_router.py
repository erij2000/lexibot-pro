from fastapi import APIRouter, Depends, HTTPException, status, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import datetime
import uuid

# Imports de la sécurité et de la DB
from database.session import get_async_session
from auth.auth_config import current_active_user
from models.models import User, Appointment, AppointmentStatus, UserRole
from schemas.users_schema import AppointmentRequest, AppointmentRead, AppointmentUpdate

# ----------------------------------------------------------------------
# Router pour la gestion des Rendez-vous côté Client
# ----------------------------------------------------------------------

router = APIRouter(
    tags=["appointments"],
)

# ----------------------------------------------------------------------
# 1. Demande de Rendez-vous (Client)
# ----------------------------------------------------------------------

@router.post(
    "/request", 
    response_model=AppointmentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Soumettre une nouvelle demande de rendez-vous"
)
async def request_appointment(
    request_data: AppointmentRequest,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Permet à un client de soumettre une demande formelle de rendez-vous.
    Le statut initial est PENDING (en attente).
    """
    try:
        # Pydantic gère déjà la conversion datetime. On enlève juste la timezone si présente.
        preferred_date = request_data.preferred_date.replace(tzinfo=None)
        
        new_appointment = Appointment(
            client_id=user.id,  # CORRECTION VITAL : C'est client_id dans ton modèle, pas user_id
            preferred_date=preferred_date,
            subject=request_data.subject,
            description=request_data.reason,
            status=AppointmentStatus.PENDING
        )
        
        session.add(new_appointment)
        await session.commit()
        await session.refresh(new_appointment)
        
        return new_appointment
    
    except Exception as e:
        print(f"Erreur lors de la création du RDV: {e}")
        await session.rollback()
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'enregistrement du RDV.")

# ----------------------------------------------------------------------
# 2. Consultation des Rendez-vous Personnels (Client)
# ----------------------------------------------------------------------

@router.get(
    "/mine",
    response_model=List[AppointmentRead],
    summary="Voir tous mes rendez-vous"
)
async def get_my_appointments(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Récupère la liste de tous les rendez-vous demandés par l'utilisateur actuel."""
    try:
        result = await session.execute(
            select(Appointment)
            .where(Appointment.client_id == user.id)
            .order_by(Appointment.preferred_date.desc())
        )
        appointments = result.scalars().all()
        return appointments
    except Exception as e:
        print(f"Erreur lors de la récupération de mes RDV: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne serveur.")

# ----------------------------------------------------------------------
# 3. Annulation d'un Rendez-vous (Client)
# ----------------------------------------------------------------------

@router.patch(
    "/{appointment_id}/cancel",
    response_model=AppointmentRead,
    summary="Annuler une de mes demandes de RDV"
)
async def cancel_my_appointment(
    # CORRECTION VITAL : appointment_id doit être un UUID, pas un int
    appointment_id: uuid.UUID = Path(..., description="ID du rendez-vous à annuler"),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Permet à l'utilisateur de changer le statut d'un de ses propres RDV en CANCELLED.
    """
    try:
        result = await session.execute(
            select(Appointment)
            .where(Appointment.id == appointment_id)
            .where(Appointment.client_id == user.id)
        )
        appointment = result.scalar_one_or_none()
        
        if appointment is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rendez-vous non trouvé ou accès refusé.")

        if appointment.status in [AppointmentStatus.COMPLETED, AppointmentStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Impossible d'annuler un rendez-vous avec le statut: {appointment.status.value}"
            )

        appointment.status = AppointmentStatus.CANCELLED
        
        session.add(appointment)
        await session.commit()
        await session.refresh(appointment)

        return appointment
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur client cancel appointment: {e}")
        await session.rollback()
        raise HTTPException(status_code=500, detail="Erreur interne serveur.")
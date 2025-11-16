from fastapi import APIRouter, Depends, HTTPException, status, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import datetime
from dateutil import parser

# Imports de la sécurité et de la DB
from database.config import get_async_session
from auth.auth_config import current_active_user
from models.user_models import User, Appointment, AppointmentStatus, UserRole
from schemas.calendar import AppointmentRequest, AppointmentRead, AppointmentUpdateStatus # Ajout de AppointmentUpdateStatus

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
        preferred_date = parser.parse(request_data.preferred_date).replace(tzinfo=None)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format de date invalide. Veuillez utiliser un format ISO 8601 (ex: YYYY-MM-DD HH:MM)."
        )

    try:
        new_appointment = Appointment(
            user_id=user.id,
            request_timestamp=datetime.datetime.now(),
            preferred_date=preferred_date,
            reason=request_data.reason,
            status=AppointmentStatus.PENDING
        )
        
        session.add(new_appointment)
        await session.commit()
        await session.refresh(new_appointment)
        
        return AppointmentRead.from_orm(new_appointment)
    
    except Exception as e:
        print(f"Erreur lors de la création du RDV: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'enregistrement du RDV.")

# ----------------------------------------------------------------------
# 2. Consultation des Rendez-vous Personnels (Client)
# ----------------------------------------------------------------------

@router.get(
    "/mine",
    response_model=List[AppointmentRead],
    summary="Voir tous mes rendez-vous (y compris en attente et passés)"
)
async def get_my_appointments(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Récupère la liste de tous les rendez-vous demandés par l'utilisateur actuel."""
    try:
        result = await session.execute(
            select(Appointment)
            .where(Appointment.user_id == user.id)
            .order_by(Appointment.preferred_date.desc())
        )
        appointments = result.scalars().all()
        return [AppointmentRead.from_orm(appt) for appt in appointments]
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
    appointment_id: int = Path(..., description="ID du rendez-vous à annuler"),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Permet à l'utilisateur de changer le statut d'un de ses propres RDV en CANCELLED.
    """
    try:
        # 1. Trouver le RDV et s'assurer qu'il appartient à cet utilisateur
        result = await session.execute(
            select(Appointment)
            .where(Appointment.id == appointment_id)
            .where(Appointment.user_id == user.id)
        )
        appointment = result.scalar_one_or_none()
        
        if appointment is None:
            # On renvoie 404 même si le RDV existe mais appartient à qqn d'autre (sécurité)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rendez-vous non trouvé ou vous n'avez pas la permission.")

        # 2. Vérifier si l'annulation est possible (ex: pas si déjà terminé)
        if appointment.status in [AppointmentStatus.COMPLETED, AppointmentStatus.REJECTED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Impossible d'annuler un rendez-vous avec le statut: {appointment.status.value}"
            )

        # 3. Mettre à jour le statut
        appointment.status = AppointmentStatus.CANCELLED
        
        # 4. Sauvegarder
        session.add(appointment)
        await session.commit()
        await session.refresh(appointment)

        return AppointmentRead.from_orm(appointment)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur client cancel appointment: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne serveur.")
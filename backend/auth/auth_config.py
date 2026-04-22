import uuid
from typing import Optional, Any

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase

from database.session import get_user_db
from models.models import User, UserRole, UserStatus
from core.config import settings
from core.logging_config import auth_logger

# ----------------------------------------------------------------------
# 1. USER MANAGER
# ----------------------------------------------------------------------

class UserManager(BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = settings.SECRET
    verification_token_secret = settings.SECRET

    def parse_id(self, value: Any) -> uuid.UUID:
        """Convertit l'ID du token JWT en uuid.UUID — correctif fastapi-users."""
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except ValueError:
            return value

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        auth_logger.info(f"Nouvel utilisateur enregistré : {user.email} (id={user.id})")

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None):
        auth_logger.info(f"Mot de passe oublié : {user.email} — token={token[:8]}...")


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


# ----------------------------------------------------------------------
# 2. STRATÉGIE JWT
# ----------------------------------------------------------------------

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(
        secret=settings.SECRET,
        lifetime_seconds=3600 * 24 * 7,  # 7 jours
    )

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# ----------------------------------------------------------------------
# 3. INSTANCE FASTAPI-USERS
# ----------------------------------------------------------------------

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

# Dépendance de base — utilisateur connecté et actif (is_active=True)
current_active_user = fastapi_users.current_user(active=True)


# ----------------------------------------------------------------------
# 4. DÉPENDANCES MÉTIER (basées sur UserRole, pas is_superuser)
# ----------------------------------------------------------------------

def require_active_status(user: User = Depends(current_active_user)) -> User:
    """Vérifie que le compte n'est pas suspendu ou banni (UserStatus)."""
    if user.status not in {UserStatus.ACTIVE, UserStatus.PENDING}:
        auth_logger.warning(f"Accès refusé — statut {user.status.value} : {user.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Votre compte est {user.status.value}. Contactez l'administrateur.",
        )
    return user


def require_admin(user: User = Depends(require_active_status)) -> User:
    """Réservé à l'ADMIN système uniquement."""
    if user.role != UserRole.ADMIN:
        auth_logger.warning(f"Accès ADMIN refusé à {user.email} (rôle: {user.role.value})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé à l'administrateur système.",
        )
    return user


def require_lawyer(user: User = Depends(require_active_status)) -> User:
    """Réservé à LAWYER et ADMIN (dashboard avocat, gestion RDV, calendrier)."""
    if user.role not in {UserRole.LAWYER, UserRole.ADMIN}:
        auth_logger.warning(f"Accès LAWYER refusé à {user.email} (rôle: {user.role.value})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès réservé à l'avocate ou à l'administrateur.",
        )
    return user


def require_premium(user: User = Depends(require_active_status)) -> User:
    """Réservé à PREMIUM, LAWYER et ADMIN (fonctionnalités avancées chatbot)."""
    if user.role not in {UserRole.PREMIUM, UserRole.LAWYER, UserRole.ADMIN}:
        auth_logger.warning(f"Accès PREMIUM refusé à {user.email} (rôle: {user.role.value})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cette fonctionnalité est réservée aux membres premium.",
        )
    return user


def check_permission(permission: str):
    """
    Factory — vérifie une permission métier précise via Permission.has_permission().

    Usage dans une route :
        @router.get("/...", dependencies=[Depends(check_permission("manage_appointments"))])
    """
    from models.models import Permission  # import local pour éviter la circularité

    def _check(user: User = Depends(require_active_status)) -> User:
        if not Permission.has_permission(user, permission):
            auth_logger.warning(
                f"Permission '{permission}' refusée à {user.email} (rôle: {user.role.value})"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission requise : '{permission}'.",
            )
        return user
    return _check
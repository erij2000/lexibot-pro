# backend/routers/auth_router.py
from fastapi import APIRouter

# Imports des schémas et de la configuration de sécurité
from backend.core.security import fastapi_users, auth_backend
from backend.schemas.user_schemas import UserRead, UserCreate, UserUpdate

# --- Router d'Authentification (Connexion / Déconnexion) ---
# Ceci crée les routes /login et /logout
auth_router = APIRouter()
auth_router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["Auth"]
)

# --- Router d'Inscription (Register) ---
# Crée la route /register
register_router = APIRouter()
register_router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/register",
    tags=["Auth"]
)

# --- Router de Gestion de Profil (par l'utilisateur lui-même) ---
# Crée les routes /me (GET, PATCH, etc.)
user_profile_router = APIRouter()
user_profile_router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["Users - Profile"]
)

# --- Router de Réinitialisation de Mot de Passe ---
password_reset_router = APIRouter()
password_reset_router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/forgot-password",
    tags=["Auth"]
)

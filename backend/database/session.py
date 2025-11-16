from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from fastapi import Depends 
from passlib.context import CryptContext
import os

# 🚨 Imports depuis la nouvelle architecture 🚨
from backend.core.config import settings # Récupère DATABASE_URL, SECRET, etc.
from backend.models.models import Base, User, UserRole, Conversation, Appointment # Modèles mis à jour
from backend.schemas.user_schemas import UserCreate # Schéma mis à jour

# Import cohérent avec la v4.0.4
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase 

# -----------------------------
# CONFIGURATION DE LA BASE (POSTGRESQL)
# -----------------------------
DATABASE_URL = settings.DATABASE_URL
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

# SQLAlchemy 2.0 Engine
# Utilisation de l'URL définie dans les settings
async_engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    future=True,
    pool_size=20, 
    max_overflow=0 
)

# SQLAlchemy 2.0 sessionmaker
async_session_maker = sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession, 
    autoflush=False,
)

# ----------------------------
# DÉPENDANCES FASTAPI
# ----------------------------
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dépendance FastAPI pour obtenir une session de DB."""
    async with async_session_maker() as session:
        yield session

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    """Dépendance FastAPI pour obtenir l'adaptateur de base de données d'utilisateur."""
    # SQLAlchemyUserDatabase utilise la classe User et la session
    yield SQLAlchemyUserDatabase(session, User)

# ----------------------------
# GESTIONNAIRE DB (Création des tables, seeding, etc.)
# ----------------------------
class DatabaseManager:
    async def init_db(self):
        """Crée toutes les tables si elles n'existent pas."""
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            print("✅ Base de données et tables créées/vérifiées.")

    async def get_user_db_instance(self, session: AsyncSession) -> SQLAlchemyUserDatabase:
        """Retourne une instance de l'adaptateur DB (utilisé dans seed_users)."""
        return SQLAlchemyUserDatabase(session, User)

    async def seed_users(self):
        """Crée l'utilisateur Admin s'il n'existe pas."""
        # Les valeurs sont maintenant récupérées de settings
        SUPER_USER_EMAIL = settings.SUPER_USER_EMAIL
        SUPER_USER_PASSWORD = settings.SUPER_USER_PASSWORD
        
        users_to_seed = [
            # L'ADMIN
            {
                "email": SUPER_USER_EMAIL,
                "password": SUPER_USER_PASSWORD,
                "is_superuser": True,
                "is_active": True,
                "is_verified": True,
                "role": UserRole.ADMIN,
                "first_name": "Admin",
                "last_name": "System",
            },
        ]

        async with async_session_maker() as session:
            try:
                # Récupérer l'instance de user_db
                user_db = await self.get_user_db_instance(session)
                
                for u in users_to_seed:
                    # Vérifie si l'utilisateur existe
                    existing_user = await user_db.get_by_email(u["email"])
                    
                    if existing_user is None:
                        # Crée l'utilisateur via le schéma UserCreate
                        # FastAPI-Users hash le mot de passe ici
                        await user_db.create(UserCreate(
                            email=u["email"], 
                            password=u["password"],
                            is_superuser=u["is_superuser"],
                            is_active=u["is_active"],
                            is_verified=u["is_verified"],
                            role=u["role"],
                            first_name=u["first_name"],
                            last_name=u["last_name"],
                        ))
                        print(f"✅ User {u['email']} created as {u['role'].value}")
                    elif existing_user.role != u["role"] or existing_user.is_superuser != u["is_superuser"]:
                        # Mise à jour si le rôle/statut a changé
                        existing_user.role = u["role"]
                        existing_user.is_superuser = u.get("is_superuser", existing_user.is_superuser)
                        existing_user.is_active = u["is_active"]
                        existing_user.is_verified = u["is_verified"]
                        existing_user.first_name = u["first_name"]
                        existing_user.last_name = u["last_name"]
                        session.add(existing_user)
                        await session.commit()
                        print(f"ℹ️ User {u['email']} updated to {u['role'].value}")
                
                # S'assurer que les changements sont persistés
                await session.close()
                
            except Exception as e:
                print(f"Erreur lors du seeding des utilisateurs: {e}")
                await session.rollback() # Annuler en cas d'erreur


    async def get_stats(self):
        """Récupère des statistiques de base de la DB."""
        async with async_session_maker() as session:
            # Requêtes SQLAlchemy 2.0 asynchrones
            user_stats = await session.execute(select(User.role, func.count(User.id)).group_by(User.role))
            conv_count = await session.execute(select(func.count(Conversation.id)))
            appt_count = await session.execute(select(func.count(Appointment.id)))
            stats = {
                "users_by_role": dict(user_stats.fetchall()),
                "total_conversations": conv_count.scalar(),
                "total_appointments": appt_count.scalar()
            }
            return stats

db_manager = DatabaseManager()

# -----------------------------
# INITIALISATION RAPIDE
# -----------------------------
async def init_database():
    """Initialise et peuple la base de données au démarrage de l'API."""
    print("🚀 Initializing database...")
    await db_manager.init_db()
    await db_manager.seed_users()
    stats = await db_manager.get_stats()
    print(f"📖 DB Stats: {stats}")
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from fastapi import Depends
from passlib.context import CryptContext

from core.config import settings
from models.models import Base, User, UserRole, Conversation, Appointment
from schemas.users_schema import UserCreate
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase

# --- Cryptage mot de passe ---
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- SQLAlchemy Engine Async ---
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_size=20,
    max_overflow=0
)

# --- Session maker ---
async_session_maker = sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
    autoflush=False
)

# --- Dépendances FastAPI ---
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)

# --- Database Manager ---
class DatabaseManager:
    async def init_db(self):
        """Créer toutes les tables si inexistantes."""
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            print("✅ Base de données et tables créées/vérifiées.")

    async def get_user_db_instance(self, session: AsyncSession) -> SQLAlchemyUserDatabase:
        return SQLAlchemyUserDatabase(session, User)

    async def seed_users(self):
        """Crée l'admin s'il n'existe pas."""
        users_to_seed = [
            {
                "email": settings.SUPER_USER_EMAIL,
                "password": settings.SUPER_USER_PASSWORD,
                "is_superuser": True,
                "is_active": True,
                "is_verified": True,
                "role": UserRole.ADMIN,
                "first_name": "Admin",
                "last_name": "System",
            }
        ]

        async with async_session_maker() as session:
            try:
                user_db = await self.get_user_db_instance(session)
                for u in users_to_seed:
                    existing_user = await user_db.get_by_email(u["email"])
                    if existing_user is None:
                        await user_db.create(UserCreate(
                            email=u["email"],
                            password=u["password"],
                            is_superuser=u["is_superuser"],
                            is_active=u["is_active"],
                            is_verified=u["is_verified"],
                            role=u["role"],
                            first_name=u["first_name"],
                            last_name=u["last_name"]
                        ))
                        print(f"✅ User {u['email']} créé comme {u['role'].value}")
                    else:
                        # Mise à jour si nécessaire
                        existing_user.role = u["role"]
                        existing_user.is_superuser = u.get("is_superuser", existing_user.is_superuser)
                        existing_user.is_active = u["is_active"]
                        existing_user.is_verified = u["is_verified"]
                        existing_user.first_name = u["first_name"]
                        existing_user.last_name = u["last_name"]
                        session.add(existing_user)
                        await session.commit()
                        print(f"ℹ️ User {u['email']} mis à jour vers {u['role'].value}")
                await session.close()
            except Exception as e:
                print(f"Erreur lors du seeding: {e}")
                await session.rollback()

    async def get_stats(self):
        async with async_session_maker() as session:
            user_stats = await session.execute(select(User.role, func.count(User.id)).group_by(User.role))
            conv_count = await session.execute(select(func.count(Conversation.id)))
            appt_count = await session.execute(select(func.count(Appointment.id)))
            return {
                "users_by_role": dict(user_stats.fetchall()),
                "total_conversations": conv_count.scalar(),
                "total_appointments": appt_count.scalar()
            }

db_manager = DatabaseManager()

# --- Initialisation rapide ---
async def init_database():
    print("🚀 Initialisation DB...")
    await db_manager.init_db()
    await db_manager.seed_users()
    stats = await db_manager.get_stats()
    print(f"📊 DB Stats: {stats}")
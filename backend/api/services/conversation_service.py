# backend/services/conversation_service.py
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, or_
from fastapi import Depends, HTTPException, status
from uuid import UUID

# Imports des dépendances et modèles
from backend.database.session import get_async_session
from backend.models.models import Conversation, Message, User, UserRole
from backend.schemas.conversation_schemas import ConversationRead, MessageRead, ConversationCreate, MessageCreate
from backend.core.logging_config import logger

class ConversationService:
    """
    Gère la logique métier pour les conversations (Conversation et Message).
    Ceci inclut la sauvegarde, la récupération, et la suppression des données.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        
    async def create_conversation(self, user_id: int) -> Conversation:
        """Crée une nouvelle conversation pour un utilisateur."""
        new_conv = Conversation(user_id=user_id)
        self.session.add(new_conv)
        await self.session.commit()
        await self.session.refresh(new_conv)
        logger.info(f"Nouvelle conversation créée : {new_conv.id} pour l'utilisateur {user_id}")
        return new_conv

    async def get_conversation(self, conversation_id: UUID, user_id: int) -> Optional[Conversation]:
        """Récupère une conversation par ID, en s'assurant qu'elle appartient à l'utilisateur."""
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id # Sécurité: l'utilisateur ne peut voir que ses propres conversations
        )
        result = await self.session.execute(stmt)
        conversation = result.scalar_one_or_none()
        
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation non trouvée ou accès non autorisé."
            )
        return conversation

    async def get_conversations_list(self, user_id: int) -> List[Conversation]:
        """Récupère la liste des conversations d'un utilisateur, triées par date de dernière mise à jour."""
        stmt = select(Conversation).where(
            Conversation.user_id == user_id
        ).order_by(
            desc(Conversation.updated_at)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_messages_by_conversation(self, conversation_id: UUID) -> List[Message]:
        """Récupère tous les messages d'une conversation, triés chronologiquement."""
        stmt = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(
            Message.created_at
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def save_message(self, message_data: MessageCreate, conversation_id: UUID, role: str) -> Message:
        """Sauvegarde un message (utilisateur ou assistant) dans la conversation."""
        new_message = Message(
            conversation_id=conversation_id,
            role=role,
            content=message_data.content,
            metadata_=message_data.metadata_ # Utilise le champ de métadonnées
        )
        self.session.add(new_message)
        await self.session.commit()
        await self.session.refresh(new_message)
        
        # Mise à jour de la date de la conversation parente
        conv = await self.session.get(Conversation, conversation_id)
        if conv:
            conv.updated_at = new_message.created_at # Met à jour la date de modification
            await self.session.commit()
        
        logger.debug(f"Message sauvegardé (rôle: {role}, conv: {conversation_id}).")
        return new_message
    
    async def delete_conversation(self, conversation_id: UUID, user: User):
        """Supprime une conversation. L'admin peut supprimer n'importe quelle conversation."""
        # Pour les admins, nous ignorons la vérification user_id
        if user.role == UserRole.ADMIN and user.is_superuser:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
        else:
            # Pour les autres, vérifiez l'appartenance
            stmt = select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id
            )
            
        result = await self.session.execute(stmt)
        conversation_to_delete = result.scalar_one_or_none()
        
        if conversation_to_delete is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation non trouvée ou accès non autorisé pour la suppression."
            )
            
        await self.session.delete(conversation_to_delete)
        await self.session.commit()
        logger.info(f"Conversation {conversation_id} supprimée par {user.email}.")
        return {"detail": "Conversation supprimée avec succès."}


# --- Dépendance pour FastAPI ---

def get_conversation_service(
    session: AsyncSession = Depends(get_async_session)
) -> ConversationService:
    """Dépendance qui fournit l'instance du service de conversation."""
    return ConversationService(session)
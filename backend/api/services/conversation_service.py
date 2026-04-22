from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from fastapi import Depends, HTTPException, status
from uuid import UUID

from database.session import get_async_session
from models.models import Conversation, Message, User, UserRole
from schemas.users_schema import ConversationRead, ConversationCreate
from schemas.chatbot_schema import MessageRead, MessageCreate
from core.logging_config import logger


class ConversationService:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_conversation(self, user_id: UUID) -> Conversation:
        new_conv = Conversation(user_id=user_id, title="Consultation Juridique")
        self.session.add(new_conv)
        await self.session.commit()
        await self.session.refresh(new_conv)
        logger.info(f"Nouvelle conversation créée : {new_conv.id} pour l'utilisateur {user_id}")
        return new_conv

    async def get_or_create_conversation(self, user_id: UUID, conversation_id: UUID = None) -> Conversation:
        if conversation_id:
            stmt = select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
            result = await self.session.execute(stmt)
            conv = result.scalar_one_or_none()
            if conv:
                return conv
        new_conv = Conversation(user_id=user_id, title="Consultation Juridique")
        self.session.add(new_conv)
        await self.session.commit()
        await self.session.refresh(new_conv)
        return new_conv

    async def get_conversation(self, conversation_id: UUID, user_id: UUID) -> Optional[Conversation]:
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )
        result = await self.session.execute(stmt)
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation non trouvée ou accès non autorisé."
            )
        return conversation

    async def get_conversations_list(self, user_id: UUID) -> List[Conversation]:
        stmt = select(Conversation).where(
            Conversation.user_id == user_id
        ).order_by(desc(Conversation.updated_at))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_messages_by_conversation(self, conversation_id: UUID) -> List[Message]:
        stmt = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def add_message(self, conversation_id: UUID, role: str, content: str) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        self.session.add(msg)
        await self.session.commit()
        await self.session.refresh(msg)
        logger.debug(f"Message sauvegardé (rôle: {role}, conv: {conversation_id}).")
        return msg

    async def get_history_for_llm(self, conversation_id: UUID) -> List[Dict[str, str]]:
        messages = await self.get_messages_by_conversation(conversation_id)
        return [{"role": m.role, "content": m.content} for m in messages]

    async def save_message(self, message_data: MessageCreate, conversation_id: UUID, role: str) -> Message:
        new_message = Message(
            conversation_id=conversation_id,
            role=role,
            content=message_data.content
        )
        self.session.add(new_message)
        conv = await self.session.get(Conversation, conversation_id)
        if conv:
            from datetime import datetime
            conv.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(new_message)
        logger.debug(f"Message sauvegardé (rôle: {role}, conv: {conversation_id}).")
        return new_message

    async def delete_conversation(self, conversation_id: UUID, user: User):
        if user.role == UserRole.ADMIN:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
        else:
            stmt = select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id
            )
        result = await self.session.execute(stmt)
        conversation_to_delete = result.scalar_one_or_none()
        if conversation_to_delete is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation non trouvée ou accès non autorisé."
            )
        await self.session.delete(conversation_to_delete)
        await self.session.commit()
        logger.info(f"Conversation {conversation_id} supprimée par {user.email}.")
        return {"detail": "Conversation supprimée avec succès."}


def get_conversation_service(
    session: AsyncSession = Depends(get_async_session)
) -> ConversationService:
    return ConversationService(session)
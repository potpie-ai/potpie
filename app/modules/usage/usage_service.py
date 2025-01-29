from datetime import datetime

from fastapi import logger
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import SessionLocal
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType


class UsageService:
    @staticmethod
    async def get_usage_data(start_date: datetime, end_date: datetime, user_id: str):
        try:
            with SessionLocal() as session:
                agent_query = (
                    session.query(
                        func.unnest(Conversation.agent_ids).label("agent_id"),
                        func.count(Message.id).label("message_count"),
                    )
                    .join(Message, Message.conversation_id == Conversation.id)
                    .filter(
                        Conversation.user_id == user_id,
                        Message.created_at.between(start_date, end_date),
                        Message.type == MessageType.HUMAN,
                    )
                    .group_by(func.unnest(Conversation.agent_ids))
                    .all()
                )

                agent_message_counts = {
                    agent_id: count for agent_id, count in agent_query
                }

                total_human_messages = sum(agent_message_counts.values())

                return {
                    "total_human_messages": total_human_messages,
                    "agent_message_counts": agent_message_counts,
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to fetch usage data: {e}")
            raise Exception("Failed to fetch usage data") from e

from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.modules.conversations.message.message_model import Message, MessageType
from app.modules.conversations.conversation.conversation_model import Conversation

class UsageService:
    @staticmethod
    async def get_usage_data(start_date: datetime, end_date: datetime, user_id: str):
        with SessionLocal() as session:
            # Query to get human messages count per agent
            results = (
                session.query(
                    Conversation.agent_ids,
                    func.count(Message.id).label('message_count')
                )
                .join(Message, Message.conversation_id == Conversation.id)
                .filter(
                    Conversation.user_id == user_id,
                    Message.created_at.between(start_date, end_date),
                    Message.type == MessageType.HUMAN
                )
                .group_by(Conversation.agent_ids)
                .all()
            )

            # Process results
            total_human_messages = sum(count for _, count in results)
            agent_message_counts = {}

            for agent_ids, count in results:
                for agent_id in agent_ids:
                    if agent_id in agent_message_counts:
                        agent_message_counts[agent_id] += count
                    else:
                        agent_message_counts[agent_id] = count

            return {
                "total_human_messages": total_human_messages,
                "agent_message_counts": agent_message_counts
            }
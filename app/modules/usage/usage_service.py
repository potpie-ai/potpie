from datetime import datetime, timedelta
import os
import httpx

from fastapi import logger, HTTPException
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

    @staticmethod
    async def check_usage_limit(user_id):
        if not os.getenv("SUBSCRIPTION_BASE_URL"):
            return True

        subscription_url = f"{os.getenv('SUBSCRIPTION_BASE_URL')}/subscriptions/info"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    subscription_url,
                    params={"user_id": user_id},
                    timeout=10.0,  # Set a reasonable timeout
                )
                response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                subscription_data = response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            # Log the error
            logger.error(
                f"Failed to fetch subscription data for user {user_id}: {str(e)}"
            )
            # Default to a free plan if the subscription service is unavailable
            subscription_data = {"plan_type": "free", "end_date": None}
        end_date_str = subscription_data.get("end_date")
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        else:
            end_date = datetime.utcnow()

        start_date = end_date - timedelta(days=30)

        usage_data = await UsageService.get_usage_data(
            start_date=start_date, end_date=end_date, user_id=user_id
        )
        total_human_messages = usage_data["total_human_messages"]

        plan_type = subscription_data.get("plan_type", "free")
        message_limit = 500 if plan_type == "pro" else 50

        if total_human_messages >= message_limit:
            raise HTTPException(
                status_code=402,
                detail=f"Message limit of {message_limit} reached for {plan_type} plan.",
            )
        else:
            return True

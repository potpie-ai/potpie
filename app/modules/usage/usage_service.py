from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message, MessageType
from app.modules.utils.logger import setup_logger
from app.modules.billing.subscription_service import billing_subscription_service

logger = setup_logger(__name__)


class UsageService:
    @staticmethod
    async def get_usage_data(
        session: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        user_id: str,
    ):
        try:
            stmt = (
                select(
                    func.unnest(Conversation.agent_ids).label("agent_id"),
                    func.count(Message.id).label("message_count"),
                )
                .select_from(Conversation)
                .join(Message, Message.conversation_id == Conversation.id)
                .where(
                    Conversation.user_id == user_id,
                    Message.created_at.between(start_date, end_date),
                    Message.type == MessageType.HUMAN,
                )
                .group_by(func.unnest(Conversation.agent_ids))
            )
            result = await session.execute(stmt)
            agent_query = result.all()

            agent_message_counts = {agent_id: count for agent_id, count in agent_query}

            total_human_messages = sum(agent_message_counts.values())

            return {
                "total_human_messages": total_human_messages,
                "agent_message_counts": agent_message_counts,
            }

        except SQLAlchemyError as e:
            logger.exception("Failed to fetch usage data: %s", e)
            raise Exception("Failed to fetch usage data") from e

    @staticmethod
    async def check_usage_limit(user_id: str, session: AsyncSession = None):
        """
        Check if user has available credits in Dodo.
        Auto-initializes free user if no Dodo customer exists.

        When ``BILLING_ENABLED`` is unset/falsy (local dev), skip silently.

        Raises HTTPException with 402 status if credits are exhausted.
        """
        from app.modules.billing.subscription_service import _billing_enabled

        if not _billing_enabled():
            return True

        # Get or create Dodo customer (auto-initializes free user)
        dodo_customer_id = (
            await billing_subscription_service.get_or_create_dodo_customer_id(user_id)
        )

        if not dodo_customer_id:
            # Could not get or create Dodo customer - allow usage but log warning
            logger.warning(
                f"Could not get or create Dodo customer for user {user_id}, allowing usage"
            )
            return True

        # Get real-time credit balance from Dodo
        credit_balance = await billing_subscription_service.get_credit_balance(user_id)

        credits_available = credit_balance.get("credits_available", 0)
        plan_type = credit_balance.get("plan_type", "free")
        credits_total = credit_balance.get(
            "credits_total", 50 if plan_type == "free" else 500
        )

        if credits_available <= 0:
            raise HTTPException(
                status_code=402,
                detail=f"Credit limit reached. You've used {credits_total}/{credits_total} credits on your {plan_type} plan. Please upgrade to continue.",
            )

        return True

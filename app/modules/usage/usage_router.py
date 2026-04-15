from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_db
from app.modules.auth.auth_service import AuthService
from app.modules.usage.usage_controller import UsageController
from app.modules.usage.usage_schema import UsageResponse
from app.modules.billing.subscription_service import billing_subscription_service

router = APIRouter()


class UsageAPI:
    @staticmethod
    @router.get("/usage", response_model=UsageResponse)
    async def get_usage(
        start_date: datetime,
        end_date: datetime,
        user=Depends(AuthService.check_auth),
        async_db: AsyncSession = Depends(get_async_db),
    ):
        user_id = user["user_id"]
        return await UsageController.get_user_usage(
            async_db, start_date, end_date, user_id
        )

    @staticmethod
    @router.get("/credit-balance")
    async def get_credit_balance(
        user_id: str,
        user=Depends(AuthService.check_auth),
    ):
        """
        Get real-time credit balance from Dodo.
        Auto-initializes a free Dodo subscription if none exists.
        Returns: { plan_type, credits_total, credits_used, credits_available, dodo_customer_id }
        """
        # Verify the user is requesting their own credit balance
        if user["user_id"] != user_id:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Cannot access other user's credit balance")

        # Auto-initialize free user if needed, then get credit balance
        dodo_customer_id = await billing_subscription_service.get_or_create_dodo_customer_id(user_id)

        # Get credit balance from Dodo
        balance = await billing_subscription_service.get_credit_balance(user_id)

        # Add dodo_customer_id to response
        balance["dodo_customer_id"] = dodo_customer_id

        return balance

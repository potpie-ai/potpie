from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.utils.APIRouter import APIRouter
from app.modules.users.user_model import User

from .subscription_service import SubscriptionService

router = APIRouter()


@router.get("/subscription/info")
async def get_subscription_info(
    user = Depends(AuthService.check_auth),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the current user's subscription plan type and end date.
    """
    subscription_info = await SubscriptionService.get_user_subscription_info(user["user_id"], db)
    
    if not subscription_info:
       return {"plan_type": "free", "end_date": None}
    
    return {
        "plan_type": subscription_info.type,
        "end_date": subscription_info.next_billing_date
    }


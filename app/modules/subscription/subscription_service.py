from sqlalchemy.orm import Session
from app.modules.subscription.subscription_model import Subscription

class SubscriptionService:
    @staticmethod
    async def get_user_subscription_info(user_id: int, db: Session):
        """
        Retrieve the subscription information for a given user.
        """
        query = db.query(Subscription).where(Subscription.user_id == user_id)
        subscription = query.first()
        return subscription


"""
Subscription Service for Dodo Payments

This service interacts with stripe-potpie to get subscription information.
"""

import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

STRIPE_POTPIE_URL = os.getenv("STRIPE_POTPIE_URL", "http://localhost:8003")


class BillingSubscriptionService:
    """
    Service for getting subscription info from stripe-potpie.
    """

    @staticmethod
    async def get_dodo_customer_id(user_id: str) -> Optional[str]:
        """
        Get the Dodo customer ID for a user.

        Args:
            user_id: The user ID

        Returns:
            str: The Dodo customer ID or None if not found
        """
        logger.info(f"[billing] Checking subscription status for user: {user_id}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{STRIPE_POTPIE_URL}/dodo/subscription-status",
                    params={"user_id": user_id},
                    timeout=5.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    dodo_customer_id = data.get("dodo_customer_id")
                    subscription_id = data.get("subscription_id")
                    plan_type = data.get("plan_type")

                    if dodo_customer_id:
                        logger.info(
                            f"[billing] User EXISTS: user_id={user_id}, "
                            f"dodo_customer_id={dodo_customer_id}, "
                            f"subscription_id={subscription_id}, "
                            f"plan_type={plan_type}"
                        )
                    else:
                        logger.info(
                            f"[billing] User NOT FOUND in subscriptions table: user_id={user_id}, "
                            f"plan_type={plan_type}, will initialize free user"
                        )
                    return dodo_customer_id
                else:
                    logger.warning(
                        f"[billing] Failed to get subscription status: "
                        f"status_code={response.status_code}, user_id={user_id}"
                    )
                    return None

        except Exception as e:
            logger.error(f"[billing] Error getting dodo_customer_id for user {user_id}: {e}")
            return None

    @staticmethod
    async def get_or_create_dodo_customer_id(user_id: str) -> Optional[str]:
        """
        Get the Dodo customer ID for a user, or create one if it doesn't exist.
        This auto-initializes free users like potpie-workflows does.

        Args:
            user_id: The user ID

        Returns:
            str: The Dodo customer ID or None if creation failed
        """
        # First try to get existing customer ID
        dodo_customer_id = await BillingSubscriptionService.get_dodo_customer_id(user_id)
        if dodo_customer_id:
            return dodo_customer_id

        # No customer ID found, initialize as free user
        logger.info(f"[billing] No dodo_customer_id for user {user_id} - initializing free user")
        return await BillingSubscriptionService._initialize_free_user(user_id)

    @staticmethod
    async def _initialize_free_user(user_id: str) -> Optional[str]:
        """
        Initialize a free user in Dodo by calling stripe-potpie.
        Returns the dodo_customer_id if successful, None otherwise.
        """
        logger.info(f"[billing] Initializing free user in Dodo: {user_id}")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{STRIPE_POTPIE_URL}/dodo/initialize-free-user",
                    json={"user_id": user_id},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    dodo_customer_id = data.get("dodo_customer_id")
                    logger.info(
                        f"[billing] Free user initialized: {user_id} -> {dodo_customer_id} "
                        f"(existing={data.get('existing', False)})"
                    )
                    return dodo_customer_id
                else:
                    logger.warning(
                        f"[billing] initialize-free-user returned {resp.status_code}: {resp.text}"
                    )
        except Exception as e:
            logger.error(f"[billing] Failed to initialize free user {user_id}: {e}")
        return None

    @staticmethod
    async def get_subscription_status(user_id: str) -> dict:
        """
        Get the subscription status for a user from Dodo.
        Uses the credit-balance endpoint to get real-time credits.

        Args:
            user_id: The user ID

        Returns:
            dict: Subscription status with credits from Dodo
        """
        try:
            async with httpx.AsyncClient() as client:
                # Use the new credit-balance endpoint for real-time credits
                response = await client.get(
                    f"{STRIPE_POTPIE_URL}/dodo/credit-balance",
                    params={"user_id": user_id},
                    timeout=5.0,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to get credit balance: {response.status_code}")
                    return {
                        "plan_type": "free",
                        "credits_total": 50,
                        "credits_used": 0,
                        "credits_available": 50,
                    }

        except Exception as e:
            logger.error(f"Error getting subscription status: {e}")
            return {
                "plan_type": "free",
                "credits_total": 50,
                "credits_used": 0,
                "credits_available": 50,
            }

    @staticmethod
    async def get_credit_balance(user_id: str) -> dict:
        """
        Get real-time credit balance from Dodo.

        Returns:
            dict: { plan_type, credits_total, credits_used, credits_available }
        """
        return await BillingSubscriptionService.get_subscription_status(user_id)


# Singleton instance
billing_subscription_service = BillingSubscriptionService()

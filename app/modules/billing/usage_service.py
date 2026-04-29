"""
Usage Reporting Service for Dodo Payments

This service reports message usage to stripe-potpie, which then forwards to Dodo.

Like :mod:`subscription_service`, this is **opt-in** — disabled when
``BILLING_ENABLED`` is unset/falsy. Local dev runs without the sidecar.
"""

import os
import logging

import httpx

from app.modules.billing.subscription_service import _billing_enabled

logger = logging.getLogger(__name__)

# stripe-potpie service URL
STRIPE_POTPIE_URL = os.getenv("STRIPE_POTPIE_URL", "http://localhost:8003")


class UsageReportingService:
    """
    Service for reporting usage to Dodo via stripe-potpie.
    """

    @staticmethod
    async def report_message_usage(
        user_id: str,
        dodo_customer_id: str,
        conversation_id: str,
    ) -> dict:
        """
        Report a message usage event to Dodo.

        Args:
            user_id: The user ID
            dodo_customer_id: The Dodo customer ID (from subscription)
            conversation_id: The conversation ID

        Returns:
            dict: Status of the usage reporting
        """
        if not _billing_enabled():
            return {"status": "skipped", "reason": "billing_disabled"}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{STRIPE_POTPIE_URL}/dodo/usage/report",
                    json={
                        "user_id": user_id,
                        "dodo_customer_id": dodo_customer_id,
                        "event_type": "message",
                        "resource_id": conversation_id,
                        "resource_type": "conversation",
                    },
                    timeout=5.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Usage reported successfully: {result}")
                    return result
                else:
                    logger.error(
                        f"Failed to report usage: {response.status_code} - {response.text}"
                    )
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}",
                    }

        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to stripe-potpie: {e}")
            return {"status": "error", "error": "stripe-potpie service unavailable"}
        except Exception as e:
            logger.error(f"Error reporting usage: {e}")
            return {"status": "error", "error": str(e)}

    @staticmethod
    def report_message_usage_sync(
        user_id: str,
        dodo_customer_id: str,
        conversation_id: str,
    ) -> dict:
        """
        Synchronous version of report_message_usage.
        Use this when you can't use async/await (e.g. sync context, Celery tasks).
        Must NOT be called from within a running event loop — use report_message_usage instead.
        """
        if not _billing_enabled():
            return {"status": "skipped", "reason": "billing_disabled"}
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context — caller should use report_message_usage directly
                logger.warning(
                    "report_message_usage_sync called from a running event loop. "
                    "Use report_message_usage (async) instead."
                )
                return {"status": "skipped", "reason": "called from async context"}
            return loop.run_until_complete(
                UsageReportingService.report_message_usage(
                    user_id, dodo_customer_id, conversation_id
                )
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    UsageReportingService.report_message_usage(
                        user_id, dodo_customer_id, conversation_id
                    )
                )
            finally:
                loop.close()


# Singleton instance
usage_reporting_service = UsageReportingService()

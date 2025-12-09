"""
HITL Timeout Monitor Service.

This module provides a background service to monitor and handle
expired HITL requests, executing timeout actions as configured.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

from app.core.executions.executions import ExecutionLogStore
from app.core.executions.hitl import HITLRequest, HITLRequestStatus, HITLResponse
from app.core.executions.hitl_response_processor import HITLResponseProcessor
from app.core.workflows import WorkflowsStore
from app.core.executions.processors import WorkflowOrchestrator
from app.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class HITLTimeoutMonitor:
    """Service to monitor and handle expired HITL requests."""

    def __init__(
        self,
        log_store: ExecutionLogStore,
        workflows_store: WorkflowsStore,
        response_processor: HITLResponseProcessor,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize the HITL Timeout Monitor.

        Args:
            log_store: Execution log store
            workflows_store: Workflows store
            response_processor: HITL response processor
            check_interval_seconds: How often to check for expired requests (default: 60 seconds)
        """
        self.log_store = log_store
        self.workflows_store = workflows_store
        self.response_processor = response_processor
        self.check_interval = check_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the timeout monitoring service."""
        if self._running:
            logger.warning("Timeout monitor is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("HITL Timeout Monitor started")

    async def stop(self):
        """Stop the timeout monitoring service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("HITL Timeout Monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_expired_requests()
            except Exception as e:
                logger.error(f"Error in timeout monitor loop: {e}")
                logger.exception("Full traceback:")

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    async def _check_expired_requests(self):
        """Check for expired HITL requests and handle them."""
        try:
            # Get all pending requests
            pending_requests = await self.log_store.list_pending_hitl_requests()

            current_time = utc_now()
            expired_requests: List[HITLRequest] = []

            # Find expired requests
            for request in pending_requests:
                if request.timeout_at < current_time:
                    expired_requests.append(request)

            # Process expired requests
            for request in expired_requests:
                await self._handle_expired_request(request)

        except Exception as e:
            logger.error(f"Error checking expired requests: {e}")
            logger.exception("Full traceback:")

    async def _handle_expired_request(self, request: HITLRequest):
        """Handle a single expired HITL request."""
        try:
            logger.info(
                f"Handling expired HITL request {request.request_id} for node {request.node_id}"
            )

            # Update request status to expired
            await self.log_store.update_hitl_request_status(
                request.request_id, HITLRequestStatus.EXPIRED.value
            )

            # Determine timeout action
            timeout_action = request.timeout_action or "fail"

            # Create a timeout response based on the action
            if request.node_type.value == "approval":
                # For approval nodes, timeout_action can be "approve" or "reject"
                if timeout_action == "approve":
                    approved = True
                elif timeout_action == "reject":
                    approved = False
                else:
                    # Default to reject if unknown action
                    approved = False

                response_data = {
                    "approved": approved,
                    "timeout": True,
                }
            else:
                # For input nodes, timeout_action is typically "fail"
                # We'll create an empty response that indicates timeout
                response_data = {
                    "timeout": True,
                    "error": "Request expired before input was provided",
                }

            # Create timeout response
            timeout_response = HITLResponse(
                request_id=request.request_id,
                execution_id=request.execution_id,
                node_id=request.node_id,
                user_id="system",  # System-generated timeout response
                response_data=response_data,
                comment=f"Request expired at {request.timeout_at.isoformat()}. Timeout action: {timeout_action}",
                channel=request.channel,
            )

            # Store the timeout response
            await self.log_store.create_hitl_response(timeout_response)

            # Process the timeout response to resume workflow
            result = await self.response_processor.process_response(
                timeout_response, "system"
            )

            if result.get("success"):
                logger.info(
                    f"Successfully processed timeout for request {request.request_id}"
                )
            else:
                logger.error(
                    f"Failed to process timeout for request {request.request_id}: {result.get('error')}"
                )

        except Exception as e:
            logger.error(
                f"Error handling expired request {request.request_id}: {e}"
            )
            logger.exception("Full traceback:")

    async def check_and_handle_expired(self, request_id: str) -> bool:
        """
        Manually check and handle a specific expired request.

        Args:
            request_id: The request ID to check

        Returns:
            True if request was expired and handled, False otherwise
        """
        try:
            request = await self.log_store.get_hitl_request_by_id(request_id)
            if not request:
                return False

            if request.status != HITLRequestStatus.PENDING:
                return False

            if request.timeout_at < utc_now():
                await self._handle_expired_request(request)
                return True

            return False
        except Exception as e:
            logger.error(f"Error checking expired request {request_id}: {e}")
            return False


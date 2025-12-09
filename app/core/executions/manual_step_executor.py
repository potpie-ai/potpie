"""
Manual Step Executor for HITL nodes.

This module implements execution logic for manual step nodes (ApprovalNode and InputNode),
creating HITL requests and managing the waiting state.
"""

import logging
import uuid
from datetime import timedelta
from typing import Optional

from app.core.executions.state import (
    NodeExecutionContext,
    NodeExecutionResult,
    NodeExecutionStatus,
)
from app.core.executions.hitl import HITLRequest, HITLNodeType, HITLRequestStatus
from app.core.executions.hitl_channel import HITLChannel
from app.core.executions.executions import ExecutionLogStore
from app.core.nodes.manual_steps import ApprovalNode, InputNode
from app.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class ManualStepExecutor:
    """Executor for manual step nodes (Approval and Input)."""

    def __init__(
        self,
        log_store: ExecutionLogStore,
        email_channel: Optional[HITLChannel] = None,
        webui_channel: Optional[HITLChannel] = None,
    ):
        """
        Initialize the Manual Step Executor.

        Args:
            log_store: Execution log store for storing HITL requests
            email_channel: Email channel for sending email notifications
            webui_channel: WebUI channel for displaying requests in the app
        """
        self.log_store = log_store
        self.email_channel = email_channel
        self.webui_channel = webui_channel
    
    def _get_channel(self, channel_name: Optional[str]) -> Optional[HITLChannel]:
        """
        Get the appropriate channel based on channel name.
        
        Args:
            channel_name: Channel name from node data ('email' or 'app')
        
        Returns:
            The appropriate HITLChannel instance, or None
        """
        if channel_name == "email":
            return self.email_channel
        elif channel_name == "app":
            return self.webui_channel
        else:
            # Default to webui if channel is not specified or invalid
            return self.webui_channel

    async def execute(
        self, node: ApprovalNode | InputNode, ctx: NodeExecutionContext
    ) -> NodeExecutionResult:
        """
        Execute a manual step node by creating a HITL request.

        Args:
            node: The manual step node (ApprovalNode or InputNode)
            ctx: Execution context

        Returns:
            NodeExecutionResult with WAITING_FOR_APPROVAL status
        """
        execution_id = ctx.execution_id
        node_id = node.id
        iteration = ctx.current_iteration

        try:
            logger.info(
                f"[execution:{execution_id}] 👤 Executing manual step node: {node_id} (type: {node.type})"
            )

            # Generate unique request ID
            request_id = f"hitl_{uuid.uuid4().hex[:16]}"

            # Calculate timeout
            timeout_hours = 24  # Default
            if isinstance(node, ApprovalNode):
                timeout_hours = node.data.timeout_hours or 24
            elif isinstance(node, InputNode):
                timeout_hours = node.data.timeout_hours or 24

            timeout_at = utc_now() + timedelta(hours=timeout_hours)

            # Determine node type and create appropriate request
            if isinstance(node, ApprovalNode):
                node_type = HITLNodeType.APPROVAL
                message = node.data.approval_message or "Please approve or reject this request"
                approvers = node.data.approvers or []
                assignee = None
                fields = None
                timeout_action = "reject"  # Default timeout action for approval

            elif isinstance(node, InputNode):
                node_type = HITLNodeType.INPUT
                message = "Please provide the required input"
                approvers = None
                assignee = node.data.assignee
                fields = node.data.input_fields or []
                timeout_action = "fail"  # Default timeout action for input

            else:
                logger.error(
                    f"[execution:{execution_id}] ❌ Unknown manual step node type: {type(node)}"
                )
                return NodeExecutionResult(
                    status=NodeExecutionStatus.FAILED,
                    output=f"Unknown manual step node type: {type(node)}",
                )

            # Get channel from node data
            channel_name = None
            if isinstance(node, ApprovalNode):
                channel_name = node.data.channel if hasattr(node.data, 'channel') else "app"
            elif isinstance(node, InputNode):
                channel_name = node.data.channel if hasattr(node.data, 'channel') else "app"
            
            # Get the appropriate channel
            channel = self._get_channel(channel_name)
            channel_id = channel.channel_id if channel else "web"

            # Create HITL request
            hitl_request = HITLRequest(
                request_id=request_id,
                execution_id=execution_id,
                node_id=node_id,
                iteration=iteration,
                node_type=node_type,
                message=message,
                fields=fields,
                timeout_at=timeout_at,
                channel=channel_id,
                approvers=approvers,
                assignee=assignee,
                timeout_action=timeout_action,
            )

            # Store HITL request
            await self.log_store.create_hitl_request(hitl_request)
            
            # Log audit event for HITL request creation
            from app.core.executions.executions import NodeExecutionLog
            await self.log_store.append_log(
                execution_id,
                node_id,
                iteration,
                NodeExecutionLog(
                    status=NodeExecutionStatus.WAITING_FOR_APPROVAL,
                    timestamp=utc_now(),
                    details=f"HITL request created: {request_id}, timeout: {timeout_at.isoformat()}, channel: {hitl_request.channel}",
                ),
            )
            
            logger.info(
                f"[execution:{execution_id}] ✅ Created HITL request {request_id} for node {node_id} (channel: {channel_name})"
            )

            # Send notification through channel if available
            if channel:
                try:
                    sent = await channel.send_request(hitl_request)
                    if sent:
                        logger.info(
                            f"[execution:{execution_id}] 📤 Sent HITL request {request_id} via {channel.channel_id} channel"
                        )
                    else:
                        logger.warning(
                            f"[execution:{execution_id}] ⚠️ Failed to send HITL request {request_id} via {channel.channel_id} channel"
                        )
                except Exception as e:
                    logger.error(
                        f"[execution:{execution_id}] ❌ Error sending HITL request via channel: {e}"
                    )
                    # Don't fail the execution if channel fails - request is still stored
            else:
                logger.warning(
                    f"[execution:{execution_id}] ⚠️ No channel available for HITL request {request_id}"
                )

            # Return waiting status
            return NodeExecutionResult(
                status=NodeExecutionStatus.WAITING_FOR_APPROVAL,
                output={
                    "request_id": request_id,
                    "message": "Waiting for human input",
                    "timeout_at": timeout_at.isoformat(),
                },
            )

        except Exception as e:
            logger.error(
                f"[execution:{execution_id}] 💥 Error executing manual step node {node_id}: {e}"
            )
            logger.exception(f"[execution:{execution_id}] Full traceback:")
            return NodeExecutionResult(
                status=NodeExecutionStatus.FAILED,
                output=f"Error executing manual step node: {str(e)}",
            )


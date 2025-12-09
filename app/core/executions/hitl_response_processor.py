"""
HITL Response Processor for handling user responses and resuming workflow execution.

This module processes HITL responses, validates them, updates node execution status,
and resumes workflow execution by queuing the next nodes.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.executions.hitl import (
    HITLRequest,
    HITLResponse,
    HITLRequestStatus,
    HITLNodeType,
)
from app.core.executions.executions import (
    ExecutionLogStore,
    NodeExecutionLog,
)
from app.core.executions.state import (
    NodeExecutionContext,
    NodeExecutionResult,
    NodeExecutionStatus,
)
from app.core.executions.processors import WorkflowOrchestrator
from app.core.workflows import WorkflowsStore
from app.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class HITLResponseProcessor:
    """Processor for handling HITL responses and resuming workflow execution."""

    def __init__(
        self,
        log_store: ExecutionLogStore,
        workflows_store: WorkflowsStore,
        orchestrator: WorkflowOrchestrator,
    ):
        """
        Initialize the HITL Response Processor.

        Args:
            log_store: Execution log store
            workflows_store: Workflows store for retrieving workflow definitions
            orchestrator: Workflow orchestrator for resuming execution
        """
        self.log_store = log_store
        self.workflows_store = workflows_store
        self.orchestrator = orchestrator

    async def process_response(
        self, response: HITLResponse, user_id: str
    ) -> Dict[str, Any]:
        """
        Process a HITL response and resume workflow execution.

        Args:
            response: The HITL response to process
            user_id: The user ID submitting the response

        Returns:
            Dict with processing result
        """
        try:
            # Get the HITL request
            request = await self.log_store.get_hitl_request_by_id(
                response.request_id
            )
            if not request:
                return {
                    "success": False,
                    "error": f"HITL request {response.request_id} not found",
                }

            # Validate request is still pending
            if request.status != HITLRequestStatus.PENDING:
                return {
                    "success": False,
                    "error": f"HITL request {response.request_id} is not pending (status: {request.status})",
                }

            # Check if request has expired
            if request.timeout_at < utc_now():
                await self.log_store.update_hitl_request_status(
                    response.request_id, HITLRequestStatus.EXPIRED.value
                )
                return {
                    "success": False,
                    "error": f"HITL request {response.request_id} has expired",
                }

            # Validate user permissions
            permission_error = await self._validate_permissions(request, user_id)
            if permission_error:
                return {"success": False, "error": permission_error}

            # Validate response format (skip for system/timeout responses)
            if user_id != "system":
                validation_error = await self._validate_response(request, response)
                if validation_error:
                    return {"success": False, "error": validation_error}

            # Check for duplicate responses
            existing_response = await self.log_store.get_hitl_response(
                response.request_id
            )
            if existing_response:
                return {
                    "success": False,
                    "error": f"Response already submitted for request {response.request_id}",
                }

            # Store the response
            await self.log_store.create_hitl_response(response)

            # Log audit event
            await self.log_store.append_log(
                response.execution_id,
                response.node_id,
                request.iteration,
                NodeExecutionLog(
                    status=NodeExecutionStatus.COMPLETED,
                    timestamp=utc_now(),
                    details=f"HITL response submitted by user {user_id}: {response.response_data}",
                ),
            )

            # Update request status
            await self.log_store.update_hitl_request_status(
                response.request_id, HITLRequestStatus.RESPONDED.value
            )

            # Mark node execution as completed
            await self.log_store.append_log(
                response.execution_id,
                response.node_id,
                request.iteration,
                NodeExecutionLog(
                    status=NodeExecutionStatus.COMPLETED,
                    timestamp=utc_now(),
                    details=f"HITL response received: {response.response_data}",
                ),
            )

            # Resume workflow execution
            resume_result = await self._resume_workflow_execution(
                request, response
            )

            if resume_result["success"]:
                logger.info(
                    f"[execution:{response.execution_id}] ✅ HITL response processed and workflow resumed"
                )
                return {
                    "success": True,
                    "message": "Response processed and workflow resumed",
                    "queued_nodes": resume_result.get("queued_nodes", []),
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to resume workflow: {resume_result.get('error')}",
                }

        except Exception as e:
            logger.error(f"Error processing HITL response: {e}")
            logger.exception("Full traceback:")
            return {"success": False, "error": str(e)}

    async def _validate_permissions(
        self, request: HITLRequest, user_id: str
    ) -> Optional[str]:
        """Validate user has permission to respond to this request."""
        # System user can always respond (for timeout handling)
        if user_id == "system":
            return None
            
        if request.node_type == HITLNodeType.APPROVAL:
            if not request.approvers or user_id not in request.approvers:
                return f"User {user_id} is not authorized to approve this request"
        elif request.node_type == HITLNodeType.INPUT:
            if not request.assignee or request.assignee != user_id:
                return f"User {user_id} is not assigned to provide input for this request"
        return None

    async def _validate_response(
        self, request: HITLRequest, response: HITLResponse
    ) -> Optional[str]:
        """Validate response format matches node configuration."""
        if request.node_type == HITLNodeType.APPROVAL:
            # Approval response should be boolean
            if "approved" not in response.response_data:
                return "Approval response must include 'approved' field"
            approved = response.response_data["approved"]
            if not isinstance(approved, bool):
                return "Approval response 'approved' field must be boolean"
        elif request.node_type == HITLNodeType.INPUT:
            # Input response should match input_fields
            if not request.fields:
                return "Input node has no fields defined"
            # Validate required fields
            for field in request.fields:
                field_name = field.get("name")
                required = field.get("required", False)
                if required and field_name not in response.response_data:
                    return f"Required field '{field_name}' is missing"
        return None

    async def _resume_workflow_execution(
        self, request: HITLRequest, response: HITLResponse
    ) -> Dict[str, Any]:
        """Resume workflow execution by queuing next nodes."""
        try:
            # Get workflow execution
            workflow_execution = await self.log_store.get_wf_execution_by_id(
                request.execution_id
            )

            # Get workflow definition
            workflow = await self.workflows_store.get_workflow(
                workflow_execution.wf_id
            )

            # Find the node that was waiting
            node = workflow.graph.nodes.get(request.node_id)
            if not node:
                return {
                    "success": False,
                    "error": f"Node {request.node_id} not found in workflow",
                }

            # Create a result object from the response
            # For approval nodes, output is the approval decision
            # For input nodes, output is the input data
            output = response.response_data
            if request.node_type == HITLNodeType.APPROVAL:
                output = {
                    "approved": response.response_data.get("approved", False),
                    "comment": response.comment,
                }

            result = NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=output,
                execution_variables={
                    "hitl_response": str(output),
                    "hitl_user_id": response.user_id,
                },
            )

            # Get next nodes using the node's routing logic
            next_nodes = node.get_next_nodes(
                result, workflow.graph.adjacency_list
            )

            if not next_nodes:
                logger.info(
                    f"[execution:{request.execution_id}] No next nodes to queue for node {request.node_id}"
                )
                # Update workflow status
                await self.orchestrator.infer_and_update_workflow_status(
                    request.execution_id
                )
                self.orchestrator.mark_execution_complete(request.execution_id)
                return {"success": True, "queued_nodes": []}

            # Queue next nodes
            queued_nodes = []
            for next_node_id in next_nodes:
                next_node = workflow.graph.nodes.get(next_node_id)
                if not next_node:
                    logger.warning(
                        f"[execution:{request.execution_id}] Next node {next_node_id} not found"
                    )
                    continue

                # Create execution context for next node
                ctx = NodeExecutionContext(
                    queued_at=utc_now(),
                    workflow_snapshot=workflow,
                    event=workflow_execution.event,
                    execution_variables=result.execution_variables or {},
                    execution_id=request.execution_id,
                    previous_node_result=str(result.output),
                    current_iteration=request.iteration + 1,
                    current_node=next_node,
                    max_iterations=15,  # Default max iterations
                    predecessor_node_id=request.node_id,
                )

                await self.orchestrator.queue_node_for_execution(ctx)
                queued_nodes.append(next_node_id)

            logger.info(
                f"[execution:{request.execution_id}] ✅ Queued {len(queued_nodes)} next nodes after HITL response"
            )

            return {"success": True, "queued_nodes": queued_nodes}

        except Exception as e:
            logger.error(f"Error resuming workflow execution: {e}")
            logger.exception("Full traceback:")
            return {"success": False, "error": str(e)}


from datetime import datetime
import logging
from typing import Optional, List
from app.utils.datetime_utils import utc_now
from app.core.executions.event import Event
from app.core.executions.executions import (
    ExecutionLogStore,
    NodeExecutionLog,
    NodeExecution,
    WorkflowExecution,
    WorkflowExecutionStatus,
)
from app.core.executions.agent import AgentQueryExecutor
from app.core.executions.state import (
    NodeExecutionContext,
    NodeExecutionResult,
    NodeExecutionStatus,
    DEFAULT_MAX_ITERATIONS,
)
from app.core.executions.task import TaskQueue
from app.core.nodes.agents import CustomAgent, CustomAgentExecutor
from app.core.nodes.agents.action_agent import ActionAgent, ActionAgentExecutor
from app.core.nodes.flow_control import (
    CollectNode,
    ConditionalNode,
    SelectorNode,
)
from app.core.nodes.flow_control.conditional_executor import ConditionalExecutor
from app.core.nodes.manual_steps import ManualStep
from app.core.executions.manual_step_executor import ManualStepExecutor
from app.core.executions.webui_channel import WebUIChannel
from app.core.nodes import WorkflowNode
from app.core.workflows import WorkflowsStore
from app.core.nodes.base import NodeCategory
from app.core.trigger_hashes import TriggerHashStore
from app.core.nodes.triggers.executors import get_trigger_executor_registry
from app.core.intelligence import LLMInterface


logger = logging.getLogger(__name__)


def infer_workflow_status_from_node_executions(
    node_executions: List[NodeExecution],
) -> WorkflowExecutionStatus:
    """
    Infer workflow execution status from the status of its node executions.

    Logic:
    - If any node is FAILED or INTERRUPTED, workflow is FAILED
    - If any node is WAITING_FOR_APPROVAL, workflow is WAITING_FOR_APPROVAL
    - If trigger node is SKIPPED, workflow is SKIPPED
    - If all nodes are COMPLETED, workflow is COMPLETED
    - If workflow has nodes but none are in final states, workflow is RUNNING
    - If no nodes have been executed, workflow is PENDING
    """
    if not node_executions:
        return WorkflowExecutionStatus.PENDING

    # Check for failed or interrupted nodes
    for node_exec in node_executions:
        if node_exec.status in [
            NodeExecutionStatus.FAILED,
            NodeExecutionStatus.INTERRUPTED,
        ]:
            return WorkflowExecutionStatus.FAILED

    # Check for waiting nodes
    for node_exec in node_executions:
        if node_exec.status == NodeExecutionStatus.WAITING_FOR_APPROVAL:
            return WorkflowExecutionStatus.WAITING_FOR_APPROVAL

    # Check if trigger node is skipped
    for node_exec in node_executions:
        if node_exec.status == NodeExecutionStatus.SKIPPED:
            # Check if this is a trigger node by looking at the node_id
            # Trigger nodes typically have names that start with "trigger" or contain "trigger"
            if "trigger" in node_exec.node_id.lower():
                return WorkflowExecutionStatus.SKIPPED

    # Check if all nodes are completed
    all_completed = all(
        node_exec.status == NodeExecutionStatus.COMPLETED
        for node_exec in node_executions
    )
    if all_completed:
        return WorkflowExecutionStatus.COMPLETED

    # If we have nodes but none are in final states, workflow is still running
    return WorkflowExecutionStatus.RUNNING


class NodeExecutor:
    def __init__(
        self,
        agent_executor: AgentQueryExecutor,
        log_store: ExecutionLogStore,
        orchestrator: Optional[
            "WorkflowOrchestrator"
        ] = None,  # Reference to orchestrator
        llm_interface: Optional[
            LLMInterface
        ] = None,  # LLM interface for conditional evaluation
    ):
        self.agent_executor = agent_executor
        self.log_store = log_store
        self.orchestrator = orchestrator
        self.trigger_executor_registry = get_trigger_executor_registry()

        self.custom_agent_executor = CustomAgentExecutor(agent_executor)
        self.action_agent_executor = ActionAgentExecutor(agent_executor)

        # Initialize conditional executor if LLM interface is provided
        if llm_interface:
            self.conditional_executor = ConditionalExecutor(llm_interface)
        else:
            self.conditional_executor = None

        # Initialize manual step executor with both channels
        # Nodes can choose which channel to use via their configuration
        from app.core.executions.email_channel import EmailChannel
        import os
        
        # Initialize email channel if SMTP is configured
        email_channel = None
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if smtp_user and smtp_password:
            email_channel = EmailChannel()
            logger.info("EmailChannel initialized for HITL notifications")
        else:
            logger.info("SMTP not configured, EmailChannel will not be available")
        
        # Always initialize WebUIChannel (for app-based notifications)
        webui_channel = WebUIChannel()
        
        # Initialize executor with both channels
        self.manual_step_executor = ManualStepExecutor(
            log_store,
            email_channel=email_channel,
            webui_channel=webui_channel
        )

    def set_orchestrator(self, orchestrator: "WorkflowOrchestrator"):
        """Set the orchestrator reference to avoid circular dependency in constructor."""
        self.orchestrator = orchestrator

    async def execute_node(self, ctx: NodeExecutionContext) -> NodeExecutionResult:
        """Execute a single node and return the result."""
        try:
            node = ctx.current_node
            execution_id = ctx.execution_id

            # TRIGGER nodes
            if node.category == NodeCategory.TRIGGER:
                logger.info(
                    f"[execution:{execution_id}] � Executing trigger node: {node.id}"
                )
                # Use the new trigger executor system

                executor = self.trigger_executor_registry.get_executor(node)
                return await executor.execute(node, ctx.event)

            # AGENT nodes
            elif node.category == NodeCategory.AGENT:
                if isinstance(node, CustomAgent):
                    logger.info(
                        f"[execution:{execution_id}] 🤖 Executing CustomAgent node: {node.id}"
                    )
                    return await self.custom_agent_executor.execute(node, ctx)
                elif isinstance(node, ActionAgent):
                    logger.info(
                        f"[execution:{execution_id}] ⚡ Executing ActionAgent node: {node.id}"
                    )
                    return await self.action_agent_executor.execute(node, ctx)
                # Add more agent types here as needed

            # FLOW CONTROL nodes
            elif node.category == NodeCategory.FLOW_CONTROL:
                if isinstance(node, ConditionalNode):
                    logger.info(
                        f"[execution:{execution_id}] � Executing conditional node: {node.id}"
                    )
                    if self.conditional_executor:
                        logger.info(
                            f"[execution:{execution_id}] 🧠 Using LLM for conditional evaluation"
                        )
                        return await self.conditional_executor.execute(node, ctx)
                    else:
                        logger.error(
                            f"[execution:{execution_id}] ❌ LLM interface not available for conditional node"
                        )
                        return NodeExecutionResult(
                            status=NodeExecutionStatus.FAILED,
                            output="LLM interface not available for conditional evaluation",
                        )
                elif isinstance(node, CollectNode):
                    logger.info(
                        f"[execution:{execution_id}] 📦 Executing collect node: {node.id}"
                    )
                    # TODO: Implement collect node execution
                    return NodeExecutionResult(
                        status=NodeExecutionStatus.COMPLETED,
                        output="Collect node execution not yet implemented",
                    )
                elif isinstance(node, SelectorNode):
                    logger.info(
                        f"[execution:{execution_id}] 🎯 Executing selector node: {node.id}"
                    )
                    # TODO: Implement selector node execution
                    return NodeExecutionResult(
                        status=NodeExecutionStatus.COMPLETED,
                        output="Selector node execution not yet implemented",
                    )
                else:
                    logger.error(
                        f"[execution:{execution_id}] ❌ Unknown flow control node type: {type(node)}"
                    )
                    return NodeExecutionResult(
                        status=NodeExecutionStatus.FAILED,
                        output=f"Unknown flow control node type: {type(node)}",
                    )

            # MANUAL STEP nodes
            elif node.category == NodeCategory.MANUAL_STEP:
                if isinstance(node, ManualStep):
                    logger.info(
                        f"[execution:{execution_id}] 👤 Executing manual step node: {node.id}"
                    )
                    return await self.manual_step_executor.execute(node, ctx)
                else:
                    logger.error(
                        f"[execution:{execution_id}] ❌ Unknown manual step node type: {type(node)}"
                    )
                    return NodeExecutionResult(
                        status=NodeExecutionStatus.FAILED,
                        output=f"Unknown manual step node type: {type(node)}",
                    )

            else:
                logger.error(
                    f"[execution:{execution_id}] ❌ Unknown node category: {node.category}"
                )
                return NodeExecutionResult(
                    status=NodeExecutionStatus.FAILED,
                    output=f"Unknown node category: {node.category}",
                )

        except Exception as e:
            logger.error(
                f"[execution:{ctx.execution_id}] 💥 Error executing node {ctx.current_node.id}: {e}"
            )
            logger.exception(f"[execution:{ctx.execution_id}] Full traceback:")
            return NodeExecutionResult(
                status=NodeExecutionStatus.FAILED,
                output=f"Error executing node: {str(e)}",
            )

        # This should never be reached, but ensures all code paths return a value
        logger.error(
            f"[execution:{ctx.execution_id}] 💥 Unexpected end of execute_node method"
        )
        return NodeExecutionResult(
            status=NodeExecutionStatus.FAILED,
            output="Unexpected end of execute_node method",
        )

    async def execute(self, ctx: NodeExecutionContext):
        """Execute a single node and return the result."""

        node_id = ctx.current_node.id
        execution_id = ctx.execution_id

        # Check iteration limit to prevent infinite loops
        if ctx.current_iteration >= ctx.max_iterations:
            logger.error(
                f"[execution:{execution_id}] � Maximum iterations ({ctx.max_iterations}) exceeded for node {node_id}. Stopping execution to prevent infinite loop."
            )
            await self.log_store.append_log(
                ctx.execution_id,
                node_id,
                ctx.current_iteration,
                NodeExecutionLog(
                    status=NodeExecutionStatus.INTERRUPTED,
                    timestamp=utc_now(),
                    details=f"Maximum iterations ({ctx.max_iterations}) exceeded. Stopping execution to prevent infinite loop.",
                ),
            )
            # Infer and update workflow execution status from node executions
            if self.orchestrator:
                await self.orchestrator.infer_and_update_workflow_status(execution_id)
                self.orchestrator.mark_execution_complete(execution_id)
            return

        logger.info(
            f"[execution:{execution_id}] 🚀 Starting execution of node {node_id} (type: {ctx.current_node.category}) - Iteration: {ctx.current_iteration}/{ctx.max_iterations}"
        )
        logger.debug(f"[execution:{execution_id}] Node details: {ctx.current_node}")

        # Change status to RUNNING when execution starts
        await self.log_store.append_log(
            ctx.execution_id,
            node_id,
            ctx.current_iteration,
            NodeExecutionLog(
                status=NodeExecutionStatus.RUNNING,
                timestamp=utc_now(),
                details="Starting execution of node",
            ),
            event=ctx.event,
            predecessor_node_id=ctx.predecessor_node_id,
        )

        logger.info(f"[execution:{execution_id}] ⚡ Executing node {node_id}...")
        res = await self.execute_node(ctx)
        logger.info(
            f"[execution:{execution_id}] ✅ Node {node_id} completed with status: {res.status}"
        )

        # Check if this is a skipped trigger node and update workflow status accordingly
        if (
            res.status == NodeExecutionStatus.SKIPPED
            and ctx.current_node.category == NodeCategory.TRIGGER
        ):
            logger.info(
                f"[execution:{execution_id}] ⏭️  Trigger node {node_id} was skipped, updating workflow status"
            )
            # Mark the node as skipped in logs
            await self.log_store.append_log(
                ctx.execution_id,
                node_id,
                ctx.current_iteration,
                NodeExecutionLog(
                    status=NodeExecutionStatus.SKIPPED,
                    timestamp=utc_now(),
                    details=str(res.output),
                ),
                predecessor_node_id=ctx.predecessor_node_id,
            )
            # Update workflow status to SKIPPED
            if self.orchestrator:
                await self.orchestrator.infer_and_update_workflow_status(execution_id)
                self.orchestrator.mark_execution_complete(execution_id)
            return

        try:
            # Let the node determine which nodes should be queued next
            next_nodes = ctx.current_node.get_next_nodes(
                res, ctx.workflow_snapshot.graph.adjacency_list
            )

            if not next_nodes:
                logger.info(
                    f"[execution:{execution_id}] � No next nodes to queue for node {node_id}, workflow execution complete"
                )
                # Mark current node as COMPLETED since no child nodes to queue
                await self.log_store.append_log(
                    ctx.execution_id,
                    node_id,
                    ctx.current_iteration,
                    NodeExecutionLog(
                        status=NodeExecutionStatus.COMPLETED,
                        timestamp=utc_now(),
                        details=(
                            str(res.output)
                            if not isinstance(res.output, str)
                            else res.output
                        ),
                    ),
                    predecessor_node_id=ctx.predecessor_node_id,
                )
                # Infer and update workflow execution status from node executions
                if self.orchestrator:
                    await self.orchestrator.infer_and_update_workflow_status(
                        execution_id
                    )
                    self.orchestrator.mark_execution_complete(execution_id)

            logger.info(
                f"[execution:{execution_id}] 📋 Queuing {len(next_nodes)} next nodes for node {node_id}"
            )
            for node in next_nodes:
                if self.orchestrator:
                    next_node = ctx.workflow_snapshot.graph.nodes[node]
                    logger.info(
                        f"[execution:{execution_id}] ➡️  Queuing next node: {node} (type: {next_node.category}) - Iteration: {ctx.current_iteration + 1}"
                    )
                    # Merge execution variables from the result with existing ones
                    next_execution_variables = ctx.execution_variables.copy()
                    if res.execution_variables:
                        next_execution_variables.update(res.execution_variables)
                        logger.info(
                            f"[execution:{execution_id}] 📝 Added execution variables from node {node_id}: {res.execution_variables}"
                        )

                    await self.orchestrator.queue_node_for_execution(
                        NodeExecutionContext(
                            queued_at=utc_now(),
                            workflow_snapshot=ctx.workflow_snapshot,
                            event=ctx.event,
                            execution_variables=next_execution_variables,
                            execution_id=ctx.execution_id,
                            previous_node_result=str(res.output),
                            current_iteration=ctx.current_iteration + 1,
                            current_node=next_node,
                            max_iterations=ctx.max_iterations,
                            predecessor_node_id=node_id,
                        )
                    )
                else:
                    logger.error(
                        f"[execution:{execution_id}] ❌ Orchestrator not set, cannot queue next nodes"
                    )

            # Mark current node as COMPLETED after all child nodes are queued
            await self.log_store.append_log(
                ctx.execution_id,
                node_id,
                ctx.current_iteration,
                NodeExecutionLog(
                    status=NodeExecutionStatus.COMPLETED,
                    timestamp=utc_now(),
                    details=(
                        str(res.output)
                        if not isinstance(res.output, str)
                        else res.output
                    ),
                ),
                predecessor_node_id=ctx.predecessor_node_id,
            )
        except Exception as e:
            logger.error(
                f"[execution:{ctx.execution_id}] 💥 Graph execution failed: {e}"
            )
            logger.exception(f"[execution:{ctx.execution_id}] Full traceback:")
            # Log the failed status
            await self.log_store.append_log(
                ctx.execution_id,
                node_id,
                ctx.current_iteration,
                NodeExecutionLog(
                    status=NodeExecutionStatus.FAILED,
                    timestamp=utc_now(),
                    details=f"Graph execution failed: {e}",
                ),
                predecessor_node_id=ctx.predecessor_node_id,
            )
            # Infer and update workflow execution status from node executions
            if self.orchestrator:
                await self.orchestrator.infer_and_update_workflow_status(execution_id)
                self.orchestrator.mark_execution_complete(execution_id)

    async def _update_workflow_execution_status(
        self, execution_id: str, status: WorkflowExecutionStatus
    ):
        """Update the workflow execution status in the log store."""
        try:
            if hasattr(self.log_store, "update_workflow_execution_status"):
                await self.log_store.update_workflow_execution_status(
                    execution_id, status
                )
                logger.info(
                    f"[execution:{execution_id}] 📊 Updated workflow execution status to: {status.value}"
                )
        except Exception as e:
            logger.error(
                f"[execution:{execution_id}] ❌ Failed to update workflow execution status: {e}"
            )


class WorkflowOrchestrator:
    """
    Orchestrator that manages both node execution and task queuing.
    This helps break the circular dependency.
    """

    def __init__(
        self,
        node_executor: NodeExecutor,
        task_queue: TaskQueue,
        log_store: ExecutionLogStore,
    ):
        self.node_executor = node_executor
        self.task_queue = task_queue
        self.log_store = log_store

    async def queue_node_for_execution(self, ctx: NodeExecutionContext) -> str:
        """Queue a node for asynchronous execution."""
        logger.info(
            f"[execution:{ctx.execution_id}] 📥 Queuing node {ctx.current_node.id} for execution"
        )

        # Log the node as PENDING when queuing
        await self.log_store.append_log(
            ctx.execution_id,
            ctx.current_node.id,
            ctx.current_iteration,
            NodeExecutionLog(
                status=NodeExecutionStatus.PENDING,
                timestamp=utc_now(),
                details="Node queued for execution",
            ),
            event=ctx.event,
            predecessor_node_id=ctx.predecessor_node_id,
        )

        task_id = await self.task_queue.enqueue(ctx)
        logger.info(
            f"[execution:{ctx.execution_id}] 📋 Node {ctx.current_node.id} queued with task_id: {task_id}"
        )
        return task_id

    async def execute_node_directly(
        self, ctx: NodeExecutionContext
    ) -> NodeExecutionResult:
        """Execute a node directly (synchronously)."""
        logger.info(
            f"[execution:{ctx.execution_id}] 🎯 Executing node {ctx.current_node.id} directly"
        )
        await self.node_executor.execute(ctx)
        logger.info(
            f"[execution:{ctx.execution_id}] ✅ Node {ctx.current_node.id} executed directly"
        )
        # Return a dummy result since execute doesn't return anything
        from app.core.executions.state import NodeExecutionResult, NodeExecutionStatus

        return NodeExecutionResult(
            status=NodeExecutionStatus.COMPLETED, output="Node executed successfully"
        )

    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a queued task."""
        return await self.task_queue.get_task_status(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task."""
        return await self.task_queue.cancel_task(task_id)

    async def create_workflow_execution(self, workflow_execution: WorkflowExecution):
        """Create a new workflow execution record in the log store."""
        await self.log_store.create_workflow_execution(workflow_execution)
        logger.info(
            f"[execution:{workflow_execution.wf_exec_id}] ✅ Workflow execution record created in database"
        )

    def mark_execution_complete(self, execution_id: str):
        """Mark an execution as complete."""
        logger.info(f"[execution:{execution_id}] 🏁 Execution marked as complete")

    async def _update_workflow_execution_status(
        self, execution_id: str, status: WorkflowExecutionStatus
    ):
        """Update the workflow execution status in the log store."""
        try:
            if hasattr(self.log_store, "update_workflow_execution_status"):
                await self.log_store.update_workflow_execution_status(
                    execution_id, status
                )
                logger.info(
                    f"[execution:{execution_id}] 📊 Updated workflow execution status to: {status.value}"
                )
        except Exception as e:
            logger.error(
                f"[execution:{execution_id}] ❌ Failed to update workflow execution status: {e}"
            )

    async def infer_and_update_workflow_status(self, execution_id: str):
        """
        Infer workflow status from node executions and update it.
        This is a placeholder method that you can update later with your logic.
        """
        try:
            # Get the workflow execution to access node executions
            workflow_execution = await self.log_store.get_wf_execution_by_id(
                execution_id
            )

            # Infer status from node executions
            inferred_status = infer_workflow_status_from_node_executions(
                workflow_execution.node_executions
            )

            # Update the workflow status
            await self._update_workflow_execution_status(execution_id, inferred_status)

            logger.info(
                f"[execution:{execution_id}] 📊 Inferred and updated workflow status to: {inferred_status.value}"
            )

        except Exception as e:
            logger.error(
                f"[execution:{execution_id}] ❌ Failed to infer workflow status: {e}"
            )


class EventProcessor:
    def __init__(
        self,
        wf_store: WorkflowsStore,
        orchestrator: WorkflowOrchestrator,
        trigger_hash_store: TriggerHashStore,
    ):
        self.wf_store = wf_store
        self.orchestrator = orchestrator
        self.trigger_hash_store = trigger_hash_store

    async def process_event(self, trigger_id: str, event: Event):
        """Process the event and trigger the corresponding workflow."""

        # ===== COMPREHENSIVE EVENT LOGGING =====
        logger.info("=" * 80)
        logger.info(f"🎯 EVENT RECEIVED - Event ID: {event.id}")
        logger.info(f"📅 Event Timestamp: {event.time}")
        logger.info(f"🔗 Event Source: {event.source}")
        logger.info(f"📋 Event Source Type: {event.source_type}")
        logger.info(f"🎯 Trigger ID: {trigger_id}")

        # Log event payload (with truncation for large payloads)
        payload_str = str(event.payload)
        if len(payload_str) > 500:
            logger.info(f"📦 Event Payload (truncated): {payload_str[:500]}...")
        else:
            logger.info(f"📦 Event Payload: {payload_str}")

        # Log event headers if present
        if event.headers:
            logger.info(f"📋 Event Headers: {event.headers}")
        else:
            logger.info("📋 Event Headers: None")

        logger.info("=" * 80)

        # ===== ORIGINAL PROCESSING LOGIC =====
        logger.info(f"[eventid:{event.id}] Processing event for trigger {trigger_id}")

        trigger_info = await self.trigger_hash_store.get_trigger_info(trigger_id)
        if not trigger_info:
            logger.warning(
                f"[eventid:{event.id}] Trigger hash {trigger_id} not found in database. This may be an orphaned event."
            )
            return {"status": "skipped", "reason": "trigger_hash_not_found"}

        if not trigger_info.workflow_id:
            logger.warning(
                f"[eventid:{event.id}] Trigger hash {trigger_id} found but has no associated workflow_id. This may be an unconfigured trigger."
            )
            return {"status": "skipped", "reason": "no_workflow_id"}

        wf = await self.wf_store.get_workflow(trigger_info.workflow_id)
        if not wf:
            logger.error(
                f"[eventid:{event.id}] Workflow for id {trigger_info.workflow_id} not found."
            )
            raise ValueError(f"Workflow for id {trigger_info.workflow_id} not found.")

        logger.info(
            f"[eventid:{event.id}] Retrieved workflow {wf.id} with {len(wf.graph.nodes)} nodes"
        )
        logger.info(
            f"[eventid:{event.id}] Workflow nodes: {list(wf.graph.nodes.keys())}"
        )

        if wf.is_paused:
            logger.info(
                f"[eventid:{event.id}] Workflow {wf.id} is paused. Skipping execution"
            )
            return

        # Find the trigger node with the matching hash (using node.config.hash or node.data.hash)
        trigger_node = None
        logger.info(
            f"[eventid:{event.id}] Looking for trigger node with hash {trigger_id}"
        )
        logger.info(f"[eventid:{event.id}] Workflow has {len(wf.graph.nodes)} nodes")

        for node_id, node in wf.graph.nodes.items():
            node_hash = None
            # Check node.hash first since that's where the hash is stored
            node_hash = getattr(node, "hash", None)
            if node_hash is None:
                # Fallback to config.hash or data.hash if needed
                config = getattr(node, "config", None)
                if config is not None:
                    node_hash = getattr(config, "hash", None)
                else:
                    data = getattr(node, "data", None)
                    if data is not None:
                        # Handle both object attributes and dictionary keys
                        if isinstance(data, dict):
                            # For Linear triggers, use unique_identifier (organizationId) instead of organization_id
                            node_hash = data.get("unique_identifier", None)
                            if node_hash is None:
                                node_hash = data.get("organization_id", None)
                            if node_hash is None:
                                node_hash = data.get("hash", None)
                        else:
                            # For Linear triggers, use unique_identifier (organizationId) instead of organization_id
                            node_hash = getattr(data, "unique_identifier", None)
                            if node_hash is None:
                                node_hash = getattr(data, "organization_id", None)
                            if node_hash is None:
                                node_hash = getattr(data, "hash", None)

            if node_hash == trigger_id:
                trigger_node = node
                logger.info(
                    f"[eventid:{event.id}] Found trigger node {node_id} with hash {node_hash}"
                )
                break
        if not trigger_node:
            logger.error(
                f"[eventid:{event.id}] No trigger node with hash {trigger_id} found in workflow {wf.id}."
            )
            raise ValueError(
                f"No trigger node with hash {trigger_id} found in workflow {wf.id}."
            )

        # Create execution ID and initialize workflow execution immediately
        execution_id = f"{wf.id}-{event.id}"
        logger.info(
            f"[eventid:{event.id}] 🚀 Starting workflow execution with ID: {execution_id}"
        )

        # Create workflow execution record immediately when webhook is received
        workflow_execution = WorkflowExecution(
            event=event,
            wf_exec_id=execution_id,
            wf_id=wf.id,
            status=WorkflowExecutionStatus.RUNNING,
            start_time=utc_now(),
            end_time=utc_now(),  # Will be updated when workflow completes
            node_executions=[],
        )

        # Store the workflow execution immediately
        if hasattr(self.orchestrator.log_store, "create_workflow_execution"):
            await self.orchestrator.log_store.create_workflow_execution(
                workflow_execution
            )
            logger.info(
                f"[eventid:{event.id}] ✅ Workflow execution record created in database"
            )

        logger.info(
            f"[eventid:{event.id}] 🎯 Queuing initial trigger node: {trigger_node.id}"
        )

        await self.orchestrator.queue_node_for_execution(
            NodeExecutionContext(
                queued_at=utc_now(),
                workflow_snapshot=wf,
                event=event,
                execution_variables={},
                execution_id=execution_id,
                current_iteration=0,
                previous_node_result="",
                current_node=trigger_node,
                max_iterations=DEFAULT_MAX_ITERATIONS,
            )
        )
        logger.info(f"[eventid:{event.id}] ✅ Initial trigger node queued successfully")

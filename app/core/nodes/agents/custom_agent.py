"""
Custom agent node implementation.

This module defines the custom agent node type and its executor.
"""

from typing import Dict, List
from app.core.executions.agent import AgentQueryExecutor
from app.core.executions.state import (
    NodeExecutionContext,
    NodeExecutionResult,
    NodeExecutionStatus,
)
from app.core.nodes.base import (
    NodeType,
    NodeCategory,
    NodeGroup,
    WorkflowNodeDetails,
)
from app.core.nodes.data_models import CustomAgentData
from app.core.execution_variables import ExecutionVariables
from .base import AgentNodeBase
import logging

logger = logging.getLogger(__name__)


class CustomAgent(AgentNodeBase):
    """Custom agent node for executing specific tasks."""

    type: NodeType = NodeType.CUSTOM_AGENT
    group: NodeGroup = NodeGroup.DEFAULT
    data: CustomAgentData

    def get_next_nodes(
        self, result: NodeExecutionResult, adjacency_list: Dict[str, List[str]]
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the custom agent execution result.

        For custom agent nodes, we only continue to next nodes if the agent execution was successful
        and completed without errors. If the agent failed or encountered an error, we don't queue
        any next nodes to prevent downstream failures.

        Args:
            result: The result of this custom agent node's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes

        Returns:
            List of node IDs that should be queued for execution next
        """
        # Only queue next nodes if the agent execution was successful
        if result.status == NodeExecutionStatus.COMPLETED:
            # Agent completed successfully, continue to all adjacent nodes
            return adjacency_list.get(self.id, [])
        else:
            # Agent failed, partially succeeded, or is in an error state
            # Don't queue any next nodes to prevent downstream failures
            return []


class CustomAgentExecutor:
    """Executor for custom agent nodes."""

    def __init__(self, agent_executor: AgentQueryExecutor):
        """Initialize the executor with an agent query executor."""
        self.agent_executor = agent_executor

    async def execute(
        self, node: CustomAgent, ctx: NodeExecutionContext
    ) -> NodeExecutionResult:
        """
        Execute a custom agent node.

        Args:
            node: The custom agent node to execute
            ctx: Execution context containing workflow and event data

        Returns:
            NodeExecutionResult: Result of the agent execution
        """

        agent_id = node.data.agent_id

        logger.info(
            f"[execution:{ctx.execution_id}] Starting CustomAgent execution - Agent ID: {agent_id}"
        )
        logger.info(f"[execution:{ctx.execution_id}] Task: {node.data.task}")
        logger.info(f"[execution:{ctx.execution_id}] Repository: {node.data.repo_name}")
        logger.info(
            f"[execution:{ctx.execution_id}] Use current repo: {node.data.use_current_repo}"
        )
        logger.info(
            f"[execution:{ctx.execution_id}] Use current branch: {node.data.use_current_branch}"
        )

        if ctx.execution_variables:
            logger.info(
                f"[execution:{ctx.execution_id}] 📊 Available execution variables: {list(ctx.execution_variables.keys())}"
            )
        else:
            logger.info(
                f"[execution:{ctx.execution_id}] 📊 No execution variables available"
            )

        # Build execution variables string
        execution_vars_str = "\n".join(
            [f"{k}: {v}" for k, v in ctx.execution_variables.items()]
        )

        task_description = f"""
        You are an agent that can perform tasks. You are part of a larger workflow.
        
        These are current variables use these to perform your task:
        {execution_vars_str}
        
        Use the result of the previous node to perform your task.
        Previous node result:
        -----------------------
        {ctx.previous_node_result}
        -----------------------
        from this previous result mentioned above consider the output section if any to know the status of previous node execution
        
        This is your task:
        {node.data.task}
        
        Have a output section at the end of your response that contains the result of your task.
        The output result should be verbose and detailed. This output will be used by the next node in the workflow.
        Make sure next node understands everything that you did and the result of your task.
        
        Respond with result status which can be success, failure, partially_success, partially_failure etc
        Add reason for your result status in the output section.
        """

        if not agent_id:
            logger.error(
                f"[execution:{ctx.execution_id}] Missing required field: agent_id"
            )
            return NodeExecutionResult(
                status=NodeExecutionStatus.FAILED,
                output="Missing required field: agent_id",
            )

        try:
            # Determine which repository to use
            repo_name = None
            if (
                node.data.use_current_repo
                and ExecutionVariables.CURRENT_REPO in ctx.execution_variables
            ):
                repo_name = ctx.execution_variables[ExecutionVariables.CURRENT_REPO]
                logger.info(
                    f"[execution:{ctx.execution_id}] Using current repository from execution variables: {repo_name}"
                )
            else:
                # Use the specified repository name
                if not node.data.repo_name:
                    logger.error(
                        f"[execution:{ctx.execution_id}] Missing required fields: repo_name"
                    )
                    return NodeExecutionResult(
                        status=NodeExecutionStatus.FAILED,
                        output="Missing required fields: repo_name",
                    )
                repo_name = node.data.repo_name
                logger.info(
                    f"[execution:{ctx.execution_id}] Using specified repository: {repo_name}"
                )

            # Determine which branch to use
            branch = None
            if (
                node.data.use_current_branch
                and ExecutionVariables.CURRENT_BRANCH in ctx.execution_variables
            ):
                branch = ctx.execution_variables[ExecutionVariables.CURRENT_BRANCH]
                logger.info(
                    f"[execution:{ctx.execution_id}] Using current branch from execution variables: {branch}"
                )
            else:
                # Use the specified branch name
                branch = (
                    node.data.branch_name or "main"
                )  # Fallback to main if not specified
                logger.info(
                    f"[execution:{ctx.execution_id}] Using specified branch: {branch}"
                )

            logger.info(
                f"[execution:{ctx.execution_id}] Calling agent executor with repository: {repo_name}, branch: {branch}"
            )

            res = await self.agent_executor.run_agent(
                user_id=ctx.workflow_snapshot.created_by,
                repo_name=repo_name,
                branch=branch,
                agent_id=agent_id,
                query=task_description,
            )

            # Add execution variables based on the agent execution
            execution_variables = {}
            if ExecutionVariables.CURRENT_BRANCH in ctx.execution_variables:
                execution_variables[ExecutionVariables.PROCESSED_BRANCH] = (
                    ctx.execution_variables[ExecutionVariables.CURRENT_BRANCH]
                )
                execution_variables[ExecutionVariables.AGENT_RESULT] = "success"
                logger.info(
                    f"[execution:{ctx.execution_id}] Agent execution successful - Added execution variables: {list(execution_variables.keys())}"
                )
            else:
                logger.info(
                    f"[execution:{ctx.execution_id}] Agent execution successful - No execution variables added"
                )

            logger.info(
                f"[execution:{ctx.execution_id}] Agent response length: {len(res.text)} characters"
            )
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=res.text,
                execution_variables=execution_variables,
            )
        except Exception as e:
            logger.error(
                f"[execution:{ctx.execution_id}] Agent execution failed: {str(e)}"
            )
            logger.exception(
                f"[execution:{ctx.execution_id}] Full exception traceback:"
            )
            return NodeExecutionResult(
                status=NodeExecutionStatus.FAILED,
                output=f"Agent execution failed: {str(e)}",
            )


# Node definitions for UI
ALL_CUSTOM_AGENTS = [
    WorkflowNodeDetails(
        unique_identifier="agent-custom",
        name="Custom Agent",
        description="Custom Agent Node that can call an agent to perform a given task",
        category=NodeCategory.AGENT,
        group=NodeGroup.DEFAULT,
        type=NodeType.CUSTOM_AGENT,
        icon="bot",
        color="#6f42c1",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "title": "Agent ID"},
                "task": {"type": "string", "title": "Task Description"},
                "repo_name": {
                    "type": "string",
                    "title": "Repository Name",
                    "description": "Optional - leave empty if using current repository",
                },
                "branch_name": {
                    "type": "string",
                    "title": "Branch Name",
                    "description": "Optional - leave empty if using current branch",
                },
                "use_current_repo": {
                    "type": "boolean",
                    "title": "Use Current Repository",
                    "description": "Use the current repository from the trigger context instead of a specific repository",
                },
                "use_current_branch": {
                    "type": "boolean",
                    "title": "Use Current Branch",
                    "description": "Use the current branch from the trigger context instead of a specific branch",
                },
            },
            "required": ["agent_id", "task"],
        },
    )
]

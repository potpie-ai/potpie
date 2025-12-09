"""
Action agent node implementation.

This module defines the action agent node type and its executor.
"""

import functools
import re
from typing import Dict, List
import anyio

from pydantic_ai import Agent, Tool
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.exceptions import ModelRetry, AgentRunError, UserError

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
from app.core.nodes.data_models import ActionAgentData
from app.core.execution_variables import ExecutionVariables
from app.core.intelligence import LLMInterface
from .base import AgentNodeBase
import logging

logger = logging.getLogger(__name__)


class ActionAgent(AgentNodeBase):
    """Action agent node for executing tasks using MCP servers."""

    type: NodeType = NodeType.ACTION_AGENT
    group: NodeGroup = NodeGroup.DEFAULT
    data: ActionAgentData

    def get_next_nodes(
        self, result: NodeExecutionResult, adjacency_list: Dict[str, List[str]]
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the action agent execution result.

        For action agent nodes, we only continue to next nodes if the agent execution was successful
        and completed without errors. If the agent failed or encountered an error, we don't queue
        any next nodes to prevent downstream failures.

        Args:
            result: The result of this action agent node's execution
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


def handle_exception(tool_func):
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        try:
            return tool_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in tool function: {e}")
            return "An internal error occurred. Please try again later."

    return wrapper


class ActionAgentExecutor:
    """Executor for action agent nodes using pydantic-ai."""

    def __init__(self, agent_executor: AgentQueryExecutor):
        """Initialize the executor with an agent query executor."""
        self.agent_executor = agent_executor

    def _create_mcp_servers(
        self, mcp_server_urls: List[str]
    ) -> List[MCPServerStreamableHTTP]:
        """Create MCP server instances from URLs."""
        from app.utils.url_validation import validate_url, URLValidationError
        
        mcp_toolsets: List[MCPServerStreamableHTTP] = []

        for mcp_url in mcp_server_urls:
            try:
                # Validate URL to prevent data: URI DoS attacks and other dangerous schemes
                # This is a defense-in-depth measure (validation should also happen at data model level)
                validate_url(mcp_url, allowed_schemes={"http", "https"})
                
                # Add timeout and connection handling for MCP servers
                mcp_server_instance = MCPServerStreamableHTTP(
                    url=mcp_url, timeout=10.0  # 10 second timeout
                )
                mcp_toolsets.append(mcp_server_instance)
                logger.info(f"Successfully created MCP server: {mcp_url}")
            except URLValidationError as e:
                logger.warning(f"Invalid MCP server URL {mcp_url}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to create MCP server {mcp_url}: {e}")
                continue

        logger.info(
            f"Created {len(mcp_toolsets)} MCP servers out of {len(mcp_server_urls)} configured"
        )
        return mcp_toolsets

    def _create_agent(self, node: ActionAgent, ctx: NodeExecutionContext) -> Agent:
        """Create a pydantic-ai agent with MCP servers."""

        # Create MCP servers
        mcp_toolsets = self._create_mcp_servers(node.data.mcp_servers or [])

        # For now, we'll use a simple model configuration
        # In a real implementation, you would get this from the LLM interface
        # Use a string model name instead of the model class to avoid import issues
        model = "openai:gpt-4o-mini"

        # Build execution variables string
        execution_vars_str = "\n".join(
            [f"{k}: {v}" for k, v in ctx.execution_variables.items()]
        )

        # Create task description
        task_description = f"""
        You are an action agent that can perform tasks using MCP servers. You are part of a larger workflow.
        
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
        
        You have access to the following MCP servers:
        {chr(10).join(node.data.mcp_servers or [])}
        
        Use the appropriate MCP server tools to complete your task. Describe which tool you used including the parameters you sent and the tool schema in the output.
        
        Have a output section at the end of your response that contains the result of your task.
        The output result should be verbose and detailed. This output will be used by the next node in the workflow.
        Make sure next node understands everything that you did and the result of your task.
        
        Respond with result status which can be success, failure, partially_success, partially_failure etc
        Add reason for your result status in the output section.
        """

        return Agent(
            model=model,
            toolsets=mcp_toolsets,
            instructions=task_description,
            output_type=str,
            output_retries=3,
            defer_model_check=True,
            end_strategy="exhaustive",
            model_settings={"max_tokens": 14000},
        )

    async def execute(
        self, node: ActionAgent, ctx: NodeExecutionContext
    ) -> NodeExecutionResult:
        """
        Execute an action agent node using pydantic-ai.

        Args:
            node: The action agent node to execute
            ctx: Execution context containing workflow and event data

        Returns:
            NodeExecutionResult: Result of the agent execution
        """

        logger.info(
            f"[execution:{ctx.execution_id}] Starting ActionAgent execution - Agent Name: {node.data.name}"
        )
        logger.info(f"[execution:{ctx.execution_id}] Task: {node.data.task}")
        logger.info(
            f"[execution:{ctx.execution_id}] MCP Servers: {node.data.mcp_servers}"
        )

        if ctx.execution_variables:
            logger.info(
                f"[execution:{ctx.execution_id}] Available execution variables: {list(ctx.execution_variables.keys())}"
            )
        else:
            logger.info(
                f"[execution:{ctx.execution_id}] No execution variables available"
            )

        try:
            # Create the pydantic-ai agent
            agent = self._create_agent(node, ctx)

            # Try to initialize MCP servers with timeout handling
            try:
                async with agent:
                    resp = await agent.run(
                        user_prompt=node.data.task,
                        message_history=[],
                    )
            except (TimeoutError, anyio.WouldBlock, Exception) as mcp_error:
                logger.warning(f"MCP server initialization failed: {mcp_error}")
                logger.info("Continuing without MCP servers...")

                # Fallback: run without MCP servers
                resp = await agent.run(
                    user_prompt=node.data.task,
                    message_history=[],
                )

            # Add execution variables based on the agent execution
            execution_variables = {}
            if ExecutionVariables.CURRENT_BRANCH in ctx.execution_variables:
                execution_variables[ExecutionVariables.PROCESSED_BRANCH] = (
                    ctx.execution_variables[ExecutionVariables.CURRENT_BRANCH]
                )
                execution_variables[ExecutionVariables.AGENT_RESULT] = "success"
                logger.info(
                    f"[execution:{ctx.execution_id}] ✅ Action agent execution successful - Added execution variables: {list(execution_variables.keys())}"
                )
            else:
                logger.info(
                    f"[execution:{ctx.execution_id}] ✅ Action agent execution successful - No execution variables added"
                )

            logger.info(
                f"[execution:{ctx.execution_id}] 📝 Action agent response length: {len(resp.output)} characters"
            )
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=resp.output,
                execution_variables=execution_variables,
            )
        except Exception as e:
            logger.error(
                f"[execution:{ctx.execution_id}] 💥 Action agent execution failed: {str(e)}"
            )
            logger.exception(
                f"[execution:{ctx.execution_id}] Full exception traceback:"
            )
            return NodeExecutionResult(
                status=NodeExecutionStatus.FAILED,
                output=f"Action agent execution failed: {str(e)}",
            )


# Node definitions for UI
ALL_ACTION_AGENTS = [
    WorkflowNodeDetails(
        unique_identifier="agent-action",
        name="Action Agent",
        description="Action Agent Node that can call MCP servers to perform tasks",
        category=NodeCategory.AGENT,
        group=NodeGroup.DEFAULT,
        type=NodeType.ACTION_AGENT,
        icon="zap",
        color="#10b981",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "mcp_servers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "MCP Servers",
                    "description": "List of MCP server URLs to use for task execution",
                },
                "name": {
                    "type": "string",
                    "title": "Agent Name",
                    "description": "Name of the action agent",
                },
                "task": {
                    "type": "string",
                    "title": "Task Description",
                    "description": "Task description for the action agent",
                },
            },
            "required": ["mcp_servers", "name", "task"],
        },
    )
]

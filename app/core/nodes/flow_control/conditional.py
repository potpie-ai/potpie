"""
Conditional flow control node implementation.

This module defines the conditional node type for routing execution based on conditions.
"""

from typing import List, Union, Optional, Dict
from pydantic import BaseModel, Field

from app.core.executions.state import NodeExecutionResult
from app.core.nodes.base import (
    NodeType,
    NodeCategory,
    NodeGroup,
    WorkflowNodeDetails,
)
from app.core.nodes.data_models import ConditionalNodeData
from .base import FlowControlNodeBase


class ConditionalEvaluationResult(BaseModel):
    """Structured output for conditional evaluation using LLM."""

    result: bool = Field(
        ..., description="The boolean result of the condition evaluation"
    )
    should_continue: bool = Field(
        ..., description="Whether to continue workflow execution to next nodes"
    )
    reasoning: str = Field(
        ..., description="Explanation of why the condition evaluated to this result"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in the evaluation (0.0 to 1.0)",
    )


class ConditionalNode(FlowControlNodeBase):
    """Conditional flow control node that routes execution based on a condition."""

    type: NodeType = NodeType.FLOW_CONTROL_CONDITIONAL
    data: ConditionalNodeData

    def get_next_nodes(
        self, result: NodeExecutionResult, adjacency_list: Dict[str, List[str]]
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the conditional evaluation result.

        For conditional nodes, we route to different paths based on the condition result.
        If the condition is true, we follow the true path; if false, we follow the false path.
        If should_continue is False, we don't queue any next nodes.

        Args:
            result: The result of this conditional node's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes

        Returns:
            List of node IDs that should be queued for execution next
        """
        # Get all adjacent nodes
        all_next_nodes = adjacency_list.get(self.id, [])

        # If no next nodes, return empty list
        if not all_next_nodes:
            return []

        # Check if the result contains conditional routing information
        if isinstance(result.output, dict) and "should_continue" in result.output:
            should_continue = result.output["should_continue"]

            # If should_continue is False, don't queue any next nodes
            if not should_continue:
                return []

        # Default behavior: return all adjacent nodes
        # Note: For more complex routing (true/false paths), the workflow graph
        # should be structured with separate edges for true and false conditions
        return all_next_nodes


# Node definitions for UI
ALL_CONDITIONAL_NODES = [
    WorkflowNodeDetails(
        unique_identifier="flow-control-conditional",
        name="Conditional",
        description="Conditional flow control that evaluates a condition and routes to different paths",
        category=NodeCategory.FLOW_CONTROL,
        group=NodeGroup.DEFAULT,
        type=NodeType.FLOW_CONTROL_CONDITIONAL,
        icon="git-branch",
        color="#17a2b8",
        inputs=["input"],
        outputs=["true", "false"],
        config_schema={
            "type": "object",
            "properties": {
                "condition": {"type": "string", "title": "Condition Expression"},
                "true_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "True Path",
                },
                "false_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "False Path",
                },
            },
            "required": ["condition"],
        },
    ),
]

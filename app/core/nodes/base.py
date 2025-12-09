"""
Core base models and enums for workflow nodes.

This module defines the foundational types and base classes used throughout
the node system, including enums for node categories, groups, and types,
as well as the base node model and UI definition model.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, Union, List, Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .data_models import BaseNodeData
    from app.core.executions.state import NodeExecutionResult


class Position(BaseModel):
    """Position coordinates for node placement in the UI."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class NodeCategory(str, Enum):
    """Node categories for grouping and dispatching node execution."""

    TRIGGER = "trigger"
    AGENT = "agent"
    FLOW_CONTROL = "flow_control"
    MANUAL_STEP = "manual_step"


class NodeGroup(str, Enum):
    """Node groups for organizing nodes by service or functionality."""

    GITHUB = "github"
    LINEAR = "linear"
    SENTRY = "sentry"
    DEFAULT = "default"


class NodeType(str, Enum):
    """Specific node types for identification and registration."""

    # Triggers
    TRIGGER_GITHUB_PR_OPENED = "trigger_github_pr_opened"
    TRIGGER_GITHUB_PR_CLOSED = "trigger_github_pr_closed"
    TRIGGER_GITHUB_PR_REOPENED = "trigger_github_pr_reopened"
    TRIGGER_GITHUB_PR_MERGED = "trigger_github_pr_merged"
    TRIGGER_GITHUB_ISSUE_OPENED = "trigger_github_issue_opened"
    TRIGGER_LINEAR_ISSUE_CREATED = "trigger_linear_issue_created"
    TRIGGER_SENTRY_ISSUE_CREATED = "trigger_sentry_issue_created"
    TRIGGER_WEBHOOK = "trigger_webhook"

    # Agents
    CUSTOM_AGENT = "custom_agent"
    ACTION_AGENT = "action_agent"

    # Flow Control
    FLOW_CONTROL_CONDITIONAL = "flow_control_conditional"
    FLOW_CONTROL_COLLECT = "flow_control_collect"
    FLOW_CONTROL_SELECTOR = "flow_control_selector"

    # Manual Steps
    MANUAL_STEP_APPROVAL = "manual_step_approval"
    MANUAL_STEP_INPUT = "manual_step_input"


# Type variable for node data
T = TypeVar("T", bound="BaseNodeData")


class WorkflowNodeBase(BaseModel, ABC, Generic[T]):
    """
    Base class for all workflow nodes.

    All node types must inherit from this class and define their specific
    type, group, category, and data model. The data field contains strongly-typed
    node-specific configuration and runtime data.
    """

    id: str = Field(..., description="Unique node identifier")
    type: NodeType = Field(..., description="Node type")
    group: NodeGroup = Field(..., description="Node group")
    category: NodeCategory = Field(..., description="Node category")
    position: Position = Field(..., description="Node position in UI")
    data: T = Field(..., description="Node-specific typed data")

    model_config = {"use_enum_values": True, "arbitrary_types_allowed": True}

    def get_next_nodes(
        self, result: "NodeExecutionResult", adjacency_list: Dict[str, List[str]]
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the execution result.

        This method allows each node type to control the flow of execution.
        The default implementation returns all adjacent nodes, but subclasses
        can override this to implement custom routing logic.

        Args:
            result: The result of this node's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes

        Returns:
            List of node IDs that should be queued for execution next
        """
        # Default behavior: return all adjacent nodes
        return adjacency_list.get(self.id, [])


class WorkflowNodeDetails(BaseModel):
    """
    Node definition with metadata for UI display and configuration.

    This model contains all the information needed to render a node
    in the UI, including its schema for configuration forms.
    """

    unique_identifier: str = Field(
        ..., description="Unique identifier for the node type"
    )
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Node description")
    category: NodeCategory = Field(..., description="Node category")
    group: NodeGroup = Field(..., description="Node group")
    type: NodeType = Field(..., description="Node type")
    icon: Optional[str] = Field(None, description="UI icon identifier")
    color: Optional[str] = Field(None, description="UI color hex code")
    inputs: List[str] = Field(default_factory=list, description="Input port names")
    outputs: List[str] = Field(default_factory=list, description="Output port names")
    config_schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON schema for configuration"
    )


# Union type for all possible node types
# This will be populated in __init__.py with all concrete node types
WorkflowNode = Union["WorkflowNodeBase"]

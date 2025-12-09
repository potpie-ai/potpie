"""
Core node types and registry for workflow nodes.

This module serves as the central registry for all node types in the system.
It provides utilities for node discovery, creation, and management.

To add a new node type:
1. Create the node class in the appropriate module (agents/, triggers/, flow_control/, etc.)
2. Add it to the NODE_TYPE_REGISTRY mapping
3. Add it to the WorkflowNode union type
4. Create a WorkflowNodeDetails entry for UI display
5. Update the appropriate ALL_* list and category/group mappings
"""

from typing import Dict, List, Union, Type

from app.core.nodes.base import (
    WorkflowNodeBase,
    WorkflowNodeDetails,
    NodeType,
    NodeCategory,
    NodeGroup,
    Position,
)
from app.core.nodes.data_models import BaseNodeData, NodeData

# Import all node types
from .agents import CustomAgent, ActionAgent, AgentNode, ALL_AGENTS
from .flow_control import (
    ConditionalNode,
    CollectNode,
    SelectorNode,
    FlowControlNode,
    ALL_FLOW_CONTROL,
)
from .manual_steps import ApprovalNode, InputNode, ManualStep, ALL_MANUAL_STEPS
from .triggers.github import (
    GithubPROpenedTrigger,
    GithubPRClosedTrigger,
    GithubPRReopenedTrigger,
    GithubPRMergedTrigger,
    GithubIssueOpenedTrigger,
    GithubTrigger,
    ALL_GITHUB_TRIGGERS,
)
from .triggers.linear import (
    LinearIssueCreatedTrigger,
    LinearTrigger,
    ALL_LINEAR_TRIGGERS,
)
from .triggers.sentry import (
    SentryIssueCreatedTrigger,
    SentryTrigger,
    ALL_SENTRY_TRIGGERS,
)
from .triggers.webhook import (
    WebhookTrigger,
    ALL_WEBHOOK_TRIGGERS,
)

# Union type for all possible workflow nodes
WorkflowNode = Union[
    # Triggers
    GithubPROpenedTrigger,
    GithubPRClosedTrigger,
    GithubPRReopenedTrigger,
    GithubPRMergedTrigger,
    GithubIssueOpenedTrigger,
    LinearIssueCreatedTrigger,
    SentryIssueCreatedTrigger,
    WebhookTrigger,
    # Agents
    CustomAgent,
    ActionAgent,
    # Flow Control
    ConditionalNode,
    CollectNode,
    SelectorNode,
    # Manual Steps
    ApprovalNode,
    InputNode,
]

# Node type registry - maps node type strings to their classes
# This is the single source of truth for node type → class mapping
NODE_TYPE_REGISTRY: Dict[str, Type[WorkflowNodeBase]] = {
    # GitHub Triggers
    NodeType.TRIGGER_GITHUB_PR_OPENED: GithubPROpenedTrigger,
    NodeType.TRIGGER_GITHUB_PR_CLOSED: GithubPRClosedTrigger,
    NodeType.TRIGGER_GITHUB_PR_REOPENED: GithubPRReopenedTrigger,
    NodeType.TRIGGER_GITHUB_PR_MERGED: GithubPRMergedTrigger,
    NodeType.TRIGGER_GITHUB_ISSUE_OPENED: GithubIssueOpenedTrigger,
    # Linear Triggers
    NodeType.TRIGGER_LINEAR_ISSUE_CREATED: LinearIssueCreatedTrigger,
    # Sentry Triggers
    NodeType.TRIGGER_SENTRY_ISSUE_CREATED: SentryIssueCreatedTrigger,
    # Webhook Triggers
    NodeType.TRIGGER_WEBHOOK: WebhookTrigger,
    # Agents
    NodeType.CUSTOM_AGENT: CustomAgent,
    NodeType.ACTION_AGENT: ActionAgent,
    # Flow Control
    NodeType.FLOW_CONTROL_CONDITIONAL: ConditionalNode,
    NodeType.FLOW_CONTROL_COLLECT: CollectNode,
    NodeType.FLOW_CONTROL_SELECTOR: SelectorNode,
    # Manual Steps
    NodeType.MANUAL_STEP_APPROVAL: ApprovalNode,
    NodeType.MANUAL_STEP_INPUT: InputNode,
}

# All node definitions for UI
ALL_NODE_DEFINITIONS: List[WorkflowNodeDetails] = [
    *ALL_GITHUB_TRIGGERS,
    *ALL_LINEAR_TRIGGERS,
    *ALL_SENTRY_TRIGGERS,
    *ALL_WEBHOOK_TRIGGERS,
    *ALL_AGENTS,
    *ALL_FLOW_CONTROL,
    *ALL_MANUAL_STEPS,
]

# Node definitions by category
NODE_DEFINITIONS_BY_CATEGORY: Dict[NodeCategory, List[WorkflowNodeDetails]] = {
    NodeCategory.TRIGGER: [
        *ALL_GITHUB_TRIGGERS,
        *ALL_LINEAR_TRIGGERS,
        *ALL_SENTRY_TRIGGERS,
        *ALL_WEBHOOK_TRIGGERS,
    ],
    NodeCategory.AGENT: ALL_AGENTS,
    NodeCategory.FLOW_CONTROL: ALL_FLOW_CONTROL,
    NodeCategory.MANUAL_STEP: ALL_MANUAL_STEPS,
}

# Node definitions by group
NODE_DEFINITIONS_BY_GROUP: Dict[NodeGroup, List[WorkflowNodeDetails]] = {
    NodeGroup.GITHUB: ALL_GITHUB_TRIGGERS,
    NodeGroup.LINEAR: ALL_LINEAR_TRIGGERS,
    NodeGroup.SENTRY: ALL_SENTRY_TRIGGERS,
    NodeGroup.DEFAULT: [
        *ALL_AGENTS,
        *ALL_FLOW_CONTROL,
        *ALL_MANUAL_STEPS,
        *ALL_WEBHOOK_TRIGGERS,
    ],
}


def get_node_class(node_type: str) -> Type[WorkflowNodeBase]:
    """
    Get the node class for a given node type string.

    Args:
        node_type: The node type string

    Returns:
        Type[WorkflowNodeBase]: The node class

    Raises:
        ValueError: If the node type is unknown
    """
    if node_type not in NODE_TYPE_REGISTRY:
        raise ValueError(f"Unknown node type: {node_type}")
    return NODE_TYPE_REGISTRY[node_type]


def get_node_definitions_by_category(
    category: NodeCategory,
) -> List[WorkflowNodeDetails]:
    """
    Get all node definitions for a specific category.

    Args:
        category: The node category

    Returns:
        List[WorkflowNodeDetails]: List of node definitions
    """
    return NODE_DEFINITIONS_BY_CATEGORY.get(category, [])


def get_node_definitions_by_group(group: NodeGroup) -> List[WorkflowNodeDetails]:
    """
    Get all node definitions for a specific group.

    Args:
        group: The node group

    Returns:
        List[WorkflowNodeDetails]: List of node definitions
    """
    return NODE_DEFINITIONS_BY_GROUP.get(group, [])


def get_node_definition_by_type(node_type: NodeType) -> WorkflowNodeDetails:
    """
    Get node definition for a specific node type.

    Args:
        node_type: The node type

    Returns:
        WorkflowNodeDetails: The node definition

    Raises:
        ValueError: If no definition is found for the node type
    """
    for definition in ALL_NODE_DEFINITIONS:
        if definition.type == node_type:
            return definition
    raise ValueError(f"No definition found for node type: {node_type}")


# Public API exports
__all__ = [
    # Base classes and enums
    "WorkflowNodeBase",
    "WorkflowNodeDetails",
    "NodeType",
    "NodeCategory",
    "NodeGroup",
    "Position",
    "WorkflowNode",
    # Node classes
    "CustomAgent",
    "ConditionalNode",
    "CollectNode",
    "SelectorNode",
    "ApprovalNode",
    "InputNode",
    "GithubPROpenedTrigger",
    "GithubPRClosedTrigger",
    "GithubPRReopenedTrigger",
    "GithubPRMergedTrigger",
    "GithubIssueOpenedTrigger",
    "LinearIssueCreatedTrigger",
    "WebhookTrigger",
    # Data models
    "BaseNodeData",
    "NodeData",
    # Registries and utilities
    "NODE_TYPE_REGISTRY",
    "ALL_NODE_DEFINITIONS",
    "NODE_DEFINITIONS_BY_CATEGORY",
    "NODE_DEFINITIONS_BY_GROUP",
    "get_node_class",
    "get_node_definitions_by_category",
    "get_node_definitions_by_group",
    "get_node_definition_by_type",
]

"""
Node factory for creating and deserializing typed nodes.

This module provides utilities for creating nodes with proper type safety
and deserializing nodes from JSON data with validation.
"""

from typing import Dict, Any, Type, Optional
from app.core.nodes.base import WorkflowNodeBase, NodeType, Position
from app.core.nodes.data_models import (
    NODE_DATA_MODELS,
    create_node_data,
    BaseNodeData,
)

# Import registry directly to avoid circular import
from app.core.nodes.triggers.github import (
    GithubPROpenedTrigger,
    GithubPRClosedTrigger,
    GithubPRReopenedTrigger,
    GithubPRMergedTrigger,
    GithubIssueOpenedTrigger,
)
from app.core.nodes.triggers.linear import LinearIssueCreatedTrigger
from app.core.nodes.triggers.sentry import SentryIssueCreatedTrigger
from app.core.nodes.triggers.webhook import WebhookTrigger
from app.core.nodes.agents.custom_agent import CustomAgent
from app.core.nodes.agents.action_agent import ActionAgent
from app.core.nodes.flow_control.conditional import ConditionalNode
from app.core.nodes.flow_control.collect import CollectNode
from app.core.nodes.flow_control.selector import SelectorNode
from app.core.nodes.manual_steps.manual_step import ApprovalNode, InputNode

# Create registry mapping
NODE_TYPE_REGISTRY = {
    # GitHub Triggers
    "trigger_github_pr_opened": GithubPROpenedTrigger,
    "trigger_github_pr_closed": GithubPRClosedTrigger,
    "trigger_github_pr_reopened": GithubPRReopenedTrigger,
    "trigger_github_pr_merged": GithubPRMergedTrigger,
    "trigger_github_issue_opened": GithubIssueOpenedTrigger,
    # Linear Triggers
    "trigger_linear_issue_created": LinearIssueCreatedTrigger,
    # Sentry Triggers
    "trigger_sentry_issue_created": SentryIssueCreatedTrigger,
    # Webhook Triggers
    "trigger_webhook": WebhookTrigger,
    # Agents
    "custom_agent": CustomAgent,
    "action_agent": ActionAgent,
    # Flow Control
    "flow_control_conditional": ConditionalNode,
    "flow_control_collect": CollectNode,
    "flow_control_selector": SelectorNode,
    # Manual Steps
    "manual_step_approval": ApprovalNode,
    "manual_step_input": InputNode,
}


class NodeFactory:
    """Factory for creating and deserializing typed nodes."""

    @staticmethod
    def create_node(
        node_type: NodeType,
        node_id: str,
        position: Position,
        data: Dict[str, Any],
        **kwargs,
    ) -> WorkflowNodeBase:
        """
        Create a typed node with validated data.

        Args:
            node_type: The type of node to create
            node_id: Unique node identifier
            position: Node position in UI
            data: Node configuration data
            **kwargs: Additional node properties

        Returns:
            WorkflowNodeBase: The created node

        Raises:
            ValueError: If node type is unknown or data is invalid
        """
        # Get the node class
        node_class = NODE_TYPE_REGISTRY.get(node_type.value)
        if not node_class:
            raise ValueError(f"Unknown node type: {node_type}")

        # Create typed data
        typed_data = create_node_data(node_type, data)

        # Create the node
        return node_class(
            id=node_id, type=node_type, position=position, data=typed_data, **kwargs
        )

    @staticmethod
    def deserialize_node(node_data: Dict[str, Any]) -> Optional[WorkflowNodeBase]:
        """
        Deserialize a node from JSON data with type validation.

        Args:
            node_data: Raw node data from JSON

        Returns:
            WorkflowNodeBase: The deserialized node, or None if invalid

        Raises:
            ValueError: If node data is invalid
        """
        try:
            # Extract basic properties
            node_id = node_data.get("id")
            node_type_str = node_data.get("type")
            position = node_data.get("position", {})
            raw_data = node_data.get("data", {})

            if not node_id or not node_type_str:
                return None

            # Convert string to NodeType enum
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                return None

            # Create the node using the factory
            return NodeFactory.create_node(
                node_type=node_type,
                node_id=node_id,
                position=Position(**position),
                data=raw_data,
                group=node_data.get("group"),
                category=node_data.get("category"),
            )

        except Exception as e:
            # Log the error but don't fail the entire deserialization
            print(f"Failed to deserialize node {node_data.get('id', 'unknown')}: {e}")
            return None

    @staticmethod
    def deserialize_nodes(
        nodes_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, WorkflowNodeBase]:
        """
        Deserialize multiple nodes from JSON data.

        Args:
            nodes_dict: Dictionary of node data

        Returns:
            Dict[str, WorkflowNodeBase]: Dictionary of deserialized nodes
        """
        deserialized_nodes = {}

        for node_id, node_data in nodes_dict.items():
            node = NodeFactory.deserialize_node(node_data)
            if node:
                deserialized_nodes[node_id] = node

        return deserialized_nodes

"""
Utility functions for working with workflow nodes.

This module provides utilities for creating, validating, and querying
workflow nodes, including node creation from dictionaries, validation,
and schema access.
"""

from typing import Dict, Any, Optional, Tuple, List
from pydantic import ValidationError

from app.core.nodes.base import WorkflowNodeBase, NodeType, Position
from app.core.nodes import NODE_TYPE_REGISTRY, get_node_class


def create_node_from_dict(node_data: Dict[str, Any]) -> WorkflowNodeBase:
    """
    Create a workflow node from a dictionary representation.

    Args:
        node_data: Dictionary containing node data with 'type' field

    Returns:
        WorkflowNodeBase: Created node instance

    Raises:
        ValueError: If node type is unknown or data is invalid
    """
    if "type" not in node_data:
        raise ValueError("Node data must contain 'type' field")

    node_type = node_data["type"]

    try:
        # Get the node class for this type
        node_class = get_node_class(node_type)

        # Create the node instance
        return node_class(**node_data)

    except KeyError:
        raise ValueError(f"Unknown node type: {node_type}")
    except ValidationError as e:
        raise ValueError(f"Invalid node data: {e}")


def validate_node_data(node_data: Dict[str, Any]) -> bool:
    """
    Validate node data without creating the node.

    Args:
        node_data: Dictionary containing node data

    Returns:
        bool: True if data is valid

    Raises:
        ValueError: If data is invalid
    """
    if "type" not in node_data:
        raise ValueError("Node data must contain 'type' field")

    node_type = node_data["type"]

    if node_type not in NODE_TYPE_REGISTRY:
        raise ValueError(f"Unknown node type: {node_type}")

    # Try to create the node to validate data
    try:
        node_class = get_node_class(node_type)
        node_class(**node_data)
        return True
    except ValidationError as e:
        raise ValueError(f"Invalid node data: {e}")


def get_node_config_schema(node_type: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration schema for a node type.

    Args:
        node_type: The node type string

    Returns:
        Dict[str, Any]: JSON schema for node configuration, or None if not found
    """
    from app.core.nodes import get_node_definition_by_type

    try:
        node_type_enum = NodeType(node_type)
        definition = get_node_definition_by_type(node_type_enum)
        return definition.config_schema
    except (ValueError, KeyError):
        return None


def create_position(x: float, y: float) -> Position:
    """
    Create a position object.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        Position: Position object
    """
    return Position(x=x, y=y)


def get_node_inputs_outputs(node_type: str) -> Tuple[List[str], List[str]]:
    """
    Get the input and output ports for a node type.

    Args:
        node_type: The node type string

    Returns:
        Tuple[List[str], List[str]]: (inputs, outputs) lists
    """
    from app.core.nodes import get_node_definition_by_type

    try:
        node_type_enum = NodeType(node_type)
        definition = get_node_definition_by_type(node_type_enum)
        return definition.inputs, definition.outputs
    except (ValueError, KeyError):
        return [], []


def is_valid_node_type(node_type: str) -> bool:
    """
    Check if a node type is valid.

    Args:
        node_type: The node type string to check

    Returns:
        bool: True if the node type is valid
    """
    return node_type in NODE_TYPE_REGISTRY


def get_required_node_fields(node_type: str) -> List[str]:
    """
    Get the required fields for a node type.

    Args:
        node_type: The node type string

    Returns:
        List[str]: List of required field names
    """
    schema = get_node_config_schema(node_type)
    if schema and "required" in schema:
        return schema["required"]
    return []


def migrate_node_data(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate legacy/camelCase node data fields to snake_case for each node type.
    This ensures compatibility with the latest Pydantic models.
    """
    import copy

    migrated = copy.deepcopy(node_data)
    node_type = migrated.get("type")
    if not node_type:
        return migrated

    # --- GitHub Triggers ---
    if node_type in [
        "trigger_github_pr_opened",
        "trigger_github_pr_closed",
        "trigger_github_pr_reopened",
        "trigger_github_pr_merged",
        "trigger_github_issue_opened",
        "trigger_linear_issue_created",
    ]:
        data = migrated.get("data", {})
        # Map 'repository' or 'repo_name' to 'repo_name'
        if "repository" in data:
            data["repo_name"] = data.pop("repository")
        # Map 'webhookHash' to 'hash' (only if hash is not already at top level)
        if "webhookHash" in data and "hash" not in migrated:
            migrated["hash"] = data.pop("webhookHash")
        migrated["data"] = data

    # --- Custom Agent ---
    if node_type == "custom_agent":
        data = migrated.get("data", {})
        # Map 'agentId' to 'agent_id', remove 'agentName' entirely
        if "agentId" in data:
            data["agent_id"] = data.pop("agentId")
        if "agentName" in data:
            data.pop("agentName")  # Remove agentName entirely
        migrated["data"] = data

    # --- Flow Control Conditional ---
    if node_type == "flow_control_conditional":
        data = migrated.get("data", {})
        # Ensure true_path and false_path are present
        if "true_path" not in data:
            data["true_path"] = []
        if "false_path" not in data:
            data["false_path"] = []
        migrated["data"] = data

    # Add more node type migrations as needed
    return migrated

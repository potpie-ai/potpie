"""
Core data models for manual step nodes.

This module defines manual step node types that require human interaction,
including approval steps and input collection steps.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from app.core.nodes.base import (
    NodeType,
    NodeCategory,
    NodeGroup,
    WorkflowNodeDetails,
    WorkflowNodeBase,
)
from app.core.nodes.data_models import ApprovalNodeData, InputNodeData
from app.core.executions.state import NodeExecutionResult


# Data models are now imported from data_models.py


# Manual step node classes
class ManualStepNodeBase(WorkflowNodeBase):
    """Base class for manual step nodes."""

    category: NodeCategory = NodeCategory.MANUAL_STEP
    group: NodeGroup = NodeGroup.DEFAULT
    timeout: int = Field(default=24 * 60, description="Timeout in minutes")


class ApprovalNode(ManualStepNodeBase):
    """Approval manual step node that requires human approval."""

    type: NodeType = NodeType.MANUAL_STEP_APPROVAL
    data: ApprovalNodeData

    def get_next_nodes(
        self, result: NodeExecutionResult, adjacency_list: Dict[str, List[str]]
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on approval response.

        For approval nodes, routing is based on the approval decision:
        - If approved: Route to nodes connected via "approved" edge or all edges if no labels
        - If rejected: Route to nodes connected via "rejected" edge or no nodes if no labels

        Args:
            result: The result of this approval node's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes

        Returns:
            List of node IDs that should be queued for execution next
        """
        all_next_nodes = adjacency_list.get(self.id, [])

        if not all_next_nodes:
            return []

        # Check if result contains approval decision
        if isinstance(result.output, dict):
            approved = result.output.get("approved", False)
            # Check if there's a selected_path (user explicitly chose a path)
            selected_path = result.output.get("selected_path")
            
            if selected_path:
                # User selected a specific path, route only to that node
                if selected_path in all_next_nodes:
                    return [selected_path]
                return []

            # Route based on approval decision
            # For now, if approved, route to all next nodes
            # If rejected, don't route anywhere (workflow stops)
            if approved:
                return all_next_nodes
            else:
                # Rejected - don't continue to next nodes
                return []

        # Default: if we have a response but can't parse it, don't route
        # This handles the case where the node is still waiting
        return []


class InputNode(ManualStepNodeBase):
    """Input manual step node that collects data from users."""

    type: NodeType = NodeType.MANUAL_STEP_INPUT
    data: InputNodeData

    def get_next_nodes(
        self, result: NodeExecutionResult, adjacency_list: Dict[str, List[str]]
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on input response.

        For input nodes, routing can be:
        - Based on selected_path if user explicitly chose a path
        - Based on input field values matching conditions (if configured)
        - Default: Route to all next nodes

        Args:
            result: The result of this input node's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes

        Returns:
            List of node IDs that should be queued for execution next
        """
        all_next_nodes = adjacency_list.get(self.id, [])

        if not all_next_nodes:
            return []

        # Check if result contains input response
        if isinstance(result.output, dict):
            # Check if user selected a specific path
            selected_path = result.output.get("selected_path")
            
            if selected_path:
                # User selected a specific path, route only to that node
                if selected_path in all_next_nodes:
                    return [selected_path]
                return []

            # For input nodes, by default route to all next nodes
            # Future: Could add conditional routing based on field values
            return all_next_nodes

        # Default: if we have a response, route to all next nodes
        if result.output:
            return all_next_nodes

        # If no response yet, don't route
        return []


ManualStep = Union[ApprovalNode, InputNode]

# Node definitions for UI
ALL_MANUAL_STEPS = [
    WorkflowNodeDetails(
        unique_identifier="manual-step-approval",
        name="Approval",
        description="Manual approval step that requires human approval",
        category=NodeCategory.MANUAL_STEP,
        group=NodeGroup.DEFAULT,
        type=NodeType.MANUAL_STEP_APPROVAL,
        icon="check-circle",
        color="#28a745",
        inputs=["input"],
        outputs=["approved", "rejected"],
        config_schema={
            "type": "object",
            "properties": {
                "approvers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "title": "Approvers",
                },
                "approval_message": {"type": "string", "title": "Approval Message"},
                "timeout_action": {
                    "type": "string",
                    "enum": ["approve", "reject"],
                    "title": "Timeout Action",
                },
            },
            "required": ["approval_message"],
        },
    ),
    WorkflowNodeDetails(
        unique_identifier="manual-step-input",
        name="Input",
        description="Manual input step that collects data from users",
        category=NodeCategory.MANUAL_STEP,
        group=NodeGroup.DEFAULT,
        type=NodeType.MANUAL_STEP_INPUT,
        icon="edit",
        color="#17a2b8",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "input_fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "title": "Field Name"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "text",
                                    "number",
                                    "select",
                                    "multi_select",
                                    "file",
                                ],
                                "title": "Field Type",
                            },
                            "required": {"type": "boolean", "title": "Required"},
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "title": "Options",
                            },
                        },
                        "required": ["name"],
                    },
                    "title": "Input Fields",
                },
                "assignee": {"type": "string", "title": "Assignee"},
            },
            "required": ["assignee"],
        },
    ),
]

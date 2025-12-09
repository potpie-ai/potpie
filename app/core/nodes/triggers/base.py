"""
Core data model for trigger nodes.

This module defines the base classes for trigger nodes, which are responsible
for initiating workflow executions based on external events.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from app.core.executions.event import Event
from app.core.executions.state import NodeExecutionContext, NodeExecutionResult
from app.core.nodes.base import NodeType, NodeCategory, NodeGroup, WorkflowNodeBase


class TriggerGroup(str, Enum):
    """Enum for trigger groups."""

    GITHUB = "github"
    LINEAR = "linear"
    WEBHOOK = "webhook"


class TriggerNode(WorkflowNodeBase):
    """Base class for all trigger nodes."""

    category: NodeCategory = NodeCategory.TRIGGER

    def get_next_nodes(
        self, result: NodeExecutionResult, adjacency_list: Dict[str, List[str]]
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the trigger execution result.

        This method delegates to the appropriate trigger executor to determine routing.

        Args:
            result: The result of this trigger's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes

        Returns:
            List of node IDs that should be queued for execution next
        """
        from app.core.nodes.triggers.executors import get_trigger_executor_registry

        # Get the appropriate executor for this trigger
        executor_registry = get_trigger_executor_registry()
        executor = executor_registry.get_executor(self)

        # Use the executor's get_next_nodes method, passing the trigger ID
        return executor.get_next_nodes(result, adjacency_list, self.id)


# Union type for all trigger nodes
# This will be populated in the specific trigger modules
TriggerNodeType = Union["TriggerNode"]

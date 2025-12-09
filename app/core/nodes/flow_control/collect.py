"""
Collect flow control node implementation.

This module defines the collect node type for waiting for parallel branches to complete.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field

from app.core.nodes.base import (
    NodeType,
    NodeCategory,
    NodeGroup,
    WorkflowNodeDetails,
)
from app.core.nodes.data_models import CollectNodeData
from .base import FlowControlNodeBase


class CollectNode(FlowControlNodeBase):
    """Collect flow control node that waits for parallel branches to complete."""

    type: NodeType = NodeType.FLOW_CONTROL_COLLECT
    data: CollectNodeData


# Node definitions for UI
ALL_COLLECT_NODES = [
    WorkflowNodeDetails(
        unique_identifier="flow-control-collect",
        name="Collect",
        description="Collects results from parallel branches",
        category=NodeCategory.FLOW_CONTROL,
        group=NodeGroup.DEFAULT,
        type=NodeType.FLOW_CONTROL_COLLECT,
        icon="git-merge",
        color="#28a745",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "parallel_branches": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "title": "Parallel Branches",
                },
                "join_strategy": {
                    "type": "string",
                    "enum": ["wait_for_all", "first_complete"],
                    "title": "Join Strategy",
                },
            },
        },
    ),
]

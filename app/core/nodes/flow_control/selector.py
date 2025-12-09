"""
Selector flow control node implementation.

This module defines the selector node type for choosing one of multiple branches.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field

from app.core.nodes.base import (
    NodeType,
    NodeCategory,
    NodeGroup,
    WorkflowNodeDetails,
)
from app.core.nodes.data_models import SelectorNodeData
from .base import FlowControlNodeBase


class SelectorNode(FlowControlNodeBase):
    """Selector flow control node that chooses one of multiple branches."""

    type: NodeType = NodeType.FLOW_CONTROL_SELECTOR
    data: SelectorNodeData


# Node definitions for UI
ALL_SELECTOR_NODES = [
    WorkflowNodeDetails(
        unique_identifier="flow-control-selector",
        name="Selector",
        description="Selects one of multiple branches based on an expression",
        category=NodeCategory.FLOW_CONTROL,
        group=NodeGroup.DEFAULT,
        type=NodeType.FLOW_CONTROL_SELECTOR,
        icon="git-branch",
        color="#ffc107",
        inputs=["input"],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "selector_expression": {
                    "type": "string",
                    "title": "Selector Expression",
                },
                "branches": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "title": "Branches",
                },
            },
            "required": ["selector_expression"],
        },
    ),
]

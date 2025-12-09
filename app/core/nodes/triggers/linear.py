"""
Linear trigger node types.

This module defines Linear-specific trigger nodes for Linear events
like issue creation.
"""

from typing import Literal, Union, Optional
from pydantic import BaseModel, Field
from app.core.nodes.base import NodeType, NodeCategory, NodeGroup, WorkflowNodeDetails
from app.core.nodes.triggers.base import TriggerNode
from app.core.nodes.data_models import LinearIssueCreatedTriggerData


# Data models are now imported from data_models.py


# Linear trigger node classes
class LinearTriggerBase(TriggerNode):
    """Base class for Linear triggers."""

    group: NodeGroup = NodeGroup.LINEAR


class LinearIssueCreatedTrigger(LinearTriggerBase):
    """Linear issue created trigger node."""

    type: NodeType = NodeType.TRIGGER_LINEAR_ISSUE_CREATED
    data: LinearIssueCreatedTriggerData


LinearTrigger = Union[LinearIssueCreatedTrigger]

# Node definitions for UI
ALL_LINEAR_TRIGGERS = [
    WorkflowNodeDetails(
        unique_identifier="trigger-linear-issue-created",
        name="Linear Issue Created",
        description="Triggers when a new issue is created in Linear",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.LINEAR,
        type=NodeType.TRIGGER_LINEAR_ISSUE_CREATED,
        icon="alert-circle",
        color="#5e6ad2",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "integration_id": {
                    "type": "string",
                    "title": "Integration ID",
                    "description": "Linear integration ID",
                },
                "unique_identifier": {
                    "type": "string",
                    "title": "Unique Identifier",
                    "description": "Unique identifier for the Linear integration",
                },
                "organization_id": {
                    "type": "string",
                    "title": "Organization ID",
                    "description": "Linear organization ID (used as trigger hash)",
                },
            },
            "required": ["integration_id", "organization_id"],
        },
    ),
]

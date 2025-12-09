"""
Sentry trigger node types.

This module defines Sentry-specific trigger nodes for Sentry events
like issue creation.
"""

from typing import Literal, Union, Optional
from pydantic import BaseModel, Field
from app.core.nodes.base import NodeType, NodeCategory, NodeGroup, WorkflowNodeDetails
from app.core.nodes.triggers.base import TriggerNode
from app.core.nodes.data_models import SentryIssueCreatedTriggerData


# Sentry trigger node classes
class SentryTriggerBase(TriggerNode):
    """Base class for Sentry triggers."""

    group: NodeGroup = NodeGroup.SENTRY


class SentryIssueCreatedTrigger(SentryTriggerBase):
    """Sentry issue created trigger node."""

    type: NodeType = NodeType.TRIGGER_SENTRY_ISSUE_CREATED
    data: SentryIssueCreatedTriggerData


SentryTrigger = Union[SentryIssueCreatedTrigger]

# Node definitions for UI
ALL_SENTRY_TRIGGERS = [
    WorkflowNodeDetails(
        unique_identifier="trigger-sentry-issue-created",
        name="Sentry Issue Created",
        description="Triggers when a new issue is created in Sentry",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.SENTRY,
        type=NodeType.TRIGGER_SENTRY_ISSUE_CREATED,
        icon="alert-triangle",
        color="#f43f5e",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "title": "Project ID (Optional)"}
            },
        },
    ),
]

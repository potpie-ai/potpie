"""
Webhook trigger node types and processor.

This module defines webhook-specific trigger nodes for handling generic webhook events.
"""

from typing import Literal, Union
from pydantic import BaseModel, Field
from app.core.executions.event import Event
from app.core.executions.state import NodeExecutionResult, NodeExecutionStatus
from app.core.nodes.base import NodeType, NodeCategory, NodeGroup, WorkflowNodeDetails
from app.core.nodes.triggers.base import TriggerNode
from app.core.nodes.data_models import WebhookTriggerData


# Webhook trigger node classes
class WebhookTrigger(TriggerNode):
    """Webhook trigger node."""

    type: NodeType = NodeType.TRIGGER_WEBHOOK
    group: NodeGroup = NodeGroup.DEFAULT
    data: WebhookTriggerData


# Union type for webhook triggers
WebhookTriggerType = Union[WebhookTrigger]


# Node definitions for UI
ALL_WEBHOOK_TRIGGERS = [
    WorkflowNodeDetails(
        unique_identifier="trigger-webhook",
        name="Webhook Trigger",
        description="Triggers when a webhook is received",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.DEFAULT,
        type=NodeType.TRIGGER_WEBHOOK,
        icon="webhook",
        color="#6366f1",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "hash": {"type": "string", "title": "Trigger Hash"},
            },
            "required": ["hash"],
        },
    ),
]

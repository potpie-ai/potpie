"""
Human-in-the-Loop (HITL) data models and types.

This module defines data models for HITL requests and responses,
including request storage, response handling, and status tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.utils.datetime_utils import utc_now


class HITLRequestStatus(str, Enum):
    """Status of a HITL request."""

    PENDING = "pending"
    RESPONDED = "responded"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class HITLNodeType(str, Enum):
    """Type of HITL node."""

    APPROVAL = "approval"
    INPUT = "input"


class HITLRequest(BaseModel):
    """Represents a HITL request waiting for human input."""

    request_id: str = Field(..., description="Unique request identifier")
    execution_id: str = Field(..., description="Workflow execution ID")
    node_id: str = Field(..., description="Node ID that requires human input")
    iteration: int = Field(..., description="Node execution iteration")
    node_type: HITLNodeType = Field(..., description="Type of HITL node")
    message: str = Field(..., description="Message/instructions for the user")
    fields: Optional[List[Dict[str, Any]]] = Field(
        None, description="Input fields for input nodes"
    )
    timeout_at: datetime = Field(..., description="When the request expires")
    channel: str = Field(default="web", description="Channel used for notification")
    created_at: datetime = Field(default_factory=utc_now, description="Request creation time")
    status: HITLRequestStatus = Field(
        default=HITLRequestStatus.PENDING, description="Current request status"
    )
    # Node-specific configuration
    approvers: Optional[List[str]] = Field(
        None, description="List of approver user IDs (for approval nodes)"
    )
    assignee: Optional[str] = Field(
        None, description="User ID assigned to provide input (for input nodes)"
    )
    timeout_action: Optional[str] = Field(
        None, description="Action to take on timeout (approve/reject/fail)"
    )

    model_config = {
        "use_enum_values": True,
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }


class HITLResponse(BaseModel):
    """Represents a user response to a HITL request."""

    request_id: str = Field(..., description="Request ID this response is for")
    execution_id: str = Field(..., description="Workflow execution ID")
    node_id: str = Field(..., description="Node ID that received the response")
    user_id: str = Field(..., description="User ID who submitted the response")
    response_data: Dict[str, Any] = Field(
        ..., description="Response data (boolean for approval, object for input)"
    )
    timestamp: datetime = Field(default_factory=utc_now, description="Response submission time")
    comment: Optional[str] = Field(None, description="Optional comment/notes")
    selected_path: Optional[str] = Field(
        None, description="Selected path identifier (for path selection)"
    )
    channel: str = Field(default="web", description="Channel used for response")

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
    }


class HITLRequestWithContext(BaseModel):
    """HITL request with additional context information."""

    request: HITLRequest
    workflow_id: str
    workflow_title: Optional[str] = None
    previous_node_result: Optional[str] = None
    execution_variables: Dict[str, str] = Field(default_factory=dict)
    time_remaining_seconds: Optional[int] = None


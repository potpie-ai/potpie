"""
Event Bus Schemas

Pydantic models for event bus data structures.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EventMetadata(BaseModel):
    """Metadata for events."""

    source: str = Field(..., description="Source of the event")
    version: str = Field(default="1.0", description="Event schema version")
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for tracing"
    )
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    project_id: Optional[str] = Field(None, description="Project ID if applicable")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Event timestamp"
    )
    retry_count: int = Field(default=0, description="Number of retries attempted")


class WebhookEvent(BaseModel):
    """Schema for webhook events from integrations."""

    event_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique event ID"
    )
    integration_id: str = Field(..., description="ID of the integration")
    integration_type: str = Field(
        ..., description="Type of integration (linear, sentry, etc.)"
    )
    event_type: str = Field(..., description="Type of webhook event")
    event_source: str = Field(
        ..., description="Source of the event (e.g., 'linear', 'sentry')"
    )
    payload: Dict[str, Any] = Field(..., description="Webhook payload data")
    headers: Optional[Dict[str, str]] = Field(
        None, description="HTTP headers from webhook"
    )
    source_ip: Optional[str] = Field(None, description="Source IP address")
    received_at: datetime = Field(
        default_factory=datetime.utcnow, description="When webhook was received"
    )
    metadata: EventMetadata = Field(
        default_factory=EventMetadata, description="Event metadata"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() + "Z"}


class CustomEvent(BaseModel):
    """Schema for custom events."""

    event_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique event ID"
    )
    topic: str = Field(..., description="Topic/queue name")
    event_type: str = Field(..., description="Type of event")
    event_source: str = Field(
        ..., description="Source of the event (e.g., 'custom', 'system')"
    )
    data: Dict[str, Any] = Field(..., description="Event data payload")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When event was created"
    )
    metadata: EventMetadata = Field(
        default_factory=EventMetadata, description="Event metadata"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() + "Z"}


class Event(BaseModel):
    """Generic event schema."""

    event_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique event ID"
    )
    event_type: str = Field(..., description="Type of event")
    event_source: str = Field(..., description="Source of the event")
    topic: str = Field(..., description="Topic/queue name")
    data: Dict[str, Any] = Field(..., description="Event data")
    metadata: EventMetadata = Field(
        default_factory=EventMetadata, description="Event metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When event was created"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() + "Z"}

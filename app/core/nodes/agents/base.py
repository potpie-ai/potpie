"""
Base agent node models.

This module defines the base classes and data models for agent nodes.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

from app.core.nodes.base import (
    NodeType,
    NodeCategory,
    NodeGroup,
    WorkflowNodeBase,
)


# Data models are now imported from data_models.py


# Agent node classes
class AgentNodeBase(WorkflowNodeBase):
    """Base class for agent nodes."""

    category: NodeCategory = NodeCategory.AGENT
    timeout_seconds: int = Field(
        default=600, description="Execution timeout in seconds"
    )
    retry_count: int = Field(default=3, description="Number of retry attempts")


# Union type for all agent nodes
AgentNode = Union["AgentNodeBase"]

"""
Base flow control node models.

This module defines the base classes and data models for flow control nodes.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field

from app.core.nodes.base import (
    NodeType,
    NodeCategory,
    NodeGroup,
    WorkflowNodeBase,
)


# Data models are now imported from data_models.py


# Flow control node classes
class FlowControlNodeBase(WorkflowNodeBase):
    """Base class for flow control nodes."""

    category: NodeCategory = NodeCategory.FLOW_CONTROL
    group: NodeGroup = NodeGroup.DEFAULT


# Union type for all flow control nodes
FlowControlNode = Union["FlowControlNodeBase"]

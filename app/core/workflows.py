from typing import Final, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum

from app.core.nodes import WorkflowNode
from app.core.nodes.factory import NodeFactory


""" Core data models for workflows, including workflow definitions, nodes and graph representation """


class WorkflowGraph(BaseModel):
    id: str
    workflow_id: str
    nodes: Dict[str, WorkflowNode]  # nodeId -> WorkflowNode
    adjacency_list: Dict[str, List[str]]  # adjacency list: nodeId -> [nextNodeIds]
    created_at: str
    updated_at: str

    model_config = {
        "extra": "ignore",
        "populate_by_name": True,
        "alias_generator": lambda x: x,  # Use original field names for serialization
    }

    @model_validator(mode="before")
    @classmethod
    def validate_nodes(cls, data):
        """Custom validator to properly deserialize nodes using NodeFactory."""
        if isinstance(data, dict) and "nodes" in data:
            nodes_data = data["nodes"]
            if isinstance(nodes_data, dict):
                # Check if nodes are already deserialized (have 'type' attribute)
                if nodes_data and any(
                    isinstance(v, dict) and "type" in v for v in nodes_data.values()
                ):
                    # Use NodeFactory to properly deserialize nodes
                    deserialized_nodes = NodeFactory.deserialize_nodes(nodes_data)
                    data["nodes"] = deserialized_nodes
        return data


class Workflow(BaseModel):
    id: str
    title: str
    description: str
    created_by: str
    created_at: str
    updated_at: str
    is_paused: bool
    version: str  # keep this as integer for now, it can be incremented for each update and can compare numbers

    graph: WorkflowGraph
    variables: Dict[str, str] = Field(default_factory=dict)

    model_config = {
        "extra": "ignore",
        "populate_by_name": True,
        "alias_generator": lambda x: x,  # Use original field names for serialization
    }


""" Repository for workflow management, including creation, updates, and retrieval of workflows. """


class CreateWorkflow(BaseModel):
    title: str
    description: str
    created_by: str
    # Graph structure
    nodes: Dict[str, WorkflowNode] = Field(default_factory=dict)
    adjacency_list: Dict[str, List[str]] = Field(
        default_factory=dict
    )  # adjacency list: nodeId -> [nextNodeIds]
    variables: Dict[str, str] = Field(default_factory=dict)


class UpdateWorkflow(BaseModel):
    title: str
    description: str
    # Graph structure updates
    nodes: Optional[Dict[str, WorkflowNode]] = None
    adjacency_list: Optional[Dict[str, List[str]]] = None
    variables: Optional[Dict[str, str]] = None


class WorkflowsStore(ABC):
    """Abstract adapter for workflow store."""

    @abstractmethod
    async def create_workflow(self, input: CreateWorkflow) -> Workflow:
        """Create a new workflow in the store."""
        pass

    @abstractmethod
    async def update_workflow(
        self, workflow_id: str, update: UpdateWorkflow
    ) -> Workflow:
        """Update an existing workflow in the store."""
        pass

    @abstractmethod
    async def delete_workflow(self, workflow_id: str):
        """Delete a workflow from the store."""
        pass

    @abstractmethod
    async def get_workflow(self, workflow_id: str) -> Workflow:
        """Retrieve a workflow by its ID."""
        pass

    @abstractmethod
    async def list_workflows(self, user_id) -> List[Workflow]:
        """List all workflows in the store."""
        pass

    @abstractmethod
    async def get_workflow_by_hash(self, workflow_hash: str) -> Workflow:
        """Retrieve a workflow by its hash."""
        pass

    @abstractmethod
    async def get_workflow_by_trigger(self, trigger_id: str) -> Workflow:
        """Retrieve workflows by trigger ID."""
        pass

    @abstractmethod
    async def set_pause(self, workflow_id: str, pause: bool):
        """set pause flag"""
        pass


class WorkflowNotFound(Exception):
    """Exception raised when a workflow is not found in the store."""

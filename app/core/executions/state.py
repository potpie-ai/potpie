from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Union, Optional
from pydantic import BaseModel, Field
from app.utils.datetime_utils import utc_now, ensure_utc, to_utc_isoformat

# Default maximum iterations to prevent infinite loops
DEFAULT_MAX_ITERATIONS = 15


class NodeExecutionStatus(str, Enum):
    """Enum representing the status of an node execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"


class NodeExecutionContext(BaseModel):
    """
    Represents the state of a node execution within a workflow. This is passed as input to the node execution.
    current_iteration + node_id together can be used as deduplication/idempotency key for the execution.
    """

    queued_at: datetime
    workflow_snapshot: Any  # Workflow
    event: Any  # Event
    execution_variables: Dict[str, str]
    execution_id: str
    previous_node_result: str
    current_iteration: int
    current_node: Any  # WorkflowNode
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        description="Maximum iterations to prevent infinite loops",
    )
    predecessor_node_id: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        },
    }

    def model_dump_for_celery(self, **kwargs):
        """Custom serialization method for Celery that handles complex objects properly."""
        # Use the standard model_dump but with json mode
        data = self.model_dump(mode="json", **kwargs)

        # Convert datetime back to string if it wasn't handled by json_encoders
        if isinstance(data.get("queued_at"), datetime):
            data["queued_at"] = to_utc_isoformat(data["queued_at"])

        return data

    @classmethod
    def model_validate_for_celery(cls, data):
        """Custom deserialization method for Celery that handles complex objects properly."""
        # Handle datetime conversion
        if isinstance(data.get("queued_at"), str):
            data = data.copy()
            data["queued_at"] = ensure_utc(data["queued_at"])

        # Reconstruct complex objects
        data = data.copy()

        # Import dependencies for reconstruction
        from app.core.nodes.factory import NodeFactory
        from app.core.executions.event import Event
        from app.core.workflows import Workflow, WorkflowGraph

        # Reconstruct current_node
        if isinstance(data.get("current_node"), dict):
            reconstructed_node = NodeFactory.deserialize_node(data["current_node"])
            if reconstructed_node:
                data["current_node"] = reconstructed_node

        # Reconstruct event
        if isinstance(data.get("event"), dict):
            event_data = data["event"].copy()
            if "time" in event_data and isinstance(event_data["time"], str):
                event_data["time"] = ensure_utc(event_data["time"])
            data["event"] = Event(**event_data)

        # Reconstruct workflow_snapshot
        if isinstance(data.get("workflow_snapshot"), dict):
            workflow_data = data["workflow_snapshot"].copy()

            # Reconstruct the graph if it exists
            if "graph" in workflow_data and isinstance(workflow_data["graph"], dict):
                graph_data = workflow_data["graph"].copy()

                # Reconstruct nodes in the graph
                if "nodes" in graph_data and isinstance(graph_data["nodes"], dict):
                    reconstructed_nodes = {}
                    for node_id, node_data in graph_data["nodes"].items():
                        if isinstance(node_data, dict):
                            reconstructed_node = NodeFactory.deserialize_node(node_data)
                            if reconstructed_node:
                                reconstructed_nodes[node_id] = reconstructed_node
                            else:
                                reconstructed_nodes[node_id] = node_data
                        else:
                            reconstructed_nodes[node_id] = node_data
                    graph_data["nodes"] = reconstructed_nodes

                workflow_data["graph"] = WorkflowGraph(**graph_data)

            data["workflow_snapshot"] = Workflow(**workflow_data)

        return cls.model_validate(data)


class NodeExecutionResult(BaseModel):
    status: NodeExecutionStatus
    output: Any
    execution_variables: Optional[Dict[str, str]] = Field(
        default=None,
        description="Variables to add to the execution context for subsequent nodes",
    )

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.executions.event import Event
from app.core.executions.state import NodeExecutionStatus
from app.core.executions.hitl import HITLRequest, HITLResponse


logger = logging.getLogger(__name__)


class NodeExecutionLog(BaseModel):
    status: NodeExecutionStatus
    timestamp: datetime
    details: str


class NodeExecution(BaseModel):
    node_exec_id: str

    node_id: str
    iteration: int

    status: NodeExecutionStatus
    start_time: datetime
    end_time: datetime

    logs: List[NodeExecutionLog]

    # Predecessor node information for building execution trees
    predecessor_node_id: Optional[str] = None


class WorkflowExecutionStatus(str, Enum):
    """Enum representing status of Workflow Execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_FAILED = "partially_failed"
    CANCELLED = "cancelled"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    SKIPPED = "skipped"


class WorkflowExecution(BaseModel):
    event: Event
    wf_exec_id: str
    wf_id: str
    status: WorkflowExecutionStatus
    start_time: datetime
    end_time: datetime

    node_executions: List[NodeExecution]


class ExecutionLogStore(ABC):
    """Abstract adapter for execution log store."""

    @abstractmethod
    async def list_wf_execution_for_workflow(
        self, wf_id: str
    ) -> List[WorkflowExecution]:
        """List all execution for a given workflow"""
        pass

    @abstractmethod
    async def get_wf_execution_by_id(self, exec_id: str) -> WorkflowExecution:
        """Get execution by id"""
        pass

    @abstractmethod
    async def append_log(
        self,
        exec_id: str,
        node_id: str,
        iteration: int,
        log: NodeExecutionLog,
        event: Any = None,
        predecessor_node_id: Optional[str] = None,
    ):
        """update the log for node execution"""

    @abstractmethod
    async def create_workflow_execution(
        self, workflow_execution: WorkflowExecution
    ) -> str:
        """Create a new workflow execution record."""
        pass

    @abstractmethod
    async def update_workflow_execution_status(
        self, exec_id: str, status: WorkflowExecutionStatus
    ) -> None:
        """Update the status of a workflow execution."""
        pass

    # HITL-related methods
    @abstractmethod
    async def create_hitl_request(self, request: HITLRequest) -> str:
        """Create a new HITL request."""
        pass

    @abstractmethod
    async def get_hitl_request(
        self, execution_id: str, node_id: str, iteration: int
    ) -> Optional[HITLRequest]:
        """Get a HITL request by execution ID, node ID, and iteration."""
        pass

    @abstractmethod
    async def get_hitl_request_by_id(self, request_id: str) -> Optional[HITLRequest]:
        """Get a HITL request by request ID."""
        pass

    @abstractmethod
    async def list_pending_hitl_requests(
        self, execution_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[HITLRequest]:
        """List pending HITL requests, optionally filtered by execution ID or user ID."""
        pass

    @abstractmethod
    async def update_hitl_request_status(
        self, request_id: str, status: str
    ) -> None:
        """Update the status of a HITL request."""
        pass

    @abstractmethod
    async def create_hitl_response(self, response: HITLResponse) -> str:
        """Create a HITL response."""
        pass

    @abstractmethod
    async def get_hitl_response(self, request_id: str) -> Optional[HITLResponse]:
        """Get a HITL response by request ID."""
        pass


class NodeExecutionKnownError(Exception):
    """Exception raised when node execution fails for known reason"""


class ExecutionContext(BaseModel):
    variables: Dict[str, Any] = Field(default_factory=dict)
    execution_id: str
    workflow_id: str


class ExecutionResult(BaseModel):
    status: str
    execution_id: str
    error: Optional[str] = None

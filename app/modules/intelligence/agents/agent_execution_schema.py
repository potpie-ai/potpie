from typing import List, Optional
from pydantic import BaseModel, Field


class AgentExecuteRequest(BaseModel):
    """Request payload for direct agent execution"""

    query: str = Field(..., description="The query/prompt to send to the agent")
    project_id: str = Field(..., description="The project ID for context")
    node_ids: Optional[List[str]] = Field(
        default=None, description="Optional list of node IDs for additional context"
    )
    additional_context: Optional[str] = Field(
        default="", description="Optional additional context for the agent"
    )


class AgentExecuteStartResponse(BaseModel):
    """Immediate response when execution is queued"""

    execution_id: str = Field(..., description="Unique identifier for this execution")
    status: str = Field(default="queued", description="Current status of the execution")


class AgentExecutionResultResponse(BaseModel):
    """Response containing the full agent execution result"""

    execution_id: str = Field(..., description="Unique identifier for this execution")
    status: str = Field(
        ..., description="Current status: 'queued' | 'running' | 'completed' | 'failed'"
    )
    response: Optional[str] = Field(
        default=None, description="Full response from the agent"
    )
    citations: Optional[List[str]] = Field(
        default=None, description="List of citations/references"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed"
    )

from typing import Any, Dict, List

from pydantic import BaseModel


class ToolRequest(BaseModel):
    tool_id: str
    params: Dict[str, Any]


class ToolResponse(BaseModel):
    results: Any


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool


class ToolInfo(BaseModel):
    id: str
    name: str
    description: str
    parameters: List[ToolParameter]

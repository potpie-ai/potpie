from typing import Any, Dict, Union

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
    description: Union[str, tuple]


class ToolInfoWithParameters(BaseModel):
    name: str
    description: Union[str, tuple]
    args_schema: dict

from typing import Any, Dict, Union, Optional, Type, Callable, Awaitable

from pydantic import BaseModel, ConfigDict


class OnyxTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    func: Callable[..., Any]
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        return self.func(*args, **kwargs)


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

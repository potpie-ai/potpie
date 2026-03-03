from typing import Any, Awaitable, Callable, Dict, Optional, Type, Union

from pydantic import BaseModel, ConfigDict


ArgsSchemaType = Union[Type[BaseModel], Dict[str, Any]]


class OnyxTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    description: str
    args_schema: Optional[ArgsSchemaType] = None
    func: Callable[..., Any]
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None

    @classmethod
    def from_function(
        cls,
        *,
        func: Callable[..., Any],
        name: str,
        description: str = "",
        args_schema: Optional[ArgsSchemaType] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> "OnyxTool":
        """Construct an OnyxTool from sync/async callables.

        Mirrors the subset of StructuredTool.from_function used by this codebase.
        """
        if not callable(func):
            raise TypeError("func must be callable")
        if coroutine is not None and not callable(coroutine):
            raise TypeError("coroutine must be callable when provided")

        return cls(
            name=name,
            description=description or "",
            args_schema=args_schema,
            func=func,
            coroutine=coroutine,
        )

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

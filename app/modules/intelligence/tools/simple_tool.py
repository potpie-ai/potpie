"""
SimpleTool: A lightweight, LangChain-free replacement for langchain_core.tools.StructuredTool.

Provides the same interface used throughout this codebase:
  - Constructor: SimpleTool(name, description, func, args_schema, coroutine)
  - Factory:     SimpleTool.from_function(func, name, description, args_schema, coroutine)
  - invoke():    SimpleTool.invoke(input_dict) for direct synchronous execution

The wrap_structured_tools() helper in multi_agent/utils/tool_utils.py converts
SimpleTool instances to pydantic-ai Tool objects for agent execution.
"""

import inspect
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel


def _invoke_func(func: Callable, args_schema: Any, kwargs: dict) -> Any:
    """Invoke a tool function with a kwargs dict, handling the single-Pydantic-model
    argument pattern used by many tools (func(input_data: SomeModel))."""
    if not (isinstance(args_schema, type) and issubclass(args_schema, BaseModel)):
        return func(**kwargs)
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) == 1:
        annotation = params[0].annotation
        if (
            annotation is not inspect.Parameter.empty
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            return func(annotation(**kwargs))
    return func(**kwargs)


class SimpleTool:
    """Drop-in replacement for langchain_core.tools.StructuredTool.

    Stores name, description, func (sync), args_schema, and coroutine (async).
    The wrap_structured_tools() utility converts these to pydantic-ai Tool objects.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Optional[Callable] = None,
        args_schema: Optional[Any] = None,
        coroutine: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema
        self.coroutine = coroutine

    def invoke(self, input: dict, **kwargs: Any) -> Any:
        """Invoke the tool synchronously with a kwargs dict."""
        if not callable(self.func):
            raise TypeError(f"Tool '{self.name}' has no callable func")
        return _invoke_func(self.func, self.args_schema, input)

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        coroutine: Optional[Callable] = None,
        **kwargs: Any,
    ) -> "SimpleTool":
        """Create a SimpleTool from a function, mirroring StructuredTool.from_function."""
        if name is None and func is not None:
            name = func.__name__
        if description is None and func is not None:
            description = func.__doc__ or ""
        return cls(
            name=name or "",
            description=description or "",
            func=func,
            args_schema=args_schema,
            coroutine=coroutine,
        )

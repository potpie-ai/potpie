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
import logging
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _has_single_pydantic_param(func: Callable) -> Optional[type]:
    """Return the Pydantic model class if *func* accepts exactly one BaseModel parameter."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) != 1:
        return None
    annotation = params[0].annotation
    if (
        annotation is not inspect.Parameter.empty
        and isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
    ):
        return annotation
    return None


def _invoke_func(func: Callable, args_schema: Any, kwargs: Dict[str, Any]) -> Any:
    """Invoke a tool function with a kwargs dict, handling the single-Pydantic-model
    argument pattern used by many tools (func(input_data: SomeModel))."""
    if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
        model_cls = _has_single_pydantic_param(func)
        if model_cls is not None:
            return func(model_cls(**kwargs))
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
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema
        self.coroutine = coroutine

    def invoke(self, tool_input: Dict[str, Any], **kwargs: Any) -> Any:
        """Invoke the tool synchronously with a kwargs dict."""
        if not callable(self.func):
            raise TypeError(f"Tool '{self.name}' has no callable func")
        if tool_input is None:
            tool_input = {}
        return _invoke_func(self.func, self.args_schema, tool_input)

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        coroutine: Optional[Callable] = None,
    ) -> "SimpleTool":
        """Create a SimpleTool from a function, mirroring StructuredTool.from_function."""
        resolved_name = name
        resolved_description = description
        if resolved_name is None and func is not None:
            resolved_name = func.__name__
        if resolved_description is None and func is not None:
            resolved_description = func.__doc__ or ""
        return cls(
            name=resolved_name or "",
            description=resolved_description or "",
            func=func,
            args_schema=args_schema,
            coroutine=coroutine,
        )

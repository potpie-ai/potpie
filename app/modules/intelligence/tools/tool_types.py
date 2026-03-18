"""Lightweight tool and message type definitions.

Replaces langchain_core.tools.StructuredTool and langchain_core.messages.*
with minimal, dependency-free equivalents that preserve the same interface.
"""

import inspect
from typing import Any, Callable, Optional, Type, Union

from pydantic import BaseModel


class PotpieTool:
    """Drop-in replacement for langchain_core.tools.StructuredTool.

    Provides the same interface (.name, .description, .func, .args_schema,
    .coroutine, .invoke(), .from_function()) without pulling in the full
    langchain_core dependency.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        args_schema: Optional[Union[Type[BaseModel], dict]] = None,
        coroutine: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema
        self.coroutine = coroutine

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Union[Type[BaseModel], dict]] = None,
        coroutine: Optional[Callable] = None,
        **kwargs: Any,
    ) -> "PotpieTool":
        """Create a PotpieTool from a callable (mirrors StructuredTool.from_function)."""
        if func is None and coroutine is None:
            raise ValueError("Either func or coroutine must be provided")

        effective_func = func or coroutine
        if name is None:
            name = getattr(effective_func, "__name__", "tool")
        if description is None:
            description = inspect.getdoc(effective_func) or ""

        # If only coroutine provided, create a sync wrapper
        if func is None and coroutine is not None:
            import asyncio

            def _sync_wrapper(*args: Any, **kw: Any) -> Any:
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        fut = pool.submit(asyncio.run, coroutine(*args, **kw))
                        return fut.result()
                except RuntimeError:
                    return asyncio.run(coroutine(*args, **kw))

            func = _sync_wrapper

        return cls(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema,
            coroutine=coroutine,
        )

    def invoke(self, input_data: Union[dict, Any], **kwargs: Any) -> Any:
        """Invoke the tool synchronously with dict input."""
        if isinstance(input_data, dict):
            return self._call(**input_data)
        return self.func(input_data)

    def _call(self, **kwargs: Any) -> Any:
        """Call func, adapting for single-BaseModel-arg signature if needed."""
        if self.args_schema is not None and isinstance(self.args_schema, type) and issubclass(self.args_schema, BaseModel):
            try:
                sig = inspect.signature(self.func)
                params = [p for p in sig.parameters.values() if p.name != "self"]
                if len(params) == 1:
                    annotation = params[0].annotation
                    if (
                        annotation is not inspect.Parameter.empty
                        and isinstance(annotation, type)
                        and issubclass(annotation, BaseModel)
                    ):
                        return self.func(annotation(**kwargs))
            except (TypeError, ValueError):
                pass
        return self.func(**kwargs)


# ---------------------------------------------------------------------------
# Minimal message types — replaces langchain_core.messages
# ---------------------------------------------------------------------------


class BaseMessage:
    """Base class for chat history messages."""

    def __init__(self, content: str) -> None:
        self.content = content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content={self.content!r})"


class HumanMessage(BaseMessage):
    """Represents a message from the human/user."""

    pass


class AIMessage(BaseMessage):
    """Represents a message from the AI assistant."""

    pass

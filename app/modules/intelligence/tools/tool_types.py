"""Lightweight tool and message type definitions.

Replaces langchain_core.tools.StructuredTool and langchain_core.messages.*
with minimal, dependency-free equivalents that preserve the same interface.
"""

import asyncio
import concurrent.futures
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
        **_kwargs: Any,  # accepted for API compatibility; not used
    ) -> "PotpieTool":
        """Create a PotpieTool from a callable (mirrors StructuredTool.from_function)."""
        if func is None and coroutine is None:
            raise ValueError("Either func or coroutine must be provided")

        effective_func = func or coroutine
        if name is None:
            name = getattr(effective_func, "__name__", "tool")
        if description is None:
            description = inspect.getdoc(effective_func) or ""

        # If only coroutine provided, create a sync wrapper that detects whether
        # a running event loop exists and adapts accordingly.
        if func is None and coroutine is not None:

            def _sync_wrapper(*args: Any, **kw: Any) -> Any:
                running_loop: Optional[asyncio.AbstractEventLoop] = None
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass  # No running loop — safe to call asyncio.run directly.

                if running_loop is not None:
                    # Inside a running loop — spin up a thread with its own loop.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(asyncio.run, coroutine(*args, **kw)).result()
                return asyncio.run(coroutine(*args, **kw))

            func = _sync_wrapper

        return cls(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema,
            coroutine=coroutine,
        )

    def invoke(self, input_data: Union[dict, Any], **_kwargs: Any) -> Any:
        """Invoke the tool synchronously with dict input.

        ``**_kwargs`` are accepted for API-compatibility with LangChain's
        Runnable interface (e.g. ``config``, ``run_manager``) but are not
        forwarded to the underlying function to avoid polluting tool call-sites
        that do not expect them.
        """
        if isinstance(input_data, dict):
            return self._call(**input_data)
        return self.func(input_data)

    def _call(self, **kwargs: Any) -> Any:
        """Call func, adapting for single-BaseModel-arg signature if needed."""
        schema = self.args_schema
        is_basemodel_schema = (
            schema is not None
            and isinstance(schema, type)
            and issubclass(schema, BaseModel)
        )
        if is_basemodel_schema:
            try:
                sig = inspect.signature(self.func)
                params = [p for p in sig.parameters.values() if p.name != "self"]
                if len(params) == 1:
                    annotation = params[0].annotation
                    if annotation is inspect.Parameter.empty:
                        # Unannotated single param — build from args_schema.
                        return self.func(schema(**kwargs))
                    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
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


class AIMessage(BaseMessage):
    """Represents a message from the AI assistant."""

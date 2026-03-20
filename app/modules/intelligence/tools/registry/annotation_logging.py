"""Phase 4: Log tool behavioral annotations on every tool call for audits and observability.

Single module for (1) extracting annotation dict from ToolMetadata and (2) wrapping
StructuredTools for pre-invoke logging (direct path). Discovery path calls
get_annotations_for_logging from discovery_tools._execute_tool.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel
from app.modules.intelligence.tools.simple_tool import SimpleTool as StructuredTool
from app.modules.intelligence.tools.simple_tool import _invoke_func

from app.modules.utils.logger import setup_logger

if TYPE_CHECKING:
    from app.modules.intelligence.tools.registry.registry import ToolRegistry
    from app.modules.intelligence.tools.registry.schema import ToolMetadata

logger = setup_logger(__name__)


def _invoke_inner_tool(tool: Any, kwargs: Dict[str, Any]) -> Any:
    """Invoke the inner tool's func with the given kwargs.

    Delegates to the shared _invoke_func helper for the single-Pydantic-model
    argument pattern. Falls back to tool.invoke() if no callable func is present.
    """
    func = getattr(tool, "func", None)
    if not callable(func):
        if hasattr(tool, "invoke") and callable(tool.invoke):
            return tool.invoke(kwargs)
        raise TypeError(f"Tool {getattr(tool, 'name', tool)} has no invoke or func")

    args_schema = getattr(tool, "args_schema", None)
    return _invoke_func(func, args_schema, kwargs)


# Annotation keys we log (only those set on metadata)
_ANNOTATION_KEYS = ("read_only", "destructive", "idempotent", "requires_confirmation")


def get_annotations_for_logging(
    metadata: Optional["ToolMetadata"],
) -> Dict[str, Any]:
    """
    Return a dict of annotation key -> value for logging. Only includes keys
    that are set (not None). If metadata is None, returns {}.
    Best-effort: never raises; used by both direct and discovery paths.
    """
    if metadata is None:
        return {}
    out: Dict[str, Any] = {}
    for key in _ANNOTATION_KEYS:
        val = getattr(metadata, key, None)
        if val is not None:
            out[key] = val
    return out


def wrap_tool_for_annotation_logging(
    tool: StructuredTool,
    registry: "ToolRegistry",
) -> StructuredTool:
    """
    Return a new StructuredTool that logs annotations then delegates to the inner tool.

    On invoke: looks up metadata by tool.name, logs annotations via
    get_annotations_for_logging, then invokes the inner tool. If registry lookup
    fails, logs with empty annotations dict and still invokes (best-effort).
    """
    raw_schema = getattr(tool, "args_schema", None)
    # LangChain StructuredTool accepts only BaseModel subclass or JSON schema dict
    args_schema = None
    if isinstance(raw_schema, type) and issubclass(raw_schema, BaseModel):
        args_schema = raw_schema
    elif isinstance(raw_schema, dict):
        args_schema = raw_schema

    def _invoke(**kwargs: Any) -> Any:
        meta = registry.get_metadata(tool.name)
        ann = get_annotations_for_logging(meta)
        logger.info(
            "tool_call_annotations tool=%s annotations=%s",
            tool.name,
            ann,
        )
        return _invoke_inner_tool(tool, kwargs)

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description or "",
        func=_invoke,
        args_schema=args_schema,
    )

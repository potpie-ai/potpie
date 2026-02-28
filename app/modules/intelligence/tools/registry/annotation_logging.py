"""Phase 4: Log tool behavioral annotations on every tool call for audits and observability.

Single module for (1) extracting annotation dict from ToolMetadata and (2) wrapping
OnyxTools for pre-invoke logging (direct path). Discovery path calls
get_annotations_for_logging from discovery_tools._execute_tool.
"""

import inspect
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel
from app.modules.intelligence.tools.tool_schema import OnyxTool

from app.modules.utils.logger import setup_logger

if TYPE_CHECKING:
    from app.modules.intelligence.tools.registry.registry import ToolRegistry
    from app.modules.intelligence.tools.registry.schema import ToolMetadata

logger = setup_logger(__name__)


def _invoke_inner_tool(tool: Any, kwargs: Dict[str, Any]) -> Any:
    """Invoke the inner tool's func with the given kwargs.

    If the tool's func takes a single parameter typed as a BaseModel (e.g. input_data: SearchTextInput),
    we build that model from kwargs and call func(model). Otherwise we call func(**kwargs).
    This matches the adaptation done in tool_utils._adapt_func_for_from_schema so that tools
    wrapped for annotation logging work when the executor passes **kwargs from the schema.
    """
    func = getattr(tool, "func", None)
    if not callable(func):
        if hasattr(tool, "invoke") and callable(tool.invoke):
            return tool.invoke(kwargs)
        raise TypeError(f"Tool {getattr(tool, 'name', tool)} has no invoke or func")

    raw_schema = getattr(tool, "args_schema", None)
    if not (isinstance(raw_schema, type) and issubclass(raw_schema, BaseModel)):
        return func(**kwargs)

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) != 1:
        return func(**kwargs)
    param = params[0]
    annotation = param.annotation
    if annotation is inspect.Parameter.empty:
        return func(**kwargs)
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        return func(**kwargs)

    model_cls = annotation
    return func(model_cls(**kwargs))


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
    tool: OnyxTool,
    registry: "ToolRegistry",
) -> OnyxTool:
    """
    Return a new OnyxTool that logs annotations then delegates to the inner tool.

    On invoke: looks up metadata by tool.name, logs annotations via
    get_annotations_for_logging, then invokes the inner tool. If registry lookup
    fails, logs with empty annotations dict and still invokes (best-effort).
    """
    raw_schema = getattr(tool, "args_schema", None)
    # Onyx OnyxTool accepts only BaseModel subclass or JSON schema dict
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

    return OnyxTool.from_function(
        name=tool.name,
        description=tool.description or "",
        func=_invoke,
        args_schema=args_schema,
    )

"""Phase 3: Discovery meta-tools (search_tools, describe_tool, execute_tool) for deferred tool loading.

Builds three Onyx OnyxTools scoped to an allow-list. The agent receives
these instead of the full tool list when use_tool_search_flow=True, then discovers
tools via search_tools, gets full schema via describe_tool, and runs via execute_tool.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from app.modules.intelligence.tools.tool_schema import OnyxTool

from app.modules.intelligence.tools.registry.annotation_logging import (
    get_annotations_for_logging,
)
from app.modules.intelligence.tools.registry.exceptions import RegistryError
from app.modules.utils.logger import setup_logger

if TYPE_CHECKING:
    from app.modules.intelligence.tools.registry.resolver import ToolResolver

logger = setup_logger(__name__)

# Max length for short_description in discovery responses (schema allows 200)
_SHORT_DESC_MAX = 200


def get_discovery_tools(
    resolver: "ToolResolver",
    allow_list_id: str,
    *,
    local_mode: bool = False,
    exclude_embedding_tools: bool = False,
    log_tool_annotations: bool = True,
) -> Tuple[OnyxTool, OnyxTool, OnyxTool]:
    """
    Build the three discovery meta-tools for an agent scoped to the given allow-list.

    The returned tools are bound to the same allow-list and context (local_mode,
    exclude_embedding_tools). Only tools in the resolved allow-list are discoverable
    and executable.

    Args:
        resolver: ToolResolver (registry + tools_provider).
        allow_list_id: Allow-list id (e.g. "supervisor", "execute").
        local_mode: Passed to resolve_allow_list.
        exclude_embedding_tools: Passed to resolve_allow_list.
        log_tool_annotations: If True (Phase 4), log annotations in _execute_tool before invoke.

    Returns:
        Tuple of (search_tools_tool, describe_tool_tool, execute_tool_tool) as
        Onyx OnyxTool instances.
    """
    try:
        allowed_names = resolver.registry.resolve_allow_list(
            allow_list_id,
            local_mode=local_mode,
            exclude_embedding_tools=exclude_embedding_tools,
        )
    except RegistryError as e:
        logger.error(
            "get_discovery_tools: failed to resolve allow_list_id=%s: %s",
            allow_list_id,
            e,
        )
        raise

    def _search_tools(query: Optional[str] = None) -> List[dict]:
        """List available tools with short descriptions for discovery. Optional query reserved for future semantic filtering."""
        result: List[dict] = []
        for name in allowed_names:
            meta = resolver.registry.get_metadata(name)
            if not meta:
                continue
            short_desc = (meta.short_description or meta.description or "").strip()
            if len(short_desc) > _SHORT_DESC_MAX:
                short_desc = short_desc[:_SHORT_DESC_MAX].rstrip()
            # Optional: omit tools missing from ToolService; plan allows "(unavailable)"
            tools_from_service = resolver.tools_provider.get_tools([name])
            if not tools_from_service:
                short_desc = short_desc or "(unavailable)"
            result.append(
                {
                    "name": name,
                    "short_description": short_desc or meta.description,
                    "tier": meta.tier,
                    "category": meta.category,
                }
            )
        return result

    def _describe_tool(name: str) -> dict:
        """Return full description and JSON schema for a tool. Only for tools in the allowed set."""
        if name not in allowed_names:
            return {"error": "Tool not in allowed set"}
        tools = resolver.tools_provider.get_tools(
            [name], exclude_embedding_tools=exclude_embedding_tools
        )
        if not tools:
            return {"error": "Tool not available"}
        tool = tools[0]
        schema = {}
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "schema"):
            try:
                schema = args_schema.schema()
            except Exception as e:
                logger.warning(
                    "describe_tool: could not get schema for %s: %s", name, e
                )
        return {
            "description": getattr(tool, "description", "") or "",
            "schema": schema,
        }

    def _execute_tool(name: str, tool_args: dict) -> Any:
        """Execute a tool by name with the given arguments. Only for tools in the allowed set.
        Parameter named tool_args to avoid Pydantic/Onyx reserved 'args' handling.
        Phase 4: logs tool_call_annotations before invoke when log_tool_annotations=True.
        """
        if name not in allowed_names:
            return {"error": "Tool not in allowed set"}
        tools = resolver.tools_provider.get_tools(
            [name], exclude_embedding_tools=exclude_embedding_tools
        )
        if not tools:
            return {"error": "Tool not available"}
        tool = tools[0]
        if log_tool_annotations:
            meta = resolver.registry.get_metadata(name)
            ann = get_annotations_for_logging(meta)
            logger.info(
                "tool_call_annotations tool=%s annotations=%s",
                name,
                ann,
            )
        try:
            out = tool.invoke(tool_args or {})
            return out
        except Exception as e:
            logger.exception("execute_tool: %s failed", name)
            return {"error": str(e)}

    search_tools_tool = OnyxTool.from_function(
        func=_search_tools,
        name="search_tools",
        description="List available tools with short descriptions. Call this first to discover what tools you can use, then use describe_tool(name) to get full description and argument schema, then execute_tool(name, tool_args) to run a tool. Optional 'query' is reserved for future filtering.",
    )
    describe_tool_tool = OnyxTool.from_function(
        func=_describe_tool,
        name="describe_tool",
        description="Get the full description and JSON argument schema for a tool by name. Only works for tools in your allowed set; call search_tools first to see allowed names.",
    )
    execute_tool_tool = OnyxTool.from_function(
        func=_execute_tool,
        name="execute_tool",
        description="Execute a tool by name with the given arguments (tool_args dict). Only works for tools in your allowed set. Get the schema from describe_tool(name) first to build correct tool_args.",
    )

    return (search_tools_tool, describe_tool_tool, execute_tool_tool)

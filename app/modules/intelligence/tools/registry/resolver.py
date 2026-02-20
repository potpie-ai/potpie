"""Tool resolver: resolves allow-lists via registry and fetches tools from ToolService."""

from typing import TYPE_CHECKING, List

from langchain_core.tools import StructuredTool

from app.modules.intelligence.tools.registry.annotation_logging import (
    wrap_tool_for_annotation_logging,
)
from app.modules.intelligence.tools.registry.exceptions import RegistryError
from app.modules.intelligence.tools.registry.registry import ToolRegistry
from app.modules.utils.logger import setup_logger

if TYPE_CHECKING:
    from app.modules.intelligence.tools.tool_service import ToolService

logger = setup_logger(__name__)


class ToolResolver:
    """
    Facade that holds registry + ToolService.
    Resolves allow-lists to tool names via registry, then fetches tools from ToolService.
    """

    def __init__(self, registry: ToolRegistry, tools_provider: "ToolService") -> None:
        self.registry = registry
        self.tools_provider = tools_provider

    def get_tools_for_agent(
        self,
        allow_list_id: str,
        *,
        local_mode: bool = False,
        exclude_embedding_tools: bool = False,
        include_deferred_tools: bool = True,
        log_tool_annotations: bool = True,
    ) -> List[StructuredTool]:
        """
        Resolve an allow-list to tool names, then fetch tools from ToolService.

        When include_deferred_tools is False (Phase 3), tools with defer_loading=True
        are excluded from the list so they are not sent in the initial payload; they
        remain discoverable via get_discovery_tools_for_agent when using search flow.

        When log_tool_annotations is True (Phase 4), each tool is wrapped so that
        on invoke, annotations are logged before delegating to the inner tool.

        Returns:
            List of StructuredTool instances. Missing tools are omitted (ToolService logs them).
        """
        try:
            names = self.registry.resolve_allow_list(
                allow_list_id,
                local_mode=local_mode,
                exclude_embedding_tools=exclude_embedding_tools,
            )
        except RegistryError as e:
            logger.error(
                "ToolResolver: failed to resolve allow_list_id=%s: %s",
                allow_list_id,
                e,
            )
            raise
        if not include_deferred_tools:
            names = [
                n
                for n in names
                if not (
                    (meta := self.registry.get_metadata(n))
                    and getattr(meta, "defer_loading", False)
                )
            ]
        if not names:
            logger.warning(
                "ToolResolver: allow_list_id={} resolved to empty tool list",
                allow_list_id,
            )
        tools = self.tools_provider.get_tools(
            names,
            exclude_embedding_tools=exclude_embedding_tools,
        )
        if not tools:
            logger.warning(
                "ToolResolver: allow_list_id={} returned no tools (resolved_count={}); agent may have no tools",
                allow_list_id,
                len(names),
            )
        if log_tool_annotations:
            tools = [wrap_tool_for_annotation_logging(t, self.registry) for t in tools]
        logger.info(
            "ToolResolver: allow_list_id={} local_mode={} resolved_count={} tools_returned={}",
            allow_list_id,
            local_mode,
            len(names),
            len(tools),
        )
        return tools

    def get_discovery_tools_for_agent(
        self,
        allow_list_id: str,
        *,
        local_mode: bool = False,
        exclude_embedding_tools: bool = False,
        log_tool_annotations: bool = True,
    ) -> List[StructuredTool]:
        """
        Return the three Phase 3 discovery meta-tools (search_tools, describe_tool, execute_tool)
        scoped to the given allow-list. Use when use_tool_search_flow=True instead of
        passing the full tool list. When log_tool_annotations=True (Phase 4), execute_tool
        logs annotations before invoking the underlying tool.

        Returns:
            List of three StructuredTool instances: search_tools, describe_tool, execute_tool.
        """
        from app.modules.intelligence.tools.registry.discovery_tools import (
            get_discovery_tools,
        )

        search_tool, describe_tool, execute_tool = get_discovery_tools(
            self,
            allow_list_id,
            local_mode=local_mode,
            exclude_embedding_tools=exclude_embedding_tools,
            log_tool_annotations=log_tool_annotations,
        )
        return [search_tool, describe_tool, execute_tool]

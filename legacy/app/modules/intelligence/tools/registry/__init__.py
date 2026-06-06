"""Tool registry: single source of truth for tool metadata and agent–tool binding."""

from app.modules.intelligence.tools.registry.annotation_logging import (
    get_annotations_for_logging,
    wrap_tool_for_annotation_logging,
)
from app.modules.intelligence.tools.registry.discovery_tools import get_discovery_tools
from app.modules.intelligence.tools.registry.exceptions import RegistryError
from app.modules.intelligence.tools.registry.population import (
    build_registry_from_tool_service,
)
from app.modules.intelligence.tools.registry.registry import ToolRegistry
from app.modules.intelligence.tools.registry.resolver import ToolResolver
from app.modules.intelligence.tools.registry.schema import (
    AllowListDefinition,
    ToolCategory,
    ToolMetadata,
    ToolTier,
)

__all__ = [
    "AllowListDefinition",
    "RegistryError",
    "ToolCategory",
    "ToolMetadata",
    "ToolRegistry",
    "ToolResolver",
    "ToolTier",
    "build_registry_from_tool_service",
    "get_annotations_for_logging",
    "get_discovery_tools",
    "wrap_tool_for_annotation_logging",
]

"""In-memory tool registry: metadata store and resolution (allow-lists, categories)."""

from typing import Dict, List, Optional

from app.modules.intelligence.tools.registry.exceptions import RegistryError
from app.modules.intelligence.tools.registry.schema import (
    AllowListDefinition,
    ToolCategory,
    ToolMetadata,
    ToolTier,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolRegistry:
    """
    Single source of truth for tool metadata and agent–tool binding.
    Resolves allow-lists and categories to primary tool names (no aliases).
    """

    def __init__(self) -> None:
        self._metadata_by_name: Dict[str, ToolMetadata] = {}
        self._allow_lists: Dict[str, AllowListDefinition] = {}

    def register(self, metadata: ToolMetadata) -> None:
        """Register one tool metadata. Primary name must be unique."""
        if metadata.name in self._metadata_by_name:
            logger.debug(
                "Overwriting existing tool metadata for name=%s", metadata.name
            )
        self._metadata_by_name[metadata.name] = metadata
        # Also index by id if different from name
        if metadata.id != metadata.name:
            # We key by name for lookup; id is for stability in definitions
            pass  # We only look up by name in get_metadata

    def register_allow_list(self, name: str, definition: AllowListDefinition) -> None:
        """Register an allow-list by id."""
        if definition.name != name:
            definition = definition.model_copy(update={"name": name})
        self._allow_lists[name] = definition

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Return metadata for a primary tool name, or None."""
        return self._metadata_by_name.get(name)

    def resolve_allow_list(
        self,
        allow_list_id: str,
        *,
        local_mode: bool = False,
        exclude_embedding_tools: bool = False,
    ) -> List[str]:
        """
        Resolve an allow-list to a deduplicated list of primary tool names.
        Applies allow-list filters: add_when_non_local when not local_mode, exclude_in_local when local_mode.
        Applies per-tool filters: local_mode_only tools are excluded when local_mode=False;
        non_local_only tools are excluded when local_mode=True (so e.g. terminal tools are only sent in local/VSCode mode).
        """
        if allow_list_id not in self._allow_lists:
            raise RegistryError(f"Unknown allow_list_id: {allow_list_id!r}")
        definition = self._allow_lists[allow_list_id]

        # Start from explicit tool names
        names: List[str] = list(definition.tool_names)

        # Expand categories to primary names
        if definition.categories:
            by_cat = self.resolve_categories(
                list(definition.categories), tier=definition.tier_filter
            )
            for n in by_cat:
                if n not in names:
                    names.append(n)

        # Add tools when not local_mode
        if not local_mode and definition.add_when_non_local:
            for n in definition.add_when_non_local:
                if n not in names:
                    names.append(n)

        # Add tools when embedding is ok (exclude_embedding_tools=False)
        if not exclude_embedding_tools and definition.add_when_embedding_ok:
            for n in definition.add_when_embedding_ok:
                if n not in names:
                    names.append(n)

        # Exclude tools in local mode
        if local_mode and definition.exclude_in_local:
            names = [n for n in names if n not in definition.exclude_in_local]

        # Deduplicate: resolve names to primary (support alias in allow-list), skip if unknown
        seen: set = set()
        result: List[str] = []
        for n in names:
            primary = self._resolve_to_primary_name(n)
            if primary is None:
                logger.warning(
                    "Allow-list %s references tool name %s not in registry; skipping",
                    allow_list_id,
                    n,
                )
                continue
            if primary in seen:
                continue
            seen.add(primary)
            result.append(primary)

        # Per-tool local_mode filtering: local_mode_only tools only when local_mode=True;
        # non_local_only tools only when local_mode=False (e.g. VS Code extension gets terminal tools only in local_mode).
        filtered: List[str] = []
        for name in result:
            meta = self.get_metadata(name)
            if meta is None:
                filtered.append(name)
                continue
            if local_mode and meta.non_local_only:
                continue
            if not local_mode and meta.local_mode_only:
                continue
            filtered.append(name)
        result = filtered

        logger.debug(
            "resolve_allow_list allow_list_id=%s local_mode=%s exclude_embedding=%s result_count=%s",
            allow_list_id,
            local_mode,
            exclude_embedding_tools,
            len(result),
        )
        return result

    def resolve_categories(
        self,
        categories: List[str],
        tier: Optional[ToolTier] = None,
    ) -> List[str]:
        """Resolve categories to a deduplicated list of primary tool names. Optionally filter by tier."""
        result: List[str] = []
        seen: set = set()
        for meta in self._metadata_by_name.values():
            if meta.category not in categories:
                continue
            if tier is not None and meta.tier != tier:
                continue
            if meta.name in seen:
                continue
            seen.add(meta.name)
            result.append(meta.name)
        return result

    def _resolve_to_primary_name(self, name: str) -> Optional[str]:
        """Return primary name for a given name or alias; None if not in registry."""
        if name in self._metadata_by_name:
            return name
        for meta in self._metadata_by_name.values():
            if name in meta.aliases:
                return meta.name
        return None

    def all_primary_names(self) -> List[str]:
        """Return all registered primary tool names (for validation / ToolService alignment)."""
        return list(self._metadata_by_name.keys())

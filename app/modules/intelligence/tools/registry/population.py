"""Populate ToolRegistry from definitions and ToolService (descriptions, validation)."""

from typing import TYPE_CHECKING, Dict, Optional

from app.modules.intelligence.tools.registry.definitions import (
    ALLOW_LIST_DEFINITIONS,
    OPTIONAL_TOOL_NAMES,
    TOOL_DEFINITIONS,
)
from app.modules.intelligence.tools.registry.registry import ToolRegistry
from app.modules.intelligence.tools.registry.schema import (
    ToolCategory,
    ToolMetadata,
    ToolTier,
)
from app.modules.utils.logger import setup_logger

if TYPE_CHECKING:
    from app.modules.intelligence.tools.tool_service import ToolService

logger = setup_logger(__name__)

# Phase 4: derivation rules for behavioral annotations when not set in definition.
# Explicit definition always wins. Names are matched by prefix (e.g. delete_*).
_READ_ONLY_CATEGORIES: tuple[str, ...] = ("search", "file_fetch")
_READ_ONLY_NAME_PREFIXES: tuple[str, ...] = ("get_", "fetch_", "list_", "search_")
_CODE_CHANGES_DESTRUCTIVE_PREFIXES: tuple[str, ...] = (
    "delete_",
    "clear_",
    "replace_",
    "update_file_",
    "insert_",
    "add_file_",
)


def _derive_phase4_annotations(name: str, category: ToolCategory) -> Dict[str, bool]:
    """
    Derive optional annotation defaults from category and name.
    Used only when definition does not set the field. Overridable by explicit definition.
    """
    out: Dict[str, bool] = {}
    if category in _READ_ONLY_CATEGORIES or any(
        name.startswith(p) for p in _READ_ONLY_NAME_PREFIXES
    ):
        out["read_only"] = True
        out["idempotent"] = True
    if category == "code_changes" and any(
        name.startswith(p) for p in _CODE_CHANGES_DESTRUCTIVE_PREFIXES
    ):
        out["destructive"] = True
    if category == "terminal":
        out["destructive"] = True
        out["requires_confirmation"] = True
    return out


def build_registry_from_tool_service(
    tool_service: "ToolService",
    *,
    strict: bool = False,
) -> ToolRegistry:
    """
    Build a fully populated ToolRegistry from static definitions and ToolService.

    - Descriptions are taken from ToolService.list_tools() when available.
    - Validates: registry primary names (+ aliases) vs ToolService keys; logs discrepancies.
    - If strict=True, raises on registry names not in ToolService or ToolService keys not in registry.

    Returns:
        ToolRegistry with all metadata and allow-lists registered.
    """
    registry = ToolRegistry()
    tool_info_by_id = {t.id: t for t in tool_service.list_tools()}

    # Register tool metadata from definitions (include aliases in lookup for description)
    for name, defn in TOOL_DEFINITIONS.items():
        desc = ""
        for key in [name] + defn.get("aliases", []):
            if key in tool_info_by_id:
                raw = tool_info_by_id[key].description
                desc = raw if isinstance(raw, str) else str(raw)
                break
        # Phase 3: short_description from definition or derive from full description
        short_desc: Optional[str] = defn.get("short_description")
        if not short_desc and desc:
            # First sentence or first 100 chars, trimmed
            first_sentence = (desc.split(".")[0] + ".").strip()
            short_desc = first_sentence[:100].rstrip() or desc[:100].strip() or None
        if short_desc and len(short_desc) > 200:
            short_desc = short_desc[:200].rstrip()
        defer_loading = defn.get("defer_loading", False)
        # Phase 4: behavioral annotations from definition; derive if not set
        derived = _derive_phase4_annotations(name, defn["category"])
        read_only = defn.get("read_only")
        if read_only is None and "read_only" in derived:
            read_only = derived["read_only"]
        destructive = defn.get("destructive")
        if destructive is None and "destructive" in derived:
            destructive = derived["destructive"]
        idempotent = defn.get("idempotent")
        if idempotent is None and "idempotent" in derived:
            idempotent = derived["idempotent"]
        requires_confirmation = defn.get("requires_confirmation")
        if requires_confirmation is None and "requires_confirmation" in derived:
            requires_confirmation = derived["requires_confirmation"]
        meta = ToolMetadata(
            id=name,
            name=name,
            description=desc or name,
            short_description=short_desc,
            tier=defn["tier"],
            category=defn["category"],
            defer_loading=defer_loading,
            aliases=defn.get("aliases", []),
            local_mode_only=defn.get("local_mode_only", False),
            non_local_only=defn.get("non_local_only", False),
            embedding_ok_only=defn.get("embedding_ok_only", False),
            read_only=read_only,
            destructive=destructive,
            idempotent=idempotent,
            requires_confirmation=requires_confirmation,
        )
        registry.register(meta)

    # Register allow-lists
    for allow_list in ALLOW_LIST_DEFINITIONS:
        registry.register_allow_list(allow_list.name, allow_list)

    # Align with ToolService: compare keys
    service_keys = set(tool_service.tools.keys())
    registry_all_names = set(registry.all_primary_names())
    for n in registry.all_primary_names():
        meta = registry.get_metadata(n)
        if meta:
            registry_all_names.update(meta.aliases)

    in_registry_not_service = registry_all_names - service_keys - OPTIONAL_TOOL_NAMES
    in_service_not_registry = service_keys - registry_all_names

    if in_registry_not_service:
        logger.warning(
            "Tool registry has names not in ToolService: {}",
            sorted(in_registry_not_service),
        )
        if strict:
            raise ValueError(
                f"Registry names not in ToolService: {sorted(in_registry_not_service)}"
            )
    if in_service_not_registry:
        logger.warning(
            "ToolService has keys not in registry (consider adding to definitions): {}",
            sorted(in_service_not_registry),
        )
        if strict:
            raise ValueError(
                f"ToolService keys not in registry: {sorted(in_service_not_registry)}"
            )

    logger.info(
        "Tool registry loaded with {} tools, {} allow-lists",
        len(registry.all_primary_names()),
        len(ALLOW_LIST_DEFINITIONS),
    )
    return registry

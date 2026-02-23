"""Tool registry schema: metadata, enums, and allow-list definitions."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# --- Enums / literals ---

ToolTier = Literal["low", "medium", "high"]

ToolCategory = Literal[
    "search",
    "code_changes",
    "terminal",
    "todo",
    "requirement",
    "integration_jira",
    "integration_github",
    "integration_confluence",
    "integration_linear",
    "delegation",
    "web",
    "knowledge_graph",
    "file_fetch",
    "analysis",
    "other",
]


# --- Metadata ---


class ToolMetadata(BaseModel):
    """Per-tool metadata: single source of truth for name, tier, category, aliases."""

    id: str = Field(..., description="Stable unique id (e.g. primary name or slug)")
    name: str = Field(
        ...,
        description="Primary name used in ToolService (must match a key in ToolService)",
    )
    short_description: Optional[str] = Field(
        None,
        max_length=200,
        description="Short description for discovery (Phase 3); < ~100 chars recommended",
    )
    description: str = Field(..., description="Full description (from tool)")
    tier: ToolTier = Field(
        ...,
        description="Abstraction level: low (bash/read/write), medium (edit/grep), high (task/web)",
    )
    category: ToolCategory = Field(
        ...,
        description="E.g. search, code_changes, terminal, integration_jira, todo",
    )
    defer_loading: bool = Field(
        False,
        description="For Phase 3; mark true for rarely used tools",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Other keys in ToolService for the same tool (e.g. github_tool for code_provider_tool)",
    )
    # Optional filters for allow-list resolution
    local_mode_only: bool = Field(
        False,
        description="If True, include only when local_mode=True",
    )
    non_local_only: bool = Field(
        False,
        description="If True, include only when local_mode=False",
    )
    embedding_ok_only: bool = Field(
        False,
        description="If True, exclude when exclude_embedding_tools=True",
    )
    # Phase 4: behavioral annotations (optional; for logging and future safety/confirmation)
    read_only: Optional[bool] = Field(
        None,
        description="True if tool does not modify state (e.g. search, get, fetch)",
    )
    destructive: Optional[bool] = Field(
        None,
        description="True if tool can delete or overwrite data",
    )
    idempotent: Optional[bool] = Field(
        None,
        description="True if safe to call multiple times with same effect",
    )
    requires_confirmation: Optional[bool] = Field(
        None,
        description="True if user confirmation recommended before use",
    )


# --- Allow-list definition ---


class AllowListDefinition(BaseModel):
    """Named set of tool names and/or category references with optional tier filter."""

    name: str = Field(..., description="Allow-list id (e.g. code_gen, general_purpose)")
    tool_names: List[str] = Field(
        default_factory=list,
        description="Explicit primary tool names",
    )
    categories: List[ToolCategory] = Field(
        default_factory=list,
        description="Categories to expand to primary names via registry",
    )
    tier_filter: Optional[ToolTier] = Field(
        None,
        description="If set, only include tools of this tier when expanding categories",
    )
    # Optional additive/s subtractive lists keyed by context (for code_gen: non_local, embedding_ok)
    add_when_non_local: List[str] = Field(
        default_factory=list,
        description="Primary names to add when local_mode=False",
    )
    add_when_embedding_ok: List[str] = Field(
        default_factory=list,
        description="Primary names to add when exclude_embedding_tools=False (not used to subtract)",
    )
    exclude_in_local: List[str] = Field(
        default_factory=list,
        description="Primary names to exclude when local_mode=True (e.g. show_diff)",
    )

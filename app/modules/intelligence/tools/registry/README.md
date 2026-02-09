# Tool Registry (Phase 1 + Phase 2)

Single source of truth for tool metadata and agent–tool binding. Agents resolve tool names via **allow-lists** or **categories** instead of hardcoded lists.

## Schema

- **ToolMetadata**: `id`, `name`, `description`, `tier`, `category`, optional `short_description`, `defer_loading`, `aliases`, and (Phase 4) optional `read_only`, `destructive`, `idempotent`, `requires_confirmation`.
- **ToolTier**: `low` | `medium` | `high`.
- **ToolCategory**: e.g. `search`, `code_changes`, `terminal`, `integration_jira`, `todo`, `requirement`, etc.
- **AllowListDefinition**: named set of `tool_names` and/or `categories`, with optional `add_when_non_local`, `exclude_in_local`, `add_when_embedding_ok`.

## Usage

- **Population**: `build_registry_from_tool_service(tool_service, strict=False)` builds a `ToolRegistry` from `definitions.py` and ToolService (descriptions, validation).
- **Resolver**: `ToolResolver(registry, tools_provider).get_tools_for_agent(allow_list_id, local_mode=..., exclude_embedding_tools=...)` returns `List[StructuredTool]`.
- **Agents**: CodeGenAgent and GeneralPurposeAgent accept optional `tool_resolver`; when set they use `get_tools_for_agent("code_gen")` / `get_tools_for_agent("general_purpose")` instead of hardcoded lists.

## Adding tools / allow-lists

1. **New tool**: Add an entry to `TOOL_DEFINITIONS` in `definitions.py` with `tier` and `category`. Ensure the key matches the ToolService key (primary name).
2. **New allow-list**: Append an `AllowListDefinition` to `ALLOW_LIST_DEFINITIONS` with `name`, `tool_names`, and optionally `add_when_non_local`, `categories`, etc.
3. **New agent using registry**: Inject `ToolResolver` in the agent constructor and call `tool_resolver.get_tools_for_agent(allow_list_id, ...)` in the agent build path.

## Files

- `schema.py`: ToolMetadata, ToolTier, ToolCategory, AllowListDefinition.
- `registry.py`: ToolRegistry (register, resolve_allow_list, resolve_categories, get_metadata, all_primary_names).
- `definitions.py`: TOOL_DEFINITIONS, ALLOW_LIST_DEFINITIONS (code_gen, general_purpose).
- `population.py`: build_registry_from_tool_service.
- `resolver.py`: ToolResolver.get_tools_for_agent.
- `exceptions.py`: RegistryError.

## Phase 2: Scoped tool sets (implemented)

When `PydanticMultiAgent` is given a `tool_resolver` (CodeGenAgent and GeneralPurposeAgent pass it when available), it forwards it to `AgentFactory`. The factory then builds supervisor, delegate, and integration tool lists from registry allow-lists:

- **supervisor**: Curated set for coordination and light discovery (search, fetch, web). Supervisor delegates execution to THINK_EXECUTE; no terminal or full execution suite.
- **execute**: Full execution set for THINK_EXECUTE delegate (code_gen minus todo/requirement); same capability as today, source is registry.
- **explore**: Minimal read-only set for future “explore” subagent; defined in registry, not yet used.
- **integration_jira**, **integration_github**, **integration_confluence**, **integration_linear**: Tool names for each integration agent, replacing the hardcoded map when `tool_resolver` is set.

If `tool_resolver` is `None`, AgentFactory keeps backward-compatible behavior (uses `self.tools` and hardcoded integration map).

See `docs/phase2_scoped_tool_sets_implementation_plan.md` for the full implementation plan.

## Phase 3: Discovery metadata and optional deferred loading (implemented)

Phase 3 reduces tool-related token usage by:

1. **Short description (discovery):** Every tool has `short_description` in the registry (explicit in definitions or derived from full description). Used in discovery responses (e.g. `search_tools` return value).

2. **Optional search → describe → execute flow:** When `use_tool_search_flow=True` (e.g. on `ChatContext`), the agent receives three meta-tools instead of the full list:
   - **search_tools**: List available tools with short descriptions (optional `query` reserved for future filtering).
   - **describe_tool(name)**: Return full description and JSON argument schema for a tool. Only for tools in the agent’s allowed set.
   - **execute_tool(name, tool_args)**: Run a tool by name with the given arguments. Only for tools in the allowed set.

   The allowed set is the same as for “all upfront” mode (resolve_allow_list for the agent’s allow-list). Instructions are appended so the model knows to use search_tools → describe_tool → execute_tool.

3. **Optional defer_loading:** In `TOOL_DEFINITIONS`, set `defer_loading: True` for rarely used tools. Population sets `ToolMetadata.defer_loading`. In all-upfront mode, `get_tools_for_agent(..., include_deferred_tools=False)` excludes these tools from the initial payload; in search flow all tools remain discoverable via `search_tools`.

**Config and wiring:**

- **ChatContext**: `use_tool_search_flow: bool = False`. Set per request for A/B testing.
- **Resolver**: `get_tools_for_agent(..., include_deferred_tools=True)`; `get_discovery_tools_for_agent(allow_list_id, ...)` returns the three discovery tools.
- **AgentFactory**: `build_supervisor_agent_tools` and `build_delegate_agent_tools` accept `use_tool_search_flow`; when True and `tool_resolver` is set, they use `get_discovery_tools_for_agent` instead of `get_tools_for_agent`. Cache keys include `use_tool_search_flow`.

**Files:**

- `definitions.py`: Optional `short_description` and `defer_loading` per tool.
- `population.py`: Sets `short_description` (from defn or derived) and `defer_loading`.
- `discovery_tools.py`: Builds search_tools, describe_tool, execute_tool (scoped to allow-list).
- `resolver.py`: `include_deferred_tools` on `get_tools_for_agent`; `get_discovery_tools_for_agent`.

See `docs/phase3_discovery_metadata_deferred_loading_implementation_plan.md` for the full plan.

## Phase 4: Behavioral annotations and safety (implemented)

Phase 4 adds **behavioral annotations** to the registry and uses them for **logging/observability** and future safety features.

1. **Annotations in registry:** Optional per-tool fields on `ToolMetadata`: `read_only`, `destructive`, `idempotent`, `requires_confirmation`. Populated from definitions and derivation rules (e.g. search → read_only; code_changes write/delete → destructive; terminal → destructive, requires_confirmation).

2. **Logging on tool use:** When `log_tool_annotations=True` (default):
   - **Direct path:** Each tool returned by `get_tools_for_agent` is wrapped so that on invoke, annotations are logged then the inner tool runs.
   - **Discovery path:** In `execute_tool`, annotations are logged before invoking the underlying tool.
     Both paths use `get_annotations_for_logging(metadata)` from `annotation_logging.py`; logging is best-effort (missing registry/metadata never breaks execution).

3. **Config:** `ChatContext.log_tool_annotations: bool = True` so callers can disable annotation logging if needed.

4. **Future extensions (out of scope for Phase 4):**
   - **Confirmation policy:** A component can inspect annotations (e.g. `requires_confirmation` or `destructive`) before invoke and require user approval in sensitive modes. Integration point: same pre-invoke hook where we log.
   - **SecurityAnalyzer:** A component could compute a risk level (LOW/MEDIUM/HIGH) from annotations and attach to tool metadata or run context for UI/audit.

**Files:**

- `schema.py`: ToolMetadata has optional `read_only`, `destructive`, `idempotent`, `requires_confirmation`.
- `definitions.py`: Optional annotation keys on TOOL_DEFINITIONS for representative tools.
- `population.py`: Sets annotations from definition; derivation rules when not set.
- `annotation_logging.py`: `get_annotations_for_logging(metadata)`, `wrap_tool_for_annotation_logging(tool, registry)`.
- `discovery_tools.py`: In `_execute_tool`, log annotations before invoke when `log_tool_annotations=True`.
- `resolver.py`: `log_tool_annotations` on `get_tools_for_agent` and `get_discovery_tools_for_agent`.
- `chat_agent.py`: ChatContext.log_tool_annotations.

See `docs/phase4_behavioral_annotations_safety_implementation_plan.md` for the full plan.

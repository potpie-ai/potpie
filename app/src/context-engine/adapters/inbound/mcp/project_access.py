"""Restrict MCP tool project_id to an explicit allowlist (multi-tenant boundary)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _McpAccessLogState:
    allowlist_logged: bool = False
    trust_all_logged: bool = False


_log_state = _McpAccessLogState()


def _parse_allowlist() -> frozenset[str] | None:
    raw = (os.getenv("CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(
            "CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS must be valid JSON array of project IDs"
        )
        return frozenset()
    if not isinstance(data, list):
        logger.error("CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS must be a JSON array")
        return frozenset()
    return frozenset(str(x) for x in data)


def _trust_all_projects() -> bool:
    return (os.getenv("CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def assert_mcp_project_allowed(project_id: str) -> None:
    """
    Enforce MCP project scope.

    - If CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS is set to a JSON array, project_id must be listed.
    - If unset/empty and CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS is not true, all access is denied.
    - If CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS=true, any project_id is allowed (dev only).
    """
    allow = _parse_allowlist()
    trust_all = _trust_all_projects()

    if allow is not None:
        if not _log_state.allowlist_logged:
            logger.info(
                "MCP project allowlist active (%d project id(s))",
                len(allow),
            )
            _log_state.allowlist_logged = True
        if project_id not in allow:
            raise ValueError(
                f"project_id not permitted for MCP (not in CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS): {project_id!r}"
            )
        return

    if trust_all:
        if not _log_state.trust_all_logged:
            logger.warning(
                "MCP CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS is enabled; any project_id is accepted"
            )
            _log_state.trust_all_logged = True
        return

    raise ValueError(
        "MCP access denied: set CONTEXT_ENGINE_MCP_ALLOWED_PROJECTS to a JSON array of "
        "allowed project IDs, or set CONTEXT_ENGINE_MCP_TRUST_ALL_PROJECTS=true for "
        "development only"
    )

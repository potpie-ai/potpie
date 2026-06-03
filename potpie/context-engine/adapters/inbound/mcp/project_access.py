"""Restrict MCP tool pot_id to an explicit allowlist (multi-tenant boundary)."""

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
    raw = os.getenv("CONTEXT_ENGINE_MCP_ALLOWED_POTS", "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(
            "CONTEXT_ENGINE_MCP_ALLOWED_POTS must be valid JSON array of pot IDs"
        )
        return frozenset()
    if not isinstance(data, list):
        logger.error("CONTEXT_ENGINE_MCP_ALLOWED_POTS must be a JSON array")
        return frozenset()
    return frozenset(str(x) for x in data)


def _trust_all_pots() -> bool:
    return os.getenv("CONTEXT_ENGINE_MCP_TRUST_ALL_POTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def assert_mcp_pot_allowed(pot_id: str) -> None:
    """
    Enforce MCP pot scope.

    - If CONTEXT_ENGINE_MCP_ALLOWED_POTS is set to a JSON array, pot_id must be listed.
    - If unset/empty and CONTEXT_ENGINE_MCP_TRUST_ALL_POTS is not true, all access is denied.
    - If CONTEXT_ENGINE_MCP_TRUST_ALL_POTS=true, any pot_id is allowed (dev only).
    """
    allow = _parse_allowlist()
    trust_all = _trust_all_pots()

    if allow is not None:
        if not _log_state.allowlist_logged:
            logger.info(
                "MCP pot allowlist active (%d pot id(s))",
                len(allow),
            )
            _log_state.allowlist_logged = True
        if pot_id not in allow:
            raise ValueError(
                f"pot_id not permitted for MCP (not in CONTEXT_ENGINE_MCP_ALLOWED_POTS): {pot_id!r}"
            )
        return

    if trust_all:
        if not _log_state.trust_all_logged:
            logger.warning(
                "MCP CONTEXT_ENGINE_MCP_TRUST_ALL_POTS is enabled; any pot_id is accepted"
            )
            _log_state.trust_all_logged = True
        return

    raise ValueError(
        "MCP access denied: set CONTEXT_ENGINE_MCP_ALLOWED_POTS to a JSON array of "
        "allowed pot IDs, or set CONTEXT_ENGINE_MCP_TRUST_ALL_POTS=true for "
        "development only"
    )

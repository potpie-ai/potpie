"""Phase 7: operator surface hardening.

Covers:

- ``_audit_operator_action`` emits a structured log record with the action,
  pot_id, actor identity, dry-run flag, and extra fields.
- ``_actor_identity`` extracts a sensible label from common actor shapes.
- Operator-tagged routes (``/reset``, ``/conflicts/*``, ``/maintenance/*``) carry
  the ``context:operator`` OpenAPI tag so docs and SDKs can group them away
  from the everyday agent surface.
"""

from __future__ import annotations

import logging

import pytest

from adapters.inbound.http.api.v1.context.router import (
    OPERATOR_TAG,
    _AUDIT_LOGGER_NAME,
    _actor_identity,
    _audit_operator_action,
    create_context_router,
)

pytestmark = pytest.mark.unit


def test_actor_identity_anonymous_when_none():
    assert _actor_identity(None) == "anonymous"


def test_actor_identity_reads_common_attrs():
    class A:
        email = "ops@example.com"

    assert _actor_identity(A()) == "ops@example.com"


def test_actor_identity_reads_dict_keys():
    assert _actor_identity({"id": "user-42"}) == "user-42"


def test_audit_operator_action_emits_structured_log(caplog):
    caplog.set_level(logging.WARNING, logger=_AUDIT_LOGGER_NAME)

    _audit_operator_action(
        action="hard_reset_pot",
        pot_id="pot-abc",
        actor={"email": "ops@example.com"},
        dry_run=False,
        extra={"skip_ledger": True, "outcome": "ok"},
    )

    records = [r for r in caplog.records if r.name == _AUDIT_LOGGER_NAME]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.WARNING
    assert "hard_reset_pot" in rec.getMessage()
    audit = getattr(rec, "audit", None)
    assert isinstance(audit, dict)
    assert audit["action"] == "hard_reset_pot"
    assert audit["pot_id"] == "pot-abc"
    assert audit["actor"] == "ops@example.com"
    assert audit["dry_run"] is False
    assert audit["skip_ledger"] is True
    assert audit["outcome"] == "ok"


def test_audit_extra_cannot_override_core_fields(caplog):
    caplog.set_level(logging.WARNING, logger=_AUDIT_LOGGER_NAME)

    _audit_operator_action(
        action="resolve_conflict",
        pot_id="pot-xyz",
        actor=None,
        extra={"pot_id": "attacker-wins", "action": "nope"},
    )

    rec = [r for r in caplog.records if r.name == _AUDIT_LOGGER_NAME][-1]
    audit = rec.audit
    assert audit["action"] == "resolve_conflict"
    assert audit["pot_id"] == "pot-xyz"
    assert audit["actor"] == "anonymous"


def _build_router_stub():
    def _require_auth():
        return None

    def _get_container():
        raise AssertionError("container should not be requested during route introspection")

    def _get_db():
        raise AssertionError("db should not be requested during route introspection")

    return create_context_router(
        require_auth=_require_auth,
        get_container=_get_container,
        get_db=_get_db,
        get_db_optional=_get_db,
    )


def test_operator_routes_carry_operator_tag():
    router = _build_router_stub()
    operator_paths = {
        "/reset",
        "/conflicts/list",
        "/conflicts/resolve",
        "/maintenance/classify-modified-edges",
    }
    seen: dict[str, list[str]] = {}
    for route in router.routes:
        path = getattr(route, "path", None)
        if path in operator_paths:
            seen[path] = list(getattr(route, "tags", []) or [])
    assert set(seen) == operator_paths, f"missing operator routes: {operator_paths - set(seen)}"
    for path, tags in seen.items():
        assert OPERATOR_TAG in tags, f"{path} missing {OPERATOR_TAG} (got {tags})"


def test_non_operator_routes_do_not_carry_operator_tag():
    router = _build_router_stub()
    for route in router.routes:
        path = getattr(route, "path", "")
        tags = list(getattr(route, "tags", []) or [])
        if path in {"/query/context-graph", "/status", "/record", "/events/reconcile"}:
            assert OPERATOR_TAG not in tags, f"{path} should not carry {OPERATOR_TAG}"

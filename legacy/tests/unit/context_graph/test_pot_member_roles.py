"""Role normalization and assignable-role parsing."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.modules.context_graph.pot_access import parse_assignable_role, parse_role
from app.modules.context_graph.pot_member_roles import (
    ALL_POT_ROLES,
    ASSIGNABLE_POT_ROLES,
    POT_ROLE_OWNER,
    POT_ROLE_USER,
    can_ingest_raw,
    can_manage_members,
    can_manage_repos_and_integrations,
    can_query_context,
    normalize_role,
)

pytestmark = pytest.mark.unit


class TestRoleNormalization:
    def test_owner_stays_owner(self) -> None:
        assert normalize_role("owner") == POT_ROLE_OWNER

    def test_user_stays_user(self) -> None:
        assert normalize_role("user") == POT_ROLE_USER

    @pytest.mark.parametrize("legacy", ["admin", "read_only", "viewer", "", None])
    def test_legacy_or_unknown_collapses_to_user(self, legacy) -> None:
        assert normalize_role(legacy) == POT_ROLE_USER

    def test_strict_active_role_set(self) -> None:
        assert set(ALL_POT_ROLES) == {"owner", "user"}
        assert ASSIGNABLE_POT_ROLES == ("user",)


class TestPermissionMatrix:
    def test_owner_can_manage(self) -> None:
        assert can_manage_members(POT_ROLE_OWNER)
        assert can_manage_repos_and_integrations(POT_ROLE_OWNER)

    def test_user_cannot_manage(self) -> None:
        assert not can_manage_members(POT_ROLE_USER)
        assert not can_manage_repos_and_integrations(POT_ROLE_USER)

    def test_both_roles_can_query_and_raw_ingest(self) -> None:
        for role in (POT_ROLE_OWNER, POT_ROLE_USER):
            assert can_query_context(role)
            assert can_ingest_raw(role)

    def test_legacy_role_rows_treat_as_user(self) -> None:
        """Legacy ``admin`` must not grant management rights after migration."""
        assert not can_manage_members("admin")
        assert not can_manage_repos_and_integrations("read_only")


class TestParseAssignableRole:
    def test_accepts_user(self) -> None:
        assert parse_assignable_role("user") == POT_ROLE_USER

    def test_accepts_user_case_insensitive(self) -> None:
        assert parse_assignable_role(" User ") == POT_ROLE_USER

    def test_rejects_owner(self) -> None:
        with pytest.raises(HTTPException) as ei:
            parse_assignable_role("owner")
        assert ei.value.status_code == 400

    @pytest.mark.parametrize("bad", ["admin", "read_only", "", "random"])
    def test_rejects_legacy_or_unknown(self, bad: str) -> None:
        with pytest.raises(HTTPException) as ei:
            parse_assignable_role(bad)
        assert ei.value.status_code == 400


class TestParseRoleAlias:
    def test_parse_role_alias_matches_assignable(self) -> None:
        assert parse_role("user") == POT_ROLE_USER

    def test_parse_role_alias_rejects_owner(self) -> None:
        with pytest.raises(HTTPException):
            parse_role("owner")

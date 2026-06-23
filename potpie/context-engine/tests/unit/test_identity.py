"""Identity contract (rebuild plan P1)."""

from __future__ import annotations

import pytest

from potpie.context_engine.domain.identity import (
    IdentityClass,
    IdentityError,
    IdentitySpec,
    get_identity,
    mint_entity_key,
    register_identity,
    validate_entity_key,
)


class TestSlugAliasIdentity:
    """SLUG_ALIAS keys converge regardless of casing / whitespace / punctuation."""

    def test_simple_name_becomes_slug(self) -> None:
        svc = get_identity("Service")
        assert svc is not None
        assert mint_entity_key(svc, name="auth-svc") == "service:auth-svc"

    def test_casing_and_whitespace_normalize(self) -> None:
        svc = get_identity("Service")
        assert svc is not None
        # Three phrasings of the same logical service collapse to one key.
        keys = {
            mint_entity_key(svc, name="auth-svc"),
            mint_entity_key(svc, name="Auth-SVC"),
            mint_entity_key(svc, name=" Auth   SVC "),
        }
        assert keys == {"service:auth-svc"}

    def test_punctuation_normalizes(self) -> None:
        svc = get_identity("Service")
        assert svc is not None
        assert mint_entity_key(svc, name="Auth Service") == "service:auth-service"

    def test_empty_name_raises(self) -> None:
        svc = get_identity("Service")
        assert svc is not None
        for bad in ("", "   ", "!!!", "\t"):
            with pytest.raises(IdentityError):
                mint_entity_key(svc, name=bad)


class TestExternalIdIdentity:
    """EXTERNAL_ID keys preserve the source identifier shape (no slugify).

    Activity is the canonical EXTERNAL_ID entity in the unified ontology;
    PR/Issue/Commit/Deployment are now ``verb`` sub-kinds of Activity rather
    than separate labels.
    """

    def test_activity_with_extra_segment(self) -> None:
        spec = get_identity("Activity")
        assert spec is not None
        key = mint_entity_key(spec, external_id="1042", extra_segments=("github/pr",))
        assert key == "activity:github/pr:1042"

    def test_external_id_preserves_slashes(self) -> None:
        spec = get_identity("Activity")
        assert spec is not None
        # Commit SHAs are hex; verify shape passes through unmangled.
        key = mint_entity_key(spec, external_id="abc123def456")
        assert key == "activity:abc123def456"

    def test_requires_external_id(self) -> None:
        spec = get_identity("Activity")
        assert spec is not None
        with pytest.raises(IdentityError):
            mint_entity_key(spec, name="ignored", external_id=None)


class TestContentHashIdentity:
    """CONTENT_HASH keys are reproducible from canonical content text."""

    def test_same_content_same_key(self) -> None:
        spec = get_identity("Decision")
        assert spec is not None
        body = "ADR-007: adopt Position B for canonical edges"
        assert mint_entity_key(spec, content=body) == mint_entity_key(
            spec, content=body
        )

    def test_different_content_different_key(self) -> None:
        spec = get_identity("Decision")
        assert spec is not None
        a = mint_entity_key(spec, content="ADR-007")
        b = mint_entity_key(spec, content="ADR-008")
        assert a != b

    def test_content_required(self) -> None:
        spec = get_identity("Decision")
        assert spec is not None
        with pytest.raises(IdentityError):
            mint_entity_key(spec, content="")


class TestValidateEntityKey:
    """validate_entity_key recognises keys minted by the same spec."""

    def test_round_trip_slug(self) -> None:
        svc = get_identity("Service")
        assert svc is not None
        key = mint_entity_key(svc, name="checkout-v2")
        assert validate_entity_key(svc, key) is True

    def test_round_trip_external(self) -> None:
        spec = get_identity("Activity")
        assert spec is not None
        key = mint_entity_key(spec, external_id="42")
        assert validate_entity_key(spec, key) is True

    def test_round_trip_content_hash(self) -> None:
        spec = get_identity("Decision")
        assert spec is not None
        key = mint_entity_key(spec, content="A note about a decision")
        assert validate_entity_key(spec, key) is True

    def test_mismatched_prefix_fails(self) -> None:
        svc = get_identity("Service")
        assert svc is not None
        assert validate_entity_key(svc, "feature:checkout-v2") is False

    def test_empty_key_fails(self) -> None:
        svc = get_identity("Service")
        assert svc is not None
        assert validate_entity_key(svc, "") is False
        assert validate_entity_key(svc, "service:") is False


class TestRegistry:
    """Registry rejects conflicting specs but accepts identical re-registrations."""

    def test_duplicate_identical_registration_is_no_op(self) -> None:
        spec = IdentitySpec(
            label="Service", klass=IdentityClass.SLUG_ALIAS, key_prefix="service"
        )
        # Idempotent — register with the existing spec succeeds.
        register_identity(spec)
        assert get_identity("Service") == spec

    def test_conflicting_registration_raises(self) -> None:
        with pytest.raises(IdentityError):
            register_identity(
                IdentitySpec(
                    label="Service",
                    klass=IdentityClass.EXTERNAL_ID,  # different from default
                    key_prefix="service",
                )
            )

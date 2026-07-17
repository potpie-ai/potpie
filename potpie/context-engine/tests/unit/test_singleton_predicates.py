"""Singleton-predicate registry (rebuild plan P2 / F3)."""

from __future__ import annotations

import pytest

from potpie_context_engine.domain.singleton_predicates import (
    all_singletons,
    is_singleton_predicate,
    register_singleton,
    replace_singletons,
    unregister_singleton,
)


def test_defaults_include_OWNED_BY() -> None:
    assert is_singleton_predicate("OWNED_BY")


def test_non_singleton_returns_false() -> None:
    # DEPENDS_ON is multi-valued — a service can have many dependencies.
    assert is_singleton_predicate("DEPENDS_ON") is False


def test_empty_and_none_return_false() -> None:
    assert is_singleton_predicate("") is False
    assert is_singleton_predicate(None) is False


def test_register_then_check() -> None:
    register_singleton("TEST_PREDICATE_FOR_REGISTRATION")
    try:
        assert is_singleton_predicate("TEST_PREDICATE_FOR_REGISTRATION")
    finally:
        unregister_singleton("TEST_PREDICATE_FOR_REGISTRATION")


def test_register_idempotent() -> None:
    register_singleton("OWNED_BY")
    register_singleton("OWNED_BY")
    # Snapshot still contains exactly one entry for OWNED_BY.
    snapshot = all_singletons()
    assert "OWNED_BY" in snapshot


def test_register_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        register_singleton("")


def test_replace_singletons_swaps_full_set() -> None:
    saved = all_singletons()
    try:
        replace_singletons(["A", "B"])
        assert all_singletons() == frozenset({"A", "B"})
        assert is_singleton_predicate("OWNED_BY") is False
        assert is_singleton_predicate("A") is True
    finally:
        replace_singletons(saved)


def test_replace_singletons_filters_invalid_entries() -> None:
    saved = all_singletons()
    try:
        replace_singletons(["VALID", "", None])  # type: ignore[list-item]
        assert all_singletons() == frozenset({"VALID"})
    finally:
        replace_singletons(saved)

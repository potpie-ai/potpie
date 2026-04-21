"""Tests for domain/intelligence_policy.py — EvidencePlan construction."""

from __future__ import annotations

import pytest

from domain.intelligence_models import (
    ArtifactRef,
    CapabilitySet,
    ContextResolutionRequest,
    ContextScope,
)
from domain.intelligence_policy import EvidencePlan, build_evidence_plan
from domain.intelligence_signals import SignalSet

pytestmark = pytest.mark.unit


def _req(**kwargs) -> ContextResolutionRequest:  # type: ignore[no-untyped-def]
    return ContextResolutionRequest(pot_id="p1", query=kwargs.pop("query", "test"), **kwargs)


def _full_caps(**overrides: bool) -> CapabilitySet:
    """All capabilities on, with optional overrides."""
    base = dict(
        semantic_search=True,
        artifact_context=True,
        change_history=True,
        decision_context=True,
        discussion_context=True,
        ownership_context=True,
        project_map_context=True,
        debugging_memory_context=True,
        causal_chain_context=True,
    )
    base.update(overrides)
    return CapabilitySet(**base)


def _no_caps() -> CapabilitySet:
    return CapabilitySet(
        semantic_search=False,
        artifact_context=False,
        change_history=False,
        decision_context=False,
        discussion_context=False,
        ownership_context=False,
        project_map_context=False,
        debugging_memory_context=False,
        causal_chain_context=False,
    )


# ---------------------------------------------------------------------------
# Basic plan construction
# ---------------------------------------------------------------------------


def test_build_evidence_plan_returns_evidence_plan() -> None:
    plan = build_evidence_plan(_req())
    assert isinstance(plan, EvidencePlan)


def test_build_evidence_plan_no_caps_disables_all() -> None:
    plan = build_evidence_plan(_req(), caps=_no_caps())
    assert not plan.run_semantic_search
    assert not plan.run_artifact
    assert not plan.run_change_history
    assert not plan.run_decisions
    assert not plan.run_discussions
    assert not plan.run_ownership
    assert not plan.run_project_map
    assert not plan.run_debugging_memory
    assert not plan.run_causal_chain


def test_build_evidence_plan_semantic_search_requires_cap() -> None:
    plan = build_evidence_plan(_req(), caps=_full_caps(semantic_search=False))
    assert not plan.run_semantic_search

    plan2 = build_evidence_plan(_req(), caps=_full_caps(semantic_search=True))
    assert plan2.run_semantic_search


def test_build_evidence_plan_semantic_search_excluded_by_exclude() -> None:
    req = _req(exclude=["semantic_search"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert not plan.run_semantic_search


# ---------------------------------------------------------------------------
# Artifact and PR handling
# ---------------------------------------------------------------------------


def test_build_evidence_plan_explicit_artifact_ref_activates_artifact() -> None:
    req = _req(artifact_ref=ArtifactRef(kind="pr", identifier="7"))
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_artifact
    assert plan.artifact_ref == ArtifactRef(kind="pr", identifier="7")


def test_build_evidence_plan_pr_scope_activates_artifact() -> None:
    req = _req(scope=ContextScope(pr_number=12))
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_artifact
    assert plan.artifact_ref is not None
    assert plan.artifact_ref.identifier == "12"


def test_build_evidence_plan_pr_scope_activates_discussions() -> None:
    req = _req(scope=ContextScope(pr_number=12))
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_discussions


def test_build_evidence_plan_no_pr_no_discussions() -> None:
    req = _req(query="Why does auth fail?")
    plan = build_evidence_plan(req, caps=_full_caps())
    assert not plan.run_discussions


def test_build_evidence_plan_artifact_ref_pr_kind_sets_pr_number_in_scope() -> None:
    req = _req(artifact_ref=ArtifactRef(kind="pr", identifier="99"))
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.scope.pr_number == 99


def test_build_evidence_plan_discussions_excluded() -> None:
    req = _req(scope=ContextScope(pr_number=5), exclude=["discussions"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert not plan.run_discussions


def test_build_evidence_plan_artifact_excluded_but_pr_scope_still_activates() -> None:
    # When artifact_ref.kind == "pr", the PR scope path activates artifact retrieval
    # even if "artifact" is excluded — PR scope is considered fundamental to context.
    req = _req(artifact_ref=ArtifactRef(kind="pr", identifier="5"), exclude=["artifact"])
    plan = build_evidence_plan(req, caps=_full_caps())
    # scope.pr_number is set from the artifact ref, which triggers the second activation path
    assert plan.scope.pr_number == 5
    assert plan.run_artifact  # PR scope activates artifact regardless of exclude


# ---------------------------------------------------------------------------
# Change history and decisions
# ---------------------------------------------------------------------------


def test_build_evidence_plan_recent_changes_include_activates_history() -> None:
    req = _req(include=["recent_changes"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_change_history


def test_build_evidence_plan_decisions_include_activates_decisions() -> None:
    req = _req(include=["decisions"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_decisions


def test_build_evidence_plan_query_history_keyword_activates_history() -> None:
    req = _req(query="Why was this method changed last quarter?")
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_change_history


def test_build_evidence_plan_recent_changes_excluded() -> None:
    req = _req(include=["recent_changes"], exclude=["recent_changes"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert not plan.run_change_history


def test_build_evidence_plan_decisions_excluded() -> None:
    req = _req(include=["decisions"], exclude=["decisions"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert not plan.run_decisions


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_build_evidence_plan_ownership_requires_file_path() -> None:
    req = _req(include=["owners"], scope=ContextScope(file_path=None))
    plan = build_evidence_plan(req, caps=_full_caps())
    assert not plan.run_ownership


def test_build_evidence_plan_ownership_activates_with_file_path() -> None:
    req = _req(include=["owners"], scope=ContextScope(file_path="src/auth.py"))
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_ownership


def test_build_evidence_plan_ownership_via_include() -> None:
    # owns/who keyword in query only triggers needs_ownership when the query
    # also mentions a file path. Use include=["owners"] instead for reliability.
    req = _req(
        query="Who is the owner?",
        include=["owners"],
        scope=ContextScope(file_path="src/auth.py"),
    )
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_ownership


def test_build_evidence_plan_ownership_keyword_and_file_in_query() -> None:
    # needs_ownership requires both an ownership keyword AND a file path in the query text
    req = _req(
        query="Who changed src/auth.py recently?",
        scope=ContextScope(file_path="src/auth.py"),
    )
    plan = build_evidence_plan(req, caps=_full_caps())
    # "Who" + "src/auth.py" in query triggers needs_ownership signal
    assert plan.run_ownership


# ---------------------------------------------------------------------------
# Project map and debugging memory
# ---------------------------------------------------------------------------


def test_build_evidence_plan_project_map_includes_filtered() -> None:
    req = _req(include=["purpose", "service_map", "prior_fixes"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_project_map
    assert "purpose" in plan.project_map_includes
    assert "service_map" in plan.project_map_includes
    assert "prior_fixes" not in plan.project_map_includes


def test_build_evidence_plan_debugging_memory_includes_filtered() -> None:
    req = _req(include=["prior_fixes", "diagnostic_signals", "purpose"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_debugging_memory
    assert "prior_fixes" in plan.debugging_memory_includes
    assert "diagnostic_signals" in plan.debugging_memory_includes
    assert "purpose" not in plan.debugging_memory_includes


def test_build_evidence_plan_project_map_requires_cap() -> None:
    req = _req(include=["purpose"])
    plan = build_evidence_plan(req, caps=_full_caps(project_map_context=False))
    assert not plan.run_project_map


def test_build_evidence_plan_debugging_memory_requires_cap() -> None:
    req = _req(include=["prior_fixes"])
    plan = build_evidence_plan(req, caps=_full_caps(debugging_memory_context=False))
    assert not plan.run_debugging_memory


def test_build_evidence_plan_causal_chain_from_include() -> None:
    req = _req(include=["causal_chain"])
    plan = build_evidence_plan(req, caps=_full_caps())
    assert plan.run_causal_chain


def test_build_evidence_plan_causal_chain_requires_cap() -> None:
    req = _req(include=["causal_chain"])
    plan = build_evidence_plan(req, caps=_full_caps(causal_chain_context=False))
    assert not plan.run_causal_chain


# ---------------------------------------------------------------------------
# Timeout clamping
# ---------------------------------------------------------------------------


def test_build_evidence_plan_timeout_clamped_to_minimum() -> None:
    from domain.intelligence_models import ContextBudget

    req = _req(budget=ContextBudget(timeout_ms=0))
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.timeout_budget_ms >= 500


def test_build_evidence_plan_timeout_clamped_to_maximum() -> None:
    from domain.intelligence_models import ContextBudget

    req = _req(budget=ContextBudget(timeout_ms=999_999))
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.timeout_budget_ms <= 30_000


def test_build_evidence_plan_timeout_within_range_is_preserved() -> None:
    from domain.intelligence_models import ContextBudget

    req = _req(budget=ContextBudget(timeout_ms=6_000))
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.timeout_budget_ms == 6_000


# ---------------------------------------------------------------------------
# Scope merging from signals
# ---------------------------------------------------------------------------


def test_build_evidence_plan_scope_merges_file_path_from_query() -> None:
    req = _req(query="What changed in src/auth/token.py recently?")
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.scope.file_path == "src/auth/token.py"


def test_build_evidence_plan_scope_explicit_file_path_not_overridden() -> None:
    req = _req(
        query="What changed in src/auth/token.py recently?",
        scope=ContextScope(file_path="src/other.py"),
    )
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.scope.file_path == "src/other.py"


def test_build_evidence_plan_scope_merges_pr_number_from_signals() -> None:
    req = _req(query="Tell me about PR #77")
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.scope.pr_number == 77


def test_build_evidence_plan_scope_explicit_pr_not_overridden() -> None:
    req = _req(
        query="Tell me about PR #77",
        scope=ContextScope(pr_number=5),
    )
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.scope.pr_number == 5


# ---------------------------------------------------------------------------
# Mandatory fields
# ---------------------------------------------------------------------------


def test_build_evidence_plan_pr_scope_adds_mandatory_artifact_context() -> None:
    req = _req(scope=ContextScope(pr_number=3))
    plan = build_evidence_plan(req, caps=_no_caps())
    assert "artifact_context" in plan.mandatory
    assert "discussion_context" in plan.mandatory


def test_build_evidence_plan_history_file_adds_mandatory_change_history() -> None:
    req = _req(
        query="Why was this file changed?",
        scope=ContextScope(file_path="src/foo.py"),
    )
    plan = build_evidence_plan(req, caps=_no_caps())
    assert "change_history" in plan.mandatory


def test_build_evidence_plan_no_special_scope_no_mandatory() -> None:
    req = _req(query="What is this project?")
    plan = build_evidence_plan(req, caps=_no_caps())
    assert plan.mandatory == []


# ---------------------------------------------------------------------------
# Pre-extracted signals
# ---------------------------------------------------------------------------


def test_build_evidence_plan_accepts_pre_extracted_signals() -> None:
    req = _req()
    signals = SignalSet(mentioned_pr=5, needs_history=True)
    plan = build_evidence_plan(req, signals=signals, caps=_full_caps())
    assert plan.scope.pr_number == 5
    assert plan.run_artifact


def test_build_evidence_plan_null_signals_falls_back_to_extraction() -> None:
    req = _req(query="Tell me about PR #8")
    plan = build_evidence_plan(req, signals=None, caps=_full_caps())
    assert plan.scope.pr_number == 8

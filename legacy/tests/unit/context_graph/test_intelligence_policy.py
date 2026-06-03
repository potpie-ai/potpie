"""Evidence planning from signals + provider capabilities."""

from __future__ import annotations

import pytest

from domain.intelligence_models import (
    ArtifactRef,
    CapabilitySet,
    ContextBudget,
    ContextResolutionRequest,
    ContextScope,
)
from domain.intelligence_policy import EvidencePlan, build_evidence_plan
from domain.intelligence_signals import extract_signals

pytestmark = pytest.mark.unit


def _request(**overrides) -> ContextResolutionRequest:
    base = {"pot_id": "p1", "query": "anything"}
    base.update(overrides)
    return ContextResolutionRequest(**base)


def _full_caps() -> CapabilitySet:
    return CapabilitySet(
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


# --- Scope merging ---------------------------------------------------------


class TestScopeMerging:
    def test_signal_file_paths_fill_missing_scope(self) -> None:
        req = _request(query="see app/auth.py for context")
        plan = build_evidence_plan(req)
        assert plan.scope.file_path == "app/auth.py"

    def test_explicit_scope_wins_over_signals(self) -> None:
        scope = ContextScope(file_path="explicit.py")
        req = _request(query="see app/auth.py", scope=scope)
        plan = build_evidence_plan(req)
        assert plan.scope.file_path == "explicit.py"

    def test_pr_signal_populates_scope(self) -> None:
        req = _request(query="look at PR #42")
        plan = build_evidence_plan(req)
        assert plan.scope.pr_number == 42

    def test_artifact_ref_pr_overrides_pr_signal(self) -> None:
        req = _request(
            query="look at PR #99",
            artifact_ref=ArtifactRef(kind="pr", identifier="42"),
        )
        plan = build_evidence_plan(req)
        assert plan.scope.pr_number == 42

    def test_artifact_ref_with_invalid_pr_id_keeps_signal_pr(self) -> None:
        req = _request(
            query="look at PR #99",
            artifact_ref=ArtifactRef(kind="pr", identifier="not-an-int"),
        )
        plan = build_evidence_plan(req)
        assert plan.scope.pr_number == 99

    def test_symbol_signal_fills_function_name(self) -> None:
        req = _request(query="explain ProcessLog please")
        plan = build_evidence_plan(req)
        # ProcessLog is camelcase → first symbol → function_name
        assert plan.scope.function_name == "ProcessLog"


# --- Capabilities gating ---------------------------------------------------


class TestCapabilitiesGating:
    def test_caps_off_disables_structural_runs(self) -> None:
        # When change_history / decision_context capability is off, the structural
        # phases stay off even when ``unknown`` intent's defaults would include them.
        plan = build_evidence_plan(
            _request(),
            caps=CapabilitySet(change_history=False, decision_context=False),
        )
        assert plan.run_change_history is False
        assert plan.run_decisions is False
        # Semantic search defaults off because the cap is off.
        assert plan.run_semantic_search is False
        # No PR scope → no artifact / discussions.
        assert plan.run_artifact is False
        assert plan.run_discussions is False
        assert plan.run_ownership is False

    def test_semantic_search_runs_when_capable(self) -> None:
        plan = build_evidence_plan(_request(), caps=_full_caps())
        assert plan.run_semantic_search is True

    def test_semantic_search_excluded(self) -> None:
        req = _request(exclude=["semantic_search"])
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_semantic_search is False

    def test_pr_scope_runs_artifact_and_discussions(self) -> None:
        req = _request(query="look at PR #5")
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_artifact is True
        assert plan.run_discussions is True
        assert plan.artifact_ref == ArtifactRef(kind="pr", identifier="5")

    def test_explicit_artifact_ref_runs_artifact_even_without_pr_scope(self) -> None:
        req = _request(artifact_ref=ArtifactRef(kind="issue", identifier="11"))
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_artifact is True
        assert plan.artifact_ref.identifier == "11"

    def test_artifact_excluded_blocks_non_pr_artifact_ref(self) -> None:
        # With a non-PR artifact ref, ``exclude=["artifact"]`` shuts off the
        # artifact path entirely.
        req = _request(
            artifact_ref=ArtifactRef(kind="issue", identifier="11"),
            exclude=["artifact"],
        )
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_artifact is False

    def test_ownership_requires_path_and_keyword(self) -> None:
        # No path → no ownership.
        req_no_path = _request(query="who owns this")
        plan_no_path = build_evidence_plan(req_no_path, caps=_full_caps())
        assert plan_no_path.run_ownership is False

        req = _request(query="who owns app/auth.py")
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_ownership is True

    def test_project_map_includes_filtered_to_canonical(self) -> None:
        req = _request(include=["repo_map", "purpose", "not_a_real_one"])
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_project_map is True
        assert "repo_map" in plan.project_map_includes
        assert "purpose" in plan.project_map_includes
        assert "not_a_real_one" not in plan.project_map_includes

    def test_debugging_memory_includes_filtered(self) -> None:
        req = _request(include=["incidents", "alerts"])
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_debugging_memory is True
        assert set(plan.debugging_memory_includes) == {"incidents", "alerts"}

    def test_causal_chain_runs_when_included_and_capable(self) -> None:
        req = _request(include=["causal_chain"])
        plan = build_evidence_plan(req, caps=_full_caps())
        assert plan.run_causal_chain is True


# --- Mode + budget ---------------------------------------------------------


class TestModeAndBudget:
    def test_default_timeout_floor(self) -> None:
        # 4000ms default; clamp range [500, 30_000].
        plan = build_evidence_plan(_request())
        assert plan.timeout_budget_ms == 4000

    def test_too_small_timeout_clamped_up(self) -> None:
        req = _request(budget=ContextBudget(timeout_ms=100))
        plan = build_evidence_plan(req)
        assert plan.timeout_budget_ms == 500

    def test_too_large_timeout_clamped_down(self) -> None:
        req = _request(budget=ContextBudget(timeout_ms=99_999))
        plan = build_evidence_plan(req)
        assert plan.timeout_budget_ms == 30_000

    def test_deep_mode_raises_minimum_to_8000(self) -> None:
        req = _request(mode="deep")
        plan = build_evidence_plan(req)
        assert plan.timeout_budget_ms >= 8_000

    def test_verify_mode_raises_minimum_to_6000(self) -> None:
        req = _request(mode="verify")
        plan = build_evidence_plan(req)
        assert plan.timeout_budget_ms >= 6_000

    def test_deep_mode_doubles_max_items(self) -> None:
        req = _request(mode="deep", budget=ContextBudget(max_items=10))
        plan = build_evidence_plan(req)
        assert plan.max_items == 20

    def test_verify_with_references_only_sets_source_policy(self) -> None:
        req = _request(mode="verify", source_policy="references_only")
        plan = build_evidence_plan(req)
        assert plan.source_policy == "verify"

    def test_non_verify_mode_leaves_source_policy_unset(self) -> None:
        req = _request(mode="fast", source_policy="references_only")
        plan = build_evidence_plan(req)
        assert plan.source_policy is None


# --- Mandatory list --------------------------------------------------------


class TestMandatoryList:
    def test_pr_scope_marks_artifact_and_discussions_mandatory(self) -> None:
        req = _request(query="look at PR #5")
        plan = build_evidence_plan(req, caps=_full_caps())
        assert "artifact_context" in plan.mandatory
        assert "discussion_context" in plan.mandatory

    def test_history_intent_with_path_marks_change_history_mandatory(self) -> None:
        req = _request(query="why did we change app/auth.py")
        plan = build_evidence_plan(req, caps=_full_caps())
        assert "change_history" in plan.mandatory

    def test_deep_mode_marks_every_activated_family_mandatory(self) -> None:
        req = _request(query="look at PR #5", mode="deep", include=["causal_chain"])
        plan = build_evidence_plan(req, caps=_full_caps())
        # Every "run_*" family that fired must be listed in ``mandatory``.
        assert "artifact_context" in plan.mandatory
        assert "discussion_context" in plan.mandatory
        assert "causal_chain_context" in plan.mandatory
        # Deduped — no repeats.
        assert len(plan.mandatory) == len(set(plan.mandatory))


class TestSignalsArgumentPassthrough:
    def test_passing_pre_extracted_signals_is_supported(self) -> None:
        req = _request(query="look at PR #5")
        signals = extract_signals(req.query)
        plan = build_evidence_plan(req, signals=signals, caps=_full_caps())
        assert plan.scope.pr_number == 5


def test_evidence_plan_has_default_values() -> None:
    """``EvidencePlan`` is a dataclass — confirm default field values exist."""
    plan = EvidencePlan()
    assert plan.run_semantic_search is False
    assert plan.timeout_budget_ms == 4000
    assert plan.mandatory == []
    assert plan.project_map_includes == []

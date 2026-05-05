"""Env-backed reconciliation feature flags."""

from __future__ import annotations

import pytest

from domain import reconciliation_flags as flags

pytestmark = pytest.mark.unit


# Each (env_var, function, default) tuple wires one flag to its env knob.
FLAG_TABLE: list[tuple[str, str, bool]] = [
    ("CONTEXT_ENGINE_RECONCILIATION_ENABLED", "reconciliation_enabled", True),
    ("CONTEXT_ENGINE_AGENT_PLANNER_ENABLED", "agent_planner_enabled", True),
    ("CONTEXT_ENGINE_INFER_LABELS", "infer_canonical_labels_enabled", True),
    ("CONTEXT_ENGINE_CONFLICT_DETECT", "conflict_detection_enabled", True),
    ("CONTEXT_ENGINE_AUTO_SUPERSEDE", "auto_supersede_enabled", True),
    ("CONTEXT_ENGINE_CAUSAL_EXPAND", "causal_expand_enabled", True),
    ("CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES", "classify_modified_edges_enabled", True),
    ("CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE", "allow_edge_classify_write_enabled", True),
    ("CONTEXT_ENGINE_STRICT_EXTRACTION", "strict_extraction_enabled", True),
    ("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", "ontology_soft_fail_enabled", False),
    ("CONTEXT_ENGINE_ONTOLOGY_STRICT", "ontology_strict_enabled", False),
]


class TestTruthyParsing:
    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", " On ", "Yes"])
    def test_truthy_values_parse_true(self, raw: str, monkeypatch) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", raw)
        assert flags.ontology_soft_fail_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE", " Off "])
    def test_falsy_values_parse_false(self, raw: str, monkeypatch) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", raw)
        assert flags.reconciliation_enabled() is False

    def test_empty_string_collapses_to_false(self, monkeypatch) -> None:
        # An empty string is explicitly mapped to False in _truthy.
        monkeypatch.setenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", "")
        assert flags.reconciliation_enabled() is False

    @pytest.mark.parametrize("raw", ["maybe", "2", "weird", "y"])
    def test_unrecognized_value_falls_back_to_default(self, raw: str, monkeypatch) -> None:
        monkeypatch.setenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", raw)
        assert flags.reconciliation_enabled() is True
        monkeypatch.setenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", raw)
        assert flags.ontology_soft_fail_enabled() is False

    def test_unset_var_returns_default(self, monkeypatch) -> None:
        monkeypatch.delenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", raising=False)
        assert flags.reconciliation_enabled() is True
        monkeypatch.delenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", raising=False)
        assert flags.ontology_soft_fail_enabled() is False


class TestAllFlagsObserveEnv:
    @pytest.mark.parametrize("env_var,fn_name,default", FLAG_TABLE)
    def test_default_when_unset(self, env_var: str, fn_name: str, default: bool, monkeypatch) -> None:
        monkeypatch.delenv(env_var, raising=False)
        assert getattr(flags, fn_name)() is default

    @pytest.mark.parametrize("env_var,fn_name,default", FLAG_TABLE)
    def test_can_be_inverted_via_env(self, env_var: str, fn_name: str, default: bool, monkeypatch) -> None:
        monkeypatch.setenv(env_var, "false" if default else "true")
        assert getattr(flags, fn_name)() is (not default)

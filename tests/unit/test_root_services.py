from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from potpie.runtime.composition import LocalRuntimeComposition, build_local_runtime
from potpie.runtime.local_engine import LocalEngineServices
from potpie.runtime.root_services import (
    RootRuntimeServices,
    build_pot_resource_service,
    build_root_runtime_services,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)


def test_local_runtime_composition_separates_root_and_engine_services(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    runtime = build_local_runtime(backend=InMemoryGraphBackend())

    assert isinstance(runtime, LocalRuntimeComposition)
    assert isinstance(runtime.root, RootRuntimeServices)
    assert isinstance(runtime.engine, LocalEngineServices)
    assert runtime.root.backend is runtime.engine.backend
    assert not hasattr(runtime.root, "graph")
    assert not hasattr(runtime.engine, "setup")


def test_pot_resource_service_exposes_only_finite_control_plane_operations() -> None:
    backend = MagicMock()
    backend.list_pots.return_value = ["pot-1"]
    backend.clear_repo_default.return_value = True
    service = build_pot_resource_service(SimpleNamespace(pots=backend))

    assert service.list_pots() == ["pot-1"]
    assert service.clear_repo_default(repo="repo") is True
    backend.list_pots.assert_called_once_with()
    backend.clear_repo_default.assert_called_once_with(repo="repo")
    with pytest.raises(AttributeError):
        getattr(service, "graph")


def test_root_runtime_excludes_context_engine_surfaces() -> None:
    host = SimpleNamespace(
        pots=SimpleNamespace(),
        backend="backend",
        auth="auth",
        config="config",
        daemon="daemon",
        installer="installer",
        ledger="ledger",
        setup="setup",
        skills="skills",
        profile="local",
        agent_context="legacy-agent-context",
        graph="legacy-graph",
        graph_workbench="legacy-workbench",
        nudge="legacy-nudge",
    )

    runtime = build_root_runtime_services(host)

    assert runtime.profile == "local"
    for name in ("agent_context", "graph", "graph_workbench", "nudge"):
        with pytest.raises(AttributeError):
            getattr(runtime, name)


def test_create_pot_preserves_legacy_optional_repo_call_shape() -> None:
    backend = MagicMock()
    service = build_pot_resource_service(SimpleNamespace(pots=backend))

    service.create_pot(name="plain", use=True)
    service.create_pot(name="linked", repo="owner/repo", use=False)

    assert backend.create_pot.call_args_list == [
        (( ), {"name": "plain", "use": True}),
        (( ), {"name": "linked", "repo": "owner/repo", "use": False}),
    ]


def test_repo_default_capability_is_explicit() -> None:
    unsupported = build_pot_resource_service(
        SimpleNamespace(pots=SimpleNamespace())
    )
    supported = build_pot_resource_service(
        SimpleNamespace(pots=SimpleNamespace(set_repo_default=lambda **_kwargs: None))
    )

    assert unsupported.supports_repo_defaults is False
    assert supported.supports_repo_defaults is True

from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from potpie.runtime.root_services import build_pot_resource_service


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

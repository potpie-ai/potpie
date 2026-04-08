"""Provider registry and bootstrap (Phase 1 integrations platform)."""

import pytest

from integrations.application.bootstrap import (
    load_providers,
    reset_load_providers_for_tests,
)
from integrations.domain.provider_definitions import ProviderDefinition
from integrations.domain.provider_registry import (
    ProviderRegistry,
    get_provider_registry,
    reset_provider_registry_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_provider_registry_for_tests()
    reset_load_providers_for_tests()
    yield
    reset_provider_registry_for_tests()
    reset_load_providers_for_tests()


def test_load_providers_registers_github_and_linear():
    load_providers()
    reg = get_provider_registry()
    gh = reg.get("github")
    assert gh is not None
    assert gh.display_name == "GitHub"
    assert gh.port_kind == "source_control"
    assert "code_host" in gh.capabilities
    assert "repository" in gh.source_kinds
    assert gh.oss_available is True
    ln = reg.get("linear")
    assert ln is not None
    assert ln.display_name == "Linear"
    assert ln.port_kind == "issue_tracker"
    assert "issue_tracker" in ln.capabilities
    assert "issue_tracker_team" in ln.source_kinds
    assert ln.oss_available is True


def test_load_providers_is_idempotent():
    load_providers()
    load_providers()
    reg = get_provider_registry()
    assert len(reg.list_all()) == 2


def test_register_duplicate_raises():
    reg = ProviderRegistry()
    reg.register(
        ProviderDefinition(
            id="x",
            display_name="X",
            capabilities=(),
            source_kinds=(),
            port_kind="issue_tracker",
        )
    )
    with pytest.raises(ValueError, match="already registered"):
        reg.register(
            ProviderDefinition(
                id="x",
                display_name="X2",
                capabilities=(),
                source_kinds=(),
                port_kind="issue_tracker",
            )
        )

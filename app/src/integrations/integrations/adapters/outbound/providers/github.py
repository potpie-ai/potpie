"""Register GitHub as the OSS code-host provider."""

from __future__ import annotations

from integrations.domain.provider_definitions import ProviderDefinition
from integrations.domain.provider_registry import ProviderRegistry


def register_github_provider(registry: ProviderRegistry) -> None:
    registry.register(
        ProviderDefinition(
            id="github",
            display_name="GitHub",
            capabilities=("code_host",),
            source_kinds=("repository",),
            port_kind="source_control",
            oss_available=True,
        )
    )

"""Register Linear as the OSS issue-tracker provider (catalog entry)."""

from __future__ import annotations

from integrations.domain.provider_definitions import ProviderDefinition
from integrations.domain.provider_registry import ProviderRegistry


def register_linear_provider(registry: ProviderRegistry) -> None:
    registry.register(
        ProviderDefinition(
            id="linear",
            display_name="Linear",
            capabilities=("issue_tracker",),
            source_kinds=("issue_tracker_team",),
            port_kind="issue_tracker",
            oss_available=True,
        )
    )

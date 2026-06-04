"""In-process registry of integration providers (GitHub OSS + optional commercial plugins)."""

from __future__ import annotations

from integrations.domain.provider_definitions import ProviderDefinition

_registry: ProviderRegistry | None = None


class ProviderRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, ProviderDefinition] = {}

    def register(self, definition: ProviderDefinition) -> None:
        if definition.id in self._by_id:
            raise ValueError(f"Provider already registered: {definition.id!r}")
        self._by_id[definition.id] = definition

    def get(self, provider_id: str) -> ProviderDefinition | None:
        return self._by_id.get(provider_id)

    def list_all(self) -> list[ProviderDefinition]:
        return sorted(self._by_id.values(), key=lambda d: d.display_name.lower())

    def list_available(self) -> list[ProviderDefinition]:
        """Providers that are present in this process (OSS + loaded commercial)."""
        return self.list_all()


def get_provider_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def reset_provider_registry_for_tests() -> None:
    """Clear singleton; tests only."""
    global _registry
    _registry = None

"""Load provider registrations once per process (API + Celery worker)."""

from __future__ import annotations

import logging

from integrations.domain.provider_registry import ProviderRegistry, get_provider_registry
from integrations.adapters.outbound.providers.github import register_github_provider
from integrations.adapters.outbound.providers.linear import register_linear_provider

logger = logging.getLogger(__name__)

_loaded = False


def load_providers() -> None:
    """Idempotent: register OSS providers and optional commercial package."""
    global _loaded
    if _loaded:
        return
    registry = get_provider_registry()
    register_github_provider(registry)
    register_linear_provider(registry)
    _try_register_commercial(registry)
    _loaded = True
    logger.info(
        "Integration providers loaded: %s",
        ", ".join(p.id for p in registry.list_all()) or "(none)",
    )


def _try_register_commercial(registry: ProviderRegistry) -> None:
    try:
        import potpie_integrations_commercial  # type: ignore[import-not-found]
    except ImportError:
        return
    register = getattr(potpie_integrations_commercial, "register_providers", None)
    if not callable(register):
        return
    try:
        register(registry)
    except Exception:
        logger.exception("Commercial integrations register_providers() failed")


def reset_load_providers_for_tests() -> None:
    """Tests only."""
    global _loaded
    _loaded = False

"""Application entry point for context intelligence resolution."""

from __future__ import annotations

from domain.intelligence_models import ContextResolutionRequest, IntelligenceBundle
from application.services.context_resolution import ContextResolutionService


async def resolve_context(
    service: ContextResolutionService,
    request: ContextResolutionRequest,
) -> IntelligenceBundle:
    """Resolve contextual evidence for a query within a project."""
    return await service.resolve(request)

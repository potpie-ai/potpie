from fastapi import APIRouter

from potpie_context_engine.adapters.inbound.http.webhooks.integrations.github import (
    github_router,
)
from potpie_context_engine.adapters.inbound.http.webhooks.integrations.gitlab import (
    gitlab_router,
)

webhooks_router = APIRouter(tags=["webhooks"])
webhooks_router.include_router(github_router, prefix="/integrations")
webhooks_router.include_router(gitlab_router, prefix="/integrations")

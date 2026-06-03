from fastapi import APIRouter

from adapters.inbound.http.webhooks.integrations.github import github_router

webhooks_router = APIRouter(tags=["webhooks"])
webhooks_router.include_router(github_router, prefix="/integrations")

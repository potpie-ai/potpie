from fastapi import APIRouter

from adapters.inbound.http.api.v1.context.router import context_router

api_router = APIRouter()
api_router.include_router(context_router, prefix="/context", tags=["context"])


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

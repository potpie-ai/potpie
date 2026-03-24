"""HTTP API: REST under /api/v1; webhooks at /webhooks."""

import importlib.metadata

from fastapi import FastAPI

from adapters.inbound.http.api.router import api_router
from adapters.inbound.http.webhooks.router import webhooks_router

try:
    __version__ = importlib.metadata.version("context-engine")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


def create_app() -> FastAPI:
    app = FastAPI(title="context-engine", version=__version__)
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(webhooks_router, prefix="/webhooks")
    return app


app = create_app()

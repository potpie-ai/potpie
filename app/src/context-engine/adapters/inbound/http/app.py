"""HTTP API: REST under /api/v1; webhooks at /webhooks."""

import importlib.metadata
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from adapters.inbound.http._hardening import install_hardening
from adapters.inbound.http.api.router import api_router
from adapters.inbound.http.webhooks.router import webhooks_router
from bootstrap.observability import configure_observability

try:
    __version__ = importlib.metadata.version("context-engine")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_observability()
    app = FastAPI(title="context-engine", version=__version__)

    # Attach the observability request-context middleware so every log
    # inside a request automatically carries request_id / path / method
    # (and user_id when auth populates request.state.user).
    try:
        from observability.integrations.fastapi import LoggingContextMiddleware

        app.add_middleware(LoggingContextMiddleware)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("LoggingContextMiddleware not attached: %s", exc)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        if isinstance(exc.detail, str):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": "http_error",
                        "message": exc.detail,
                    }
                },
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled server error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                }
            },
        )

    install_hardening(app)
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(webhooks_router, prefix="/webhooks")
    return app


app = create_app()

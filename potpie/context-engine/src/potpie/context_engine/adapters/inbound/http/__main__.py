import logging
import os
import sys

import uvicorn

from potpie.context_engine.bootstrap.logging_setup import configure_logging

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _assert_safe_to_bind(host: str) -> None:
    """Fail closed: do not expose an unauthenticated listener on a network.

    The standalone HTTP surface authenticates via ``CONTEXT_ENGINE_API_KEY``
    and enforces per-actor pot scoping via the policy contract. When neither
    a key nor the dev-only ``CONTEXT_ENGINE_ALLOW_NO_AUTH`` opt-in is set,
    binding to anything other than loopback would publish a fully open,
    cross-tenant API — so we refuse to start.
    """
    has_key = bool(os.getenv("CONTEXT_ENGINE_API_KEY", "").strip())
    allow_no_auth = os.getenv("CONTEXT_ENGINE_ALLOW_NO_AUTH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if host in _LOOPBACK_HOSTS or has_key or allow_no_auth:
        return
    logger.error(
        "Refusing to start: CONTEXT_ENGINE_HOST=%s is network-reachable but "
        "CONTEXT_ENGINE_API_KEY is unset. Configure an API key, bind to "
        "127.0.0.1, or set CONTEXT_ENGINE_ALLOW_NO_AUTH=1 for local dev only.",
        host,
    )
    sys.exit(1)


def main() -> None:
    configure_logging()
    host = os.environ.get("CONTEXT_ENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("CONTEXT_ENGINE_PORT", "8000"))
    reload = os.environ.get("CONTEXT_ENGINE_RELOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )
    _assert_safe_to_bind(host)
    uvicorn.run(
        "potpie.context_engine.adapters.inbound.http.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()

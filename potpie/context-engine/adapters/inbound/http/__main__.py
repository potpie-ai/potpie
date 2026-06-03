import os

import uvicorn


def main() -> None:
    host = os.environ.get("CONTEXT_ENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("CONTEXT_ENGINE_PORT", "8000"))
    reload = os.environ.get("CONTEXT_ENGINE_RELOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )
    uvicorn.run(
        "adapters.inbound.http.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()

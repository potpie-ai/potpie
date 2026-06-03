# Potpie Observability

Potpie Observability is a small logging and tracing layer used across Potpie.
The core package is intentionally dependency-light, with optional extras for
Sentry, Logfire, FastAPI, Celery, and Loguru integrations.

The Python distribution is published as `potpie-observability` and exposes the
`observability` import package.

## Installation

```bash
pip install potpie-observability
```

Optional integrations can be installed as extras:

```bash
pip install "potpie-observability[loguru,sentry,logfire,fastapi,celery]"
```

## Links

- Homepage: https://potpie.ai
- Documentation: https://docs.potpie.ai
- Source: https://github.com/potpie-ai/potpie

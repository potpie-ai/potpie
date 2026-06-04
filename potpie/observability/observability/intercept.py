"""Library-level control for ambient stdlib loggers.

stdlib-core means we do NOT need a global stdlib->loguru bridge: third-party
loggers (uvicorn, sqlalchemy, httpx, celery, ...) propagate to root, where
configure() attached our managed handlers — so they flow through our sinks +
redaction + context with zero caller changes.

This module only dials per-library verbosity and stops libraries that attach
their OWN handlers (uvicorn) from double-emitting, by clearing those handlers
and forcing propagation.

DEFAULT_LIBRARY_LEVELS is the verbatim map ported from
app/modules/utils/logger.py.
"""

from __future__ import annotations

import logging

DEFAULT_LIBRARY_LEVELS: dict[str, str] = {
    "uvicorn": "INFO",
    "uvicorn.access": "WARNING",
    "uvicorn.error": "INFO",
    "fastapi": "INFO",
    "sqlalchemy.engine": "WARNING",
    "sqlalchemy.pool": "WARNING",
    "sqlalchemy.orm": "WARNING",
    "alembic": "INFO",
    "celery": "INFO",
    "kombu": "WARNING",
    "httpx": "WARNING",
    "urllib3": "WARNING",
    "boto3": "WARNING",
    "botocore": "WARNING",
}


def apply_library_levels(levels: dict[str, str] | None) -> None:
    """Set levels and route the named libraries through root (our handlers).

    Empty/None -> use DEFAULT_LIBRARY_LEVELS so callers get sane defaults.
    Custom levels override/extend those defaults instead of replacing them.
    """
    mapping = {**DEFAULT_LIBRARY_LEVELS, **(levels or {})}
    for name, level in mapping.items():
        lib = logging.getLogger(name)
        lib.setLevel(level)
        lib.handlers = []          # don't let the library emit on its own
        lib.propagate = True       # send to root -> our managed sinks

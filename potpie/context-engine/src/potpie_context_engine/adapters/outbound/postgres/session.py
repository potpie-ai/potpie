"""Optional SQLAlchemy engine/session for standalone HTTP."""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def database_url() -> str | None:
    """Resolve sync SQLAlchemy URL (Potpie uses ``POSTGRES_SERVER`` in ``.env``)."""
    return (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("CONTEXT_ENGINE_DATABASE_URL")
        or os.getenv("POSTGRES_SERVER")
    )


def make_session_factory(url: str | None = None):
    u = url or database_url()
    if not u:
        return None
    engine = create_engine(u, pool_pre_ping=True)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


def session_scope(factory) -> Generator[Session, None, None]:
    session = factory()
    try:
        yield session
    finally:
        session.close()

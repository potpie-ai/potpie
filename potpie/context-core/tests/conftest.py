"""Shared fixtures for the temporary Context Core compatibility suite."""

from __future__ import annotations

import pytest


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"

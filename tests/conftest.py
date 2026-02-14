"""
Root conftest for all tests.

This conftest only contains minimal shared configuration.
- Unit tests (tests/unit/) use their own minimal conftest
- Integration tests (tests/integration/) use full app context conftest

This separation allows unit tests to run without external dependencies.
"""

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "github_live: marks tests that hit the live GitHub API"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires services)"
    )

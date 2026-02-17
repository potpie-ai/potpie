import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Override global DB bootstrap for pure unit tests under tests/unit/intelligence."""
    yield

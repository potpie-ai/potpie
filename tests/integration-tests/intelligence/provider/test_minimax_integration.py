"""Integration tests for MiniMax direct API provider.

These tests make real API calls to https://api.minimax.io/v1 and require
the MINIMAX_API_KEY environment variable to be set.

Run with:
    pytest tests/integration-tests/intelligence/provider/test_minimax_integration.py -v
"""

import os
import pytest
import pytest_asyncio
from unittest.mock import MagicMock

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set — skipping MiniMax integration tests",
)


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = None
    return db


@pytest.fixture
def provider_service(mock_db):
    from app.modules.intelligence.provider.provider_service import ProviderService

    return ProviderService(mock_db, "integration-test-user")


@pytest.mark.asyncio
async def test_minimax_m27_chat_completion(provider_service):
    """Verify a simple chat completion via direct MiniMax M2.7 API."""
    messages = [
        {"role": "user", "content": "Reply with exactly: HELLO_MINIMAX"},
    ]
    response = await provider_service.call_llm_with_specific_model(
        model_identifier="minimax/MiniMax-M2.7",
        messages=messages,
    )
    assert isinstance(response, str)
    assert "HELLO_MINIMAX" in response


@pytest.mark.asyncio
async def test_minimax_m27_highspeed_chat_completion(provider_service):
    """Verify a simple chat completion via direct MiniMax M2.7-highspeed API."""
    messages = [
        {"role": "user", "content": "Reply with exactly: FAST_MINIMAX"},
    ]
    response = await provider_service.call_llm_with_specific_model(
        model_identifier="minimax/MiniMax-M2.7-highspeed",
        messages=messages,
    )
    assert isinstance(response, str)
    assert "FAST_MINIMAX" in response


@pytest.mark.asyncio
async def test_minimax_streaming(provider_service):
    """Verify streaming works with the direct MiniMax API."""
    messages = [
        {"role": "user", "content": "Count from 1 to 3, separated by commas."},
    ]
    generator = await provider_service.call_llm_with_specific_model(
        model_identifier="minimax/MiniMax-M2.7-highspeed",
        messages=messages,
        stream=True,
    )
    chunks = []
    async for chunk in generator:
        chunks.append(chunk)
    full_response = "".join(chunks)
    assert "1" in full_response
    assert "2" in full_response
    assert "3" in full_response

from __future__ import annotations

# ruff: noqa: S101 - pytest assertions.

from typing import cast
from unittest.mock import AsyncMock, create_autospec

import pytest

from potpie.runtime.clients import _invoke_engine
from potpie.runtime.operations import ENGINE_OPERATION_CATALOG, EngineOperation
from potpie_context_engine import ContextEngine, Success


@pytest.mark.anyio
@pytest.mark.parametrize("operation", EngineOperation)
async def test_dispatch_reaches_the_named_engine_method(
    operation: EngineOperation,
) -> None:
    engine = create_autospec(ContextEngine, instance=True)
    method_name = (
        "slice" if operation is EngineOperation.GRAPH_SLICE else operation.value
    )
    method = cast(AsyncMock, getattr(engine, method_name))
    expected = Success(object())
    method.return_value = expected
    request = ENGINE_OPERATION_CATALOG[operation].request_type()

    result = await _invoke_engine(engine, operation, request)

    assert result is expected
    method.assert_awaited_once_with(request)
    assert len(engine.mock_calls) == 1

"""Injected legacy writers cannot claim native durable execution guarantees."""

from types import SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (
    FalkorDBGraphBackend,
)
from potpie_context_engine.core.errors import CapabilityNotImplemented
from potpie_context_engine.core.mutation_execution import MutationCapabilities
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = pytest.mark.unit


def test_custom_writer_recovery_is_unsupported_after_recreation():
    for _ in range(2):
        backend = FalkorDBGraphBackend(
            SimpleNamespace(), writer=SimpleNamespace(enabled=True)
        )
        assert backend.mutation.capabilities == MutationCapabilities()
        lookup = backend.mutation.lookup_execution(
            MutationBatch(), expected_pot_id="p", mutation_id="unknown"
        )
        assert lookup.state == "unsupported"
        assert lookup.result is None
        with pytest.raises(CapabilityNotImplemented):
            backend.mutation.current_version("p")
        with pytest.raises(CapabilityNotImplemented):
            backend.mutation.compare_and_apply(
                MutationBatch(), expected_pot_id="p", expected_version=0
            )


@pytest.mark.asyncio
async def test_custom_writer_async_recovery_is_explicitly_unsupported():
    backend = FalkorDBGraphBackend(
        SimpleNamespace(), writer=SimpleNamespace(enabled=True)
    )
    lookup = await backend.mutation.lookup_execution_async(
        MutationBatch(), expected_pot_id="p", mutation_id="unknown"
    )
    assert lookup.state == "unsupported"
    with pytest.raises(CapabilityNotImplemented):
        await backend.mutation.compare_and_apply_async(
            MutationBatch(), expected_pot_id="p", expected_version=0
        )

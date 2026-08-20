from __future__ import annotations

import pytest

from potpie_context_engine.core.lifecycle import SetupPlan as CanonicalSetupPlan
from potpie_context_core.lifecycle import SetupPlan
from potpie.daemon.rpc import TYPE_KEY, decode, encode


def test_daemon_rpc_roundtrips_domain_dataclasses() -> None:
    plan = SetupPlan(
        backend="embedded",
        repo="potpie",
        pot="default",
        agent="claude",
        assume_yes=True,
    )

    round_tripped = decode(encode(plan))

    assert round_tripped == plan  # noqa: S101
    assert type(round_tripped) is CanonicalSetupPlan  # noqa: S101


def test_daemon_rpc_rejects_non_domain_class_references() -> None:
    with pytest.raises(TypeError, match="RPC class module not allowed"):
        decode(
            {
                TYPE_KEY: "dataclass",
                "class": "os:stat_result",
                "value": {},
            }
        )

from __future__ import annotations

import pytest

from context_engine.domain.lifecycle import SetupPlan
from context_engine.host.daemon_rpc import TYPE_KEY, decode, encode


def test_daemon_rpc_roundtrips_domain_dataclasses() -> None:
    plan = SetupPlan(
        backend="embedded",
        repo="potpie",
        pot="default",
        agent="claude",
        assume_yes=True,
    )

    assert decode(encode(plan)) == plan


def test_daemon_rpc_rejects_non_domain_class_references() -> None:
    with pytest.raises(TypeError, match="RPC class module not allowed"):
        decode(
            {
                TYPE_KEY: "dataclass",
                "class": "os:stat_result",
                "value": {},
            }
        )

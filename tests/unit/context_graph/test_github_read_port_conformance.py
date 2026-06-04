"""Conformance: every ``GitHubReadPort`` implementation must implement it fully.

Regression guard for the drift that shipped ``CodeProviderSourceControl`` —
the adapter production wiring actually injects (``wiring.source_for_repo``) —
missing ``list_pull_requests`` / ``list_issues``, the two methods the
reconciliation agent's GitHub backfill tools call. ``GitHubReadPort`` is a
``typing.Protocol`` (structural, runtime-unchecked), so a partial
implementation type-checks and passes everywhere it is *used*, then blows up
with ``AttributeError`` the moment the missing method is finally *called* —
which only happened in production, never in a test.

Parametrizing over every implementation pins them to the port: add a method
to the port and any class that forgets it fails here, in CI, instead of in
the agent loop.
"""

from __future__ import annotations

import inspect

import pytest

from adapters.outbound.connectors.github.api_client import (
    GitHubReadPort,
    PyGithubSourceControl,
)
from app.modules.context_graph.code_provider_source_control import (
    CodeProviderSourceControl,
)

# Every concrete class that is passed where a ``GitHubReadPort`` is expected.
# ``CodeProviderSourceControl`` is the one production wiring injects; the
# others must satisfy the same contract so they stay drop-in substitutable.
IMPLEMENTATIONS = [PyGithubSourceControl, CodeProviderSourceControl]


def _port_methods() -> set[str]:
    """Public method stubs declared directly on the protocol (the 8 endpoints)."""
    return {
        name
        for name, member in vars(GitHubReadPort).items()
        if not name.startswith("_") and inspect.isfunction(member)
    }


@pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=lambda c: c.__name__)
def test_implements_full_github_read_port(impl: type) -> None:
    required = _port_methods()
    assert required, "GitHubReadPort declares no methods — extraction is broken"
    missing = {name for name in required if not callable(getattr(impl, name, None))}
    assert not missing, (
        f"{impl.__name__} does not satisfy GitHubReadPort: missing {sorted(missing)}"
    )

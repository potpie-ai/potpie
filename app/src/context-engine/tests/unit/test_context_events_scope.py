"""EventScope helpers for standalone pots."""

from domain.context_events import (
    STANDALONE_POT_HOST,
    STANDALONE_POT_PROVIDER,
    STANDALONE_POT_REPO,
    event_scope_from_resolved_pot,
)
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo


def test_event_scope_from_resolved_pot_with_repo() -> None:
    repo = ResolvedPotRepo(
        pot_id="p1",
        repo_id="r1",
        provider="github",
        provider_host="github.com",
        repo_name="o/r",
    )
    r = ResolvedPot(pot_id="p1", name="n", repos=[repo])
    s = event_scope_from_resolved_pot("p1", r)
    assert s.provider == "github"
    assert s.repo_name == "o/r"


def test_event_scope_standalone_no_repo() -> None:
    r = ResolvedPot(pot_id="solo", name="solo", repos=[])
    s = event_scope_from_resolved_pot("solo", r)
    assert s.provider == STANDALONE_POT_PROVIDER
    assert s.provider_host == STANDALONE_POT_HOST
    assert s.repo_name == STANDALONE_POT_REPO

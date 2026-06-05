"""S6 local-profile completion: record→predicate mapping + ingest scan.

These exercise the additive local-profile work: that a recorded preference lands
on its ontology predicate (and so surfaces in the matching reader), and that
``HostShell.ingest`` runs working-tree scanners and writes resolvable claims.
"""

from __future__ import annotations

import pytest

from adapters.outbound.graph.backends.claim_query_analytics import ClaimQueryAnalytics
from adapters.outbound.graph.backends.in_memory_backend import InMemoryGraphBackend
from bootstrap.host_wiring import build_host_shell
from domain.ports.agent_context import RecordRequest, ResolveRequest


@pytest.fixture()
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    return build_host_shell(backend=InMemoryGraphBackend())


def test_recorded_preference_surfaces_in_coding_preferences(host):
    pot = host.pots.create_pot(name="default", use=True)
    receipt = host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="preference",
            summary="Always use ruff for linting",
            scope={"language": "python", "service": "context-engine"},
        )
    )
    assert receipt.accepted

    env = host.agent_context.resolve(
        ResolveRequest(
            pot_id=pot.pot_id,
            intent="feature",
            include=("coding_preferences",),
            scope={"language": "python"},
        )
    )
    # The record landed as a POLICY_APPLIES_TO claim, not a generic RELATES_TO,
    # so the coding_preferences reader serves it.
    prefs = [i for i in env.items if i.include == "coding_preferences"]
    assert prefs, "recorded preference did not surface in coding_preferences"
    assert "ruff" in dict(prefs[0].payload).get("fact", "")


def test_free_form_record_falls_back_to_related_to(host):
    pot = host.pots.create_pot(name="default", use=True)
    # ``workflow`` has no emits_predicate → RELATED_TO fallback; still recorded.
    receipt = host.agent_context.record(
        RecordRequest(
            pot_id=pot.pot_id,
            record_type="workflow",
            summary="Run make deploy to ship",
        )
    )
    assert receipt.accepted and receipt.mutations_applied >= 1
    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",))
    )
    assert any("make deploy" in dict(i.payload).get("fact", "") for i in env.items)


def test_ingest_scan_writes_resolvable_claims(host, tmp_path):
    pot = host.pots.create_pot(name="default", use=True)
    tree = tmp_path / "repo"
    tree.mkdir()
    (tree / "CODEOWNERS").write_text("* @team-core\n/api/ @team-api\n")
    (tree / "package.json").write_text(
        '{"name": "svc", "dependencies": {"express": "^4"}}'
    )

    result = host.ingest.scan_path(
        pot_id=pot.pot_id, root=str(tree), run_id="run1", repo_name="o/r"
    )
    assert "codeowners" in result.scanners_run
    assert result.edges_upserted > 0

    env = host.agent_context.resolve(
        ResolveRequest(pot_id=pot.pot_id, include=("raw_graph",))
    )
    assert len(env.items) == result.edges_upserted


def test_ingest_scan_missing_root_raises(host):
    pot = host.pots.create_pot(name="default", use=True)
    with pytest.raises(ValueError, match="does not exist"):
        host.ingest.scan_path(pot_id=pot.pot_id, root="/no/such/path", run_id="r")


def test_claim_query_analytics_counts_match_backend():
    backend = InMemoryGraphBackend()
    analytics = ClaimQueryAnalytics(backend.claim_query)
    # Empty pot: zero counts, empty quality.
    assert analytics.counts("p1")["claims"] == 0
    assert analytics.quality("p1")["status"] == "empty"

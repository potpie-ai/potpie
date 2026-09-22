"""Real private FalkorDB checks for environment-scoped semantic corrections."""

from pathlib import Path

import pytest

from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.workbench_service import GraphWorkbenchService
from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (
    FalkorDBGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)

pytestmark = pytest.mark.integration
client = pytest.importorskip("redislite.falkordb_client")


class Settings:
    def is_enabled(self):
        return True

    def falkordb_graph_name(self):
        return "correction_tests"


@pytest.mark.parametrize(
    "operation", ["end_relation_validity", "retract_claim", "supersede_claim"]
)
def test_correction_changes_only_previewed_production_claim(tmp_path: Path, operation):
    db = client.FalkorDB(str(tmp_path / "graph.db"))
    try:
        graph = db.select_graph("correction_tests")
        backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
        service = GraphWorkbenchService(
            backend=backend, plan_store=LocalJsonGraphPlanStore(home=tmp_path / "plans")
        )
        base = {
            "op": "link_entities",
            "subgraph": "infra_topology",
            "subject": {"key": "service:payments", "type": "Service"},
            "predicate": "DEPENDS_ON",
            "object": {"key": "service:ledger", "type": "Service"},
            "truth": "source_observation",
            "evidence": [{"source_ref": "fixture:dependency"}],
            "description": "Payments depends on ledger in this environment",
        }
        proposal = service.propose(
            {
                "operations": [
                    {**base, "environment": env} for env in ("prod", "staging")
                ]
            },
            pot_id="test",
        )
        assert service.commit(proposal.plan_id, pot_id="test").ok
        query = ClaimQueryFilter(
            pot_id="test", predicate_in=("DEPENDS_ON",), include_invalidated=True
        )
        before = backend.claim_query.find_claims(query)
        assert len(before) == 2
        prod = next(row for row in before if row.environment == "prod")
        staging = next(row for row in before if row.environment == "staging")
        correction = {
            **base,
            "op": operation,
            "environment": "prod",
            "reason": "Production ledger replaced",
        }
        if operation == "supersede_claim":
            correction["superseded_by"] = {
                "key": "service:ledger-v2",
                "type": "Service",
            }
        preview = service.propose(
            {"operations": [correction]}, pot_id="test", approved_by="test:reviewer"
        )
        assert preview.ok
        assert preview.diff.retracted_claim_keys == (prod.claim_key,)
        receipt = service.commit(preview.plan_id, pot_id="test", verify=True)
        assert receipt.ok
        after = {row.claim_key: row for row in backend.claim_query.find_claims(query)}
        assert after[prod.claim_key].invalid_at is not None
        assert after[staging.claim_key] == staging
    finally:
        db.close()

"""Check filter translation at both native graph driver boundaries."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.graph.falkordb_reader import (
    FalkorDBClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_reader import (
    Neo4jClaimQueryStore,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter


class _Driver:
    def __init__(self):
        self.calls = []

    def query(self, cypher, params):
        self.calls.append(params)
        return SimpleNamespace(header=[], result_set=[])

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, cypher, **params):
        self.calls.append(params)
        return []


@pytest.mark.parametrize("backend", ["neo4j", "falkordb"])
@pytest.mark.parametrize("include_invalidated", [False, True])
@pytest.mark.parametrize("include_retired", [False, True])
def test_native_readers_preserve_scopes_and_backend_filtering(
    backend, include_invalidated, include_retired
):
    driver = _Driver()
    if backend == "neo4j":
        store = Neo4jClaimQueryStore(settings=object(), driver=driver)
    else:
        store = FalkorDBClaimQueryStore(settings=object(), graph=driver)
    point = datetime(2026, 3, 1, tzinfo=timezone.utc)
    filter_ = ClaimQueryFilter(
        pot_id="p1",
        predicate_in=("USES",),
        subject_key_in=("service:web",),
        object_key_in=("service:db",),
        claim_key_in=("claim:1",),
        record_id_in=("record:1",),
        subgraph_in=("infra_topology",),
        subgraph_not_in=("resources",),
        mutation_id_in=("mutation:1",),
        source_ref_in=("repo:config",),
        source_system_in=("github",),
        exact_text_in=("Web",),
        exact_text_pattern=".*web.*",
        environment_in=("PROD",),
        truth_in=("SOURCE_OBSERVATION",),
        subject_label="Service",
        object_label="Database",
        endpoint_label="Entity",
        as_of=point,
        valid_at_after=point,
        valid_at_before=point,
        include_invalidated=include_invalidated,
        include_retired=include_retired,
    )

    assert store.find_claims(filter_) == []
    assert driver.calls == [
        {
            "gid": "p1",
            "preds": ["USES"],
            "subjects": ["service:web"],
            "objects": ["service:db"],
            "claim_keys": ["claim:1"],
            "record_ids": ["record:1"],
            "subgraphs": ["infra_topology"],
            "excluded_subgraphs": ["resources"],
            "mutation_ids": ["mutation:1"],
            "source_refs": ["repo:config"],
            "sources": ["github"],
            "exact_text": ["web"] if backend == "neo4j" else None,
            "exact_pattern": ".*web.*",
            "environments": ["prod"],
            "truths": ["source_observation"],
            "subject_label": "Service",
            "object_label": "Database",
            "endpoint_label": "Entity",
            "as_of": point.isoformat(),
            "query_time": point.isoformat(),
            "va_after": point.isoformat() if backend == "neo4j" else None,
            "va_before": point.isoformat() if backend == "neo4j" else None,
            "include_invalid": include_invalidated if backend == "neo4j" else True,
            "include_retired": include_retired,
        }
    ]

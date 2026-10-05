"""Measure native journal deltas on fresh private FalkorDB graphs, then delete them.

Run with the source checkouts on PYTHONPATH and --port pointing at a disposable
FalkorDB server. No project pot or graph name is accepted.
"""

import argparse
import json
import statistics
import time
import uuid

from falkordb import FalkorDB
from potpie_context_core.graph_journal import journal_json
from potpie_context_core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_core.reconciliation import MutationBatch

from potpie_context_engine.adapters.outbound.graph.backends.falkordb_backend import (
    FalkorDBGraphBackend,
)


class Settings:
    def is_enabled(self):
        return True


def measure(backend, batch, commit_id):
    started = time.perf_counter()
    result = backend.mutation.compare_and_apply(
        batch,
        expected_pot_id="benchmark",
        expected_version=backend.mutation.current_version("benchmark"),
        provenance_context=ProvenanceContext(mutation_id=commit_id),
    )
    elapsed = (time.perf_counter() - started) * 1000
    assert result.ok
    receipt = backend.journal.get_receipt(pot_id="benchmark", commit_id=commit_id)
    return {
        "latency_ms": round(elapsed, 2),
        "receipt_bytes": len(journal_json(receipt).encode()),
        "affected_records": receipt.affected_record_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bulk", type=int, default=100)
    parser.add_argument("--supersession", type=int, default=500)
    args = parser.parse_args()
    graph = FalkorDB(host="127.0.0.1", port=args.port).select_graph(
        "journal_benchmark_" + uuid.uuid4().hex
    )
    backend = FalkorDBGraphBackend(Settings(), graph_provider=lambda: graph)
    try:
        backend.journal.activate(pot_id="benchmark")
        measure(
            backend,
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(
                        "service:a", ("Entity", "Service"), {"summary": "seed"}
                    )
                ]
            ),
            "seed",
        )
        patches = [
            measure(
                backend,
                MutationBatch(
                    entity_upserts=[
                        EntityUpsert(
                            "service:a", ("Entity", "Service"), {"summary": str(i)}
                        )
                    ]
                ),
                f"patch:{i}",
            )
            for i in range(7)
        ]
        bulk = measure(
            backend,
            MutationBatch(
                entity_upserts=[
                    EntityUpsert(f"service:bulk{i}", ("Entity", "Service"))
                    for i in range(args.bulk)
                ],
                edge_upserts=[
                    EdgeUpsert(
                        "DEPENDS_ON",
                        "service:a",
                        f"service:bulk{i}",
                        {
                            "claim_key": f"bulk:{i}",
                            "source_ref": "benchmark",
                            "truth": "authoritative_fact",
                        },
                    )
                    for i in range(args.bulk)
                ],
            ),
            "bulk",
        )
        measure(
            backend,
            MutationBatch(
                entity_upserts=[
                    EntityUpsert("service:owner", ("Entity", "Service")),
                    EntityUpsert("team:old", ("Entity", "Team")),
                    EntityUpsert("team:new", ("Entity", "Team")),
                ],
                edge_upserts=[
                    EdgeUpsert(
                        "OWNED_BY",
                        "service:owner",
                        "team:old",
                        {
                            "claim_key": f"old:{i}",
                            "source_ref": f"source:{i}",
                            "truth": "authoritative_fact",
                        },
                    )
                    for i in range(args.supersession)
                ],
            ),
            "owners",
        )
        supersession = measure(
            backend,
            MutationBatch(
                edge_upserts=[
                    EdgeUpsert(
                        "OWNED_BY",
                        "service:owner",
                        "team:new",
                        {
                            "claim_key": "new-owner",
                            "source_ref": "new-source",
                            "truth": "authoritative_fact",
                        },
                    )
                ]
            ),
            "supersession",
        )
        assert supersession["affected_records"] == args.supersession + 1
        print(
            json.dumps(
                {
                    "small_patch": {
                        "median_latency_ms": round(
                            statistics.median(p["latency_ms"] for p in patches), 2
                        ),
                        "max_receipt_bytes": max(p["receipt_bytes"] for p in patches),
                        "affected_records": 1,
                    },
                    "bulk_relations": bulk,
                    "implicit_supersession": supersession,
                },
                indent=2,
            )
        )
    finally:
        graph.delete()


if __name__ == "__main__":
    main()

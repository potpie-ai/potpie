"""Public frozen repair resumes across workflow loss and a real server restart."""

import os
import subprocess
import time

import pytest

from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.runtime import build_graph_runtime
from tests.graph_reliability import isolated_backend


class LostFinalizationStore(LocalJsonGraphPlanStore):
    def compare_and_set(self, *, expected, replacement):
        if replacement.status == "committed":
            raise OSError("injected lost workflow finalization")
        return super().compare_and_set(expected=expected, replacement=replacement)


async def _authorize(pot_id, access):
    assert access in {"read", "admin"}


def _workbench(backend, store):
    return build_graph_runtime(
        backend=backend,
        plan_store=store,
        commit_authorize=_authorize,
        commit_actor=lambda: "acceptance-admin",
    ).workbench


def _seed_raw_entities(profile, backend, pot):
    rows = [
        {
            "group_id": pot,
            "entity_key": f"service:legacy:{index}",
            "observed_at": "2020-01-01T05:30:00+05:30",
        }
        for index in range(101)
    ]
    query = "UNWIND $rows AS props CREATE (e:Entity:Service) SET e=props"
    if profile == "falkordb":
        backend.graph_provider().query(query, params={"rows": rows})
    else:
        from neo4j import GraphDatabase

        with (
            GraphDatabase.driver(
                os.environ["PROTOCOL_TEST_NEO4J_URI"],
                auth=(
                    os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
                    os.environ["PROTOCOL_TEST_NEO4J_PASSWORD"],
                ),
            ) as driver,
            driver.session() as session,
        ):
            session.run(query, rows=rows).consume()


@pytest.mark.parametrize(
    "profile,container_env",
    [
        ("falkordb", "POTPIE_TEST_FALKORDB_RESTART_CONTAINER"),
        ("neo4j", "PROTOCOL_TEST_NEO4J_CONTAINER"),
    ],
)
def test_public_repair_restart_retains_frozen_children_and_receipt(
    profile, container_env, tmp_path, monkeypatch
):
    container = os.environ.get(container_env)
    if not container:
        pytest.skip(f"{container_env} must identify the isolated disposable service")
    with isolated_backend(profile, tmp_path / "graph", monkeypatch) as (pot, backend):
        _seed_raw_entities(profile, backend, pot)
        home = tmp_path / "plans"
        store = LostFinalizationStore(home)
        workbench = _workbench(backend, store)
        preview = workbench.repair_legacy(pot_id=pot, mode="dry_run")
        assert preview.ok and preview.total_batches == 2
        frozen = store.get_repair_run(pot_id=pot, repair_id=preview.repair_id)
        with pytest.raises(OSError, match="lost workflow finalization"):
            workbench.repair_legacy(
                pot_id=pot, mode="apply", repair_id=preview.repair_id
            )
        child = store.get(pot_id=pot, plan_id=frozen.children[0].plan_id)
        assert (
            child.status == "committing" and backend.mutation.current_version(pot) == 1
        )
        receipt = backend.mutation.lookup_execution(
            child.lowered_batch, expected_pot_id=pot, mutation_id=child.mutation_id
        ).result
        subprocess.run(
            ["docker", "restart", container],
            check=True,
            capture_output=True,
            timeout=60,
        )
        deadline = time.monotonic() + 60
        while True:
            try:
                assert backend.mutation.current_version(pot) == 1
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        reopened = LocalJsonGraphPlanStore(home)
        runtime = _workbench(backend, reopened)
        result = runtime.repair_legacy(
            pot_id=pot, mode="apply", repair_id=preview.repair_id
        )
        assert result.ok and result.completed_batches == 2
        assert (
            reopened.get_repair_run(pot_id=pot, repair_id=preview.repair_id) == frozen
        )
        assert result.executions[0] == {
            "mutation_id": child.mutation_id,
            **receipt.execution.to_dict(),
        }
        assert backend.mutation.current_version(pot) == 2
        replay = runtime.repair_legacy(
            pot_id=pot, mode="apply", repair_id=preview.repair_id
        )
        assert (
            replay.executions == result.executions
            and backend.mutation.current_version(pot) == 2
        )
        second = runtime.repair_legacy(pot_id=pot, mode="dry_run")
        assert second.repairable_records == 0

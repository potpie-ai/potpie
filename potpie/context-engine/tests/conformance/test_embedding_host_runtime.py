"""An embedding host composes a graph runtime from the supported API alone.

The host modelled here is a hosted service: it brings async-only plan and inbox
stores, its own observer, and its own commit-history policy (actor,
authorization, host label, preview store and listing index), and wires all of
it at construction. Every engine import is ``potpie_context_engine.api``; the
in-memory backend from ``potpie_context_engine.testing`` stands in for the
host's own graph backend.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from potpie_context_engine.api import (
    DEFAULT_GRAPH_DEFINITION,
    DEFAULT_MUTATION_POLICY,
    EntityUpsert,
    GraphCatalogRequest,
    GraphInboxItem,
    GraphMutationPlanRecord,
    GraphRuntime,
    MutationBatch,
    ProvenanceContext,
    ReconciliationConfig,
    RuntimeCompositionError,
    build_graph_runtime,
)
from potpie_context_engine.testing import InMemoryGraphBackend

POT = "pot:embedding-host"


@dataclass
class _AsyncPlanStore:
    """A host-owned plan store that, like a database pool, is async only."""

    records: dict[tuple[str, str], GraphMutationPlanRecord] = field(
        default_factory=dict
    )

    async def save_async(self, record: GraphMutationPlanRecord) -> None:
        self.records[(record.pot_id, record.plan_id)] = record

    async def get_async(
        self, *, pot_id: str, plan_id: str
    ) -> GraphMutationPlanRecord | None:
        return self.records.get((pot_id, plan_id))

    async def compare_and_set_async(
        self,
        *,
        expected: GraphMutationPlanRecord,
        replacement: GraphMutationPlanRecord,
    ) -> bool:
        key = (expected.pot_id, expected.plan_id)
        if self.records.get(key) != expected:
            return False
        self.records[key] = replacement
        return True

    async def list_async(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]:
        rows = sorted(
            (
                record
                for (record_pot, _), record in self.records.items()
                if record_pot == pot_id
                and plan_id in (None, record.plan_id)
                and mutation_id in (None, record.mutation_id)
                and (since is None or record.created_at >= since)
                and (until is None or record.created_at <= until)
            ),
            key=lambda record: record.created_at,
            reverse=True,
        )
        return tuple(rows[:limit] if limit is not None else rows)


@dataclass
class _AsyncInboxStore:
    items: dict[tuple[str, str], GraphInboxItem] = field(default_factory=dict)

    async def save_async(self, item: GraphInboxItem) -> None:
        self.items[(item.pot_id, item.item_id)] = item

    async def get_async(self, *, pot_id: str, item_id: str) -> GraphInboxItem | None:
        return self.items.get((pot_id, item_id))

    async def compare_and_set_async(
        self, *, expected: GraphInboxItem, replacement: GraphInboxItem
    ) -> bool:
        key = (expected.pot_id, expected.item_id)
        if self.items.get(key) != expected:
            return False
        self.items[key] = replacement
        return True

    async def list_async(
        self,
        *,
        pot_id: str,
        status: tuple[str, ...] = (),
        claimed_by: str | None = None,
        suspected_subgraph: str | None = None,
        source_ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphInboxItem, ...]:
        rows = sorted(
            (
                item
                for (item_pot, _), item in self.items.items()
                if item_pot == pot_id
                and (not status or item.status in status)
                and claimed_by in (None, item.claimed_by)
                and (
                    suspected_subgraph is None
                    or suspected_subgraph in item.suspected_subgraphs
                )
                and (source_ref is None or source_ref in item.source_refs)
                and (since is None or item.created_at >= since)
                and (until is None or item.created_at <= until)
            ),
            key=lambda item: item.created_at,
            reverse=True,
        )
        return tuple(rows[:limit] if limit is not None else rows)


@dataclass
class _RecordingObserver:
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def observe(self, event: str, fields: Any) -> None:
        self.events.append((event, dict(fields)))


@dataclass
class _HostPreviewStore:
    previews: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    async def put_async(self, preview: Any, request: Any) -> None:
        self.previews[preview.preview_id] = (preview, request)

    async def get_async(self, *, preview_id: str, pot_id: str):
        hit = self.previews.get(preview_id)
        return hit if hit is not None and hit[0].pot_id == pot_id else None


class _ListingIndexReached(Exception):
    """Raised by the host's commit listing index to prove it was consulted."""


class _HostListingIndex:
    async def progress_async(self, *, pot_id: str, generation: str) -> int:
        raise _ListingIndexReached(pot_id)


@dataclass
class _HostPolicy:
    """Grant-backed authorization: the host, not the engine, decides access."""

    grants: dict[str, set[str]]
    checks: list[tuple[str, str]] = field(default_factory=list)

    async def authorize(self, pot_id: str, access: str) -> None:
        self.checks.append((pot_id, access))
        if access not in self.grants.get(pot_id, set()):
            raise PermissionError(f"{access} denied on {pot_id}")


def _compose(
    *,
    policy: _HostPolicy,
    previews: _HostPreviewStore | None = None,
    listing_index: Any = None,
    observer: _RecordingObserver | None = None,
    plans: _AsyncPlanStore | None = None,
    inbox: _AsyncInboxStore | None = None,
) -> GraphRuntime:
    return build_graph_runtime(
        InMemoryGraphBackend(),
        plans or _AsyncPlanStore(),
        inbox or _AsyncInboxStore(),
        definition=DEFAULT_GRAPH_DEFINITION,
        policy=DEFAULT_MUTATION_POLICY,
        observability=observer or _RecordingObserver(),
        reconciliation_config=ReconciliationConfig(infer_canonical_labels=False),
        commit_mirror=listing_index,
        preview_store=previews,
        commit_host="embedded:test-graph",
        commit_actor=lambda: "service-user:alice",
        commit_authorize=policy.authorize,
    )


def test_embedding_host_runs_a_write_and_read_journey_on_its_own_stores() -> None:
    async def journey() -> None:
        observer = _RecordingObserver()
        plans, inbox_store = _AsyncPlanStore(), _AsyncInboxStore()
        policy = _HostPolicy(grants={POT: {"read", "write", "admin"}})
        runtime = _compose(
            policy=policy, observer=observer, plans=plans, inbox=inbox_store
        )

        proposal = await runtime.propose_async(
            {
                "operations": [
                    {
                        "op": "assert_claim",
                        "subject": {"key": "service:billing", "type": "Service"},
                        "predicate": "DEPENDS_ON",
                        "object": {"key": "service:ledger", "type": "Service"},
                        "truth": "agent_claim",
                        "description": "Billing calls the ledger to post charges.",
                    }
                ]
            },
            pot_id=POT,
        )
        commit = await runtime.commit_async(proposal.plan_id, pot_id=POT, verify=True)
        receipt = await runtime.commit_status_async(proposal.plan_id, pot_id=POT)
        inbox = await runtime.inbox_add_async(
            pot_id=POT, summary="Confirm the ledger owner."
        )
        listed = await runtime.inbox_list_async(pot_id=POT)
        catalog = await runtime.catalog_async(GraphCatalogRequest(pot_id=POT))
        status = await runtime.status_async(POT)

        assert proposal.ok and commit.ok, commit.to_dict()
        assert commit.verification is not None and commit.verification.ok
        assert receipt.ok and receipt.mutation_id == commit.mutation_id
        assert plans.records[(POT, proposal.plan_id)].mutation_id == (
            commit.mutation_id
        )
        assert [item.item_id for item in listed.items] == [inbox.item.item_id]
        assert (POT, inbox.item.item_id) in inbox_store.items
        assert catalog.views
        assert status["pot_id"] == POT
        assert {event for event, _ in observer.events} >= {
            "graph.propose",
            "graph.commit",
        }

    asyncio.run(journey())


def test_commit_history_policy_is_wired_at_construction() -> None:
    async def journey() -> None:
        previews = _HostPreviewStore()
        policy = _HostPolicy(grants={POT: {"read", "write"}})
        runtime = _compose(
            policy=policy,
            previews=previews,
            listing_index=_HostListingIndex(),
        )
        runtime.backend.journal.activate(pot_id=POT, rollback_enabled=True)
        for commit_id, summary in (("c1", "before"), ("c2", "after")):
            runtime.backend.mutation.apply(
                MutationBatch(
                    entity_upserts=[
                        EntityUpsert(
                            "service:billing",
                            ("Entity", "Service"),
                            {"summary": summary},
                        )
                    ]
                ),
                expected_pot_id=POT,
                provenance_context=ProvenanceContext(mutation_id=commit_id),
            )

        status = await runtime.journal_status_async(pot_id=POT)
        preview = await runtime.revert_preview_async(
            "c2", pot_id=POT, expected_head="c2"
        )
        with pytest.raises(_ListingIndexReached):
            await runtime.commits_async(pot_id=POT)
        with pytest.raises(PermissionError):
            await runtime.rebuild_commits_async(pot_id=POT)
        with pytest.raises(PermissionError):
            await runtime.journal_status_async(pot_id="pot:not-granted")

        assert status["ok"] is True
        assert preview["ok"] is True, preview
        stored, _request = previews.previews[preview["preview"].preview_id]
        assert stored.actor == "service-user:alice"
        assert stored.host == "embedded:test-graph"
        assert (POT, "admin") in policy.checks
        assert ("pot:not-granted", "read") in policy.checks

    asyncio.run(journey())


def test_runtime_without_history_wiring_reports_it_unavailable() -> None:
    async def journey() -> None:
        runtime = _compose(policy=_HostPolicy(grants={POT: {"read", "write"}}))

        listing = await runtime.commits_async(pot_id=POT)
        preview = await runtime.revert_preview_async(
            "c1", pot_id=POT, expected_head="c1"
        )

        assert listing["ok"] is False and listing["status"] == "unavailable"
        assert preview["ok"] is False and preview["status"] == "unavailable"

    asyncio.run(journey())


def test_builder_takes_wiring_as_keyword_only_arguments() -> None:
    parameters = inspect.signature(build_graph_runtime).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

    assert positional == ["backend", "plan_store", "inbox_store", "definition"]
    assert {
        "policy",
        "observability",
        "reconciliation_config",
        "resource_index",
        "resource_store",
        "commit_mirror",
        "preview_store",
        "commit_host",
        "commit_actor",
        "commit_authorize",
    } <= {
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    with pytest.raises(TypeError):
        build_graph_runtime(  # type: ignore[misc]
            InMemoryGraphBackend(),
            _AsyncPlanStore(),
            _AsyncInboxStore(),
            DEFAULT_GRAPH_DEFINITION,
            DEFAULT_MUTATION_POLICY,
        )


@pytest.mark.parametrize("name", ["commit_actor", "commit_authorize"])
def test_builder_rejects_non_callable_commit_policy(name: str) -> None:
    with pytest.raises(RuntimeCompositionError, match=name):
        build_graph_runtime(
            InMemoryGraphBackend(),
            _AsyncPlanStore(),
            **{name: "not-callable"},
        )


_COMPOSE_SNIPPET = """
import json
import sys

from potpie_context_engine.api import GraphCatalogRequest, build_graph_runtime
from potpie_context_engine.testing import (
    InMemoryGraphBackend,
    InMemoryGraphInboxStore,
    InMemoryGraphPlanStore,
)

runtime = build_graph_runtime(
    InMemoryGraphBackend(), InMemoryGraphPlanStore(), InMemoryGraphInboxStore()
)
assert runtime.catalog(GraphCatalogRequest(pot_id="pot:isolated")).views
print(json.dumps(sorted(sys.modules)))
"""


def test_composition_needs_no_daemon_cli_or_root_product_modules() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _COMPOSE_SNIPPET],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    top_level = {module.split(".")[0] for module in loaded}

    assert "potpie" not in top_level
    assert not top_level & {"fastapi", "typer", "uvicorn", "rich", "keyring"}

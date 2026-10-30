"""Authenticate before selecting, per-actor selection and scoped commit access.

A managed host resolves the same selector differently for different actors,
and an unauthenticated caller must not learn whether a pot exists. Commit
history and rollback follow the role the request was authorized under, and
caller-asserted identities in requests are checked against the credential.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from potpie.pots.contracts import PotInfo, PotRepoSource
from potpie.runtime import (
    PROTOCOL_VERSION,
    AcquisitionRequest,
    AuthenticatedActor,
    AuthenticationError,
    AuthorizationScope,
    CompositionFingerprint,
    ContextResourceManager,
    ContextSelector,
    EngineOperation,
    EngineOperationRequest,
    ResourceComposition,
)
from potpie.runtime.commit_access import (
    authorization_scope,
    authorize_scoped_commit,
    commit_access_for,
    current_commit_actor,
)
from potpie.runtime.request_policy import RemoteRequestPolicy
from potpie.runtime.resource_manager import PassThroughAuthenticator
from potpie.runtime.selection import CatalogSelectorResolver
from potpie_context_engine import (
    ContextEngine,
    ContextIdentity,
    EngineConfig,
    EngineDependencies,
    Failure,
    Success,
)
from potpie_context_engine.core.commit_service import CommitAccessDenied
from potpie_context_engine.requests import (
    CommitRequest,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
    InboxAddRequest,
    InboxClaimRequest,
    InboxCloseRequest,
    InboxListRequest,
    InboxMarkAppliedRequest,
    InboxMarkRejectedRequest,
    ProposeRequest,
)

pytestmark = pytest.mark.unit

ALICE = AuthenticatedActor(actor_id="alice")
BOB = AuthenticatedActor(actor_id="bob")
CAROL = AuthenticatedActor(actor_id="carol")


# --- a multi-actor catalog -----------------------------------------------------


@dataclass
class _Catalog:
    pots: dict[str, PotInfo]
    grants: dict[str, set[str]]
    active: dict[str, str] = field(default_factory=dict)
    defaults: dict[tuple[str, str], str] = field(default_factory=dict)
    sources: list[PotRepoSource] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def _visible(self, actor: AuthenticatedActor) -> list[PotInfo]:
        allowed = self.grants.get(actor.actor_id, set())
        return [pot for pot_id, pot in self.pots.items() if pot_id in allowed]

    async def pots_for(self, actor):
        self.calls.append(("pots_for", actor.actor_id))
        return self._visible(actor)

    async def active_for(self, actor):
        pot_id = self.active.get(actor.actor_id)
        return self.pots.get(pot_id) if pot_id else None

    async def repo_default_for(self, actor, repo):
        return self.defaults.get((actor.actor_id, repo))

    async def repo_sources_for(self, actor):
        visible = {pot.pot_id for pot in self._visible(actor)}
        return [row for row in self.sources if row.pot_id in visible]


def _catalog() -> _Catalog:
    return _Catalog(
        pots={
            "pot_a": PotInfo(pot_id="pot_a", name="shop"),
            "pot_b": PotInfo(pot_id="pot_b", name="shop-fork"),
            "pot_old": PotInfo(pot_id="pot_old", name="legacy", archived=True),
        },
        grants={"alice": {"pot_a", "pot_b", "pot_old"}, "bob": {"pot_b"}},
        active={"alice": "pot_a"},
        sources=[
            PotRepoSource(
                pot_id="pot_b",
                pot_name="shop-fork",
                name="github.com/acme/shop",
                location="https://github.com/acme/shop.git",
            )
        ],
    )


async def _resolve(catalog, selector, actor):
    return await CatalogSelectorResolver(catalog).resolve(selector, actor=actor)


@pytest.mark.anyio
async def test_an_explicit_ref_resolves_only_among_the_actors_pots() -> None:
    catalog = _catalog()

    alice = await _resolve(catalog, ContextSelector("explicit", "shop"), ALICE)
    carol_by_name = await _resolve(catalog, ContextSelector("explicit", "shop"), CAROL)
    carol_by_id = await _resolve(catalog, ContextSelector("explicit", "pot_a"), CAROL)
    nobody = await _resolve(catalog, ContextSelector("explicit", "nope"), CAROL)

    assert alice == Success(ContextIdentity("pot_a"))
    # Without a grant, a real pot and a missing one read the same.
    for refused in (carol_by_name, carol_by_id, nobody):
        assert isinstance(refused, Failure)
        assert refused.error.code == "pot_not_found"
    assert carol_by_id.error.message == "No pot matching 'pot_a'."


@pytest.mark.anyio
async def test_archived_pots_resolve_only_by_exact_id() -> None:
    catalog = _catalog()

    by_id = await _resolve(catalog, ContextSelector("explicit", "pot_old"), ALICE)
    by_name = await _resolve(catalog, ContextSelector("explicit", "legacy"), ALICE)

    assert by_id == Success(ContextIdentity("pot_old"))
    assert isinstance(by_name, Failure) and by_name.error.code == "pot_archived"


@pytest.mark.anyio
async def test_active_and_repository_selection_are_per_actor() -> None:
    catalog = _catalog()
    repo = ContextSelector("repository", "github.com/acme/shop")

    assert await _resolve(catalog, ContextSelector("active"), ALICE) == Success(
        ContextIdentity("pot_a")
    )
    bob_active = await _resolve(catalog, ContextSelector("active"), BOB)
    assert isinstance(bob_active, Failure)
    assert bob_active.error.code == "no_active_pot"
    # The registered repo source lives in pot_b, which both may reach.
    assert await _resolve(catalog, repo, BOB) == Success(ContextIdentity("pot_b"))
    # Alice's own default for the repo wins over the source match.
    catalog.defaults[("alice", "github.com/acme/shop")] = "pot_a"
    assert await _resolve(catalog, repo, ALICE) == Success(ContextIdentity("pot_a"))
    # Carol sees no source and has no active pot.
    carol = await _resolve(catalog, repo, CAROL)
    assert isinstance(carol, Failure) and carol.error.code == "no_active_pot"


@pytest.mark.anyio
async def test_a_failing_catalog_is_a_retryable_selection_failure() -> None:
    class _Broken(_Catalog):
        async def pots_for(self, actor):
            raise OSError("store down")

    outcome = await _resolve(
        _Broken(pots={}, grants={}), ContextSelector("active"), ALICE
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "context_selection_unavailable"
    assert outcome.error.retry_posture == "safe"


# --- the Resource Manager authenticates before it selects ----------------------


class _Authorizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def authorize(self, actor, operation, context):
        self.calls.append((actor.actor_id, operation, context.value))
        return Success(
            AuthorizationScope(
                actor_id=actor.actor_id,
                operation=operation,
                context=context,
                attributes={"commit_access": "reader"},
            )
        )


class _Engine:
    def __init__(self, context: ContextIdentity, closed: list[str]) -> None:
        self.context = context
        self._closed = closed

    async def close(self):
        self._closed.append(self.context.value)
        return Success(None)


class _Composer:
    def __init__(self) -> None:
        self.composed: list[str] = []

    async def fingerprint(self, context):
        return Success(CompositionFingerprint(f"fp:{context.value}"))

    async def compose(self, context, fingerprint):
        self.composed.append(context.value)
        operations = cast(object, object())
        return Success(
            ResourceComposition(
                fingerprint=fingerprint,
                config=EngineConfig(),
                dependencies=EngineDependencies(
                    context=operations,  # type: ignore[arg-type]
                    graph=operations,  # type: ignore[arg-type]
                    workbench=operations,  # type: ignore[arg-type]
                    ingestion=operations,  # type: ignore[arg-type]
                    nudge=operations,  # type: ignore[arg-type]
                ),
            )
        )


def _manager(
    catalog: _Catalog, *, max_idle_engines: int | None = None
) -> tuple[ContextResourceManager, _Authorizer, _Composer, list[str]]:
    closed: list[str] = []
    authorizer = _Authorizer()
    composer = _Composer()

    async def factory(*, context, config, dependencies):
        del config, dependencies
        return Success(cast(ContextEngine, _Engine(context, closed)))

    manager = ContextResourceManager(
        resolver=CatalogSelectorResolver(catalog),
        authenticator=PassThroughAuthenticator(),
        authorizer=authorizer,
        composer=composer,
        engine_factory=factory,
        max_idle_engines=max_idle_engines,
    )
    return manager, authorizer, composer, closed


def _acquisition(
    selector: ContextSelector, authentication: object
) -> AcquisitionRequest:
    return AcquisitionRequest(
        request_id="request-1",
        selector=selector,
        operation="search",
        authentication=authentication,
    )


@pytest.mark.anyio
async def test_an_unauthenticated_caller_cannot_probe_pots() -> None:
    catalog = _catalog()
    manager, authorizer, composer, _ = _manager(catalog)

    outcome = await manager.acquire(
        _acquisition(ContextSelector("explicit", "pot_a"), {"bearer": "forged"})
    )

    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, AuthenticationError)
    assert catalog.calls == []
    assert authorizer.calls == [] and composer.composed == []


@pytest.mark.anyio
async def test_the_authenticated_actor_drives_selection_and_scope() -> None:
    catalog = _catalog()
    manager, authorizer, _, _ = _manager(catalog)

    bob = await manager.acquire(_acquisition(ContextSelector("explicit", "pot_b"), BOB))
    carol = await manager.acquire(
        _acquisition(ContextSelector("explicit", "pot_b"), CAROL)
    )

    assert isinstance(bob, Success)
    assert bob.value.scope.actor_id == "bob"
    assert isinstance(carol, Failure) and carol.error.code == "pot_not_found"
    assert catalog.calls == [("pots_for", "bob"), ("pots_for", "carol")]
    assert authorizer.calls == [("bob", "search", "pot_b")]
    await bob.value.release()
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_authorize_checks_a_request_without_composing_an_engine() -> None:
    manager, _, composer, _ = _manager(_catalog())

    outcome = await manager.authorize(
        _acquisition(ContextSelector("explicit", "shop"), ALICE)
    )

    assert isinstance(outcome, Success)
    assert outcome.value.actor == ALICE
    assert outcome.value.context == ContextIdentity("pot_a")
    assert outcome.value.scope.actor_id == "alice"
    assert composer.composed == []
    assert manager.cached_engine_count == 0


# --- idle engines are evicted least recently used first ------------------------


@pytest.mark.anyio
async def test_idle_engines_past_the_bound_are_closed_least_recently_used_first() -> (
    None
):
    catalog = _catalog()
    manager, _, composer, closed = _manager(catalog, max_idle_engines=1)

    first = await manager.acquire(
        _acquisition(ContextSelector("explicit", "pot_a"), ALICE)
    )
    second = await manager.acquire(
        _acquisition(ContextSelector("explicit", "pot_b"), ALICE)
    )
    assert isinstance(first, Success) and isinstance(second, Success)
    await first.value.release()
    assert closed == []  # one idle engine is within the bound
    await second.value.release()

    assert closed == ["pot_a"]
    assert manager.cached_engine_count == 1

    again = await manager.acquire(
        _acquisition(ContextSelector("explicit", "pot_a"), ALICE)
    )
    assert isinstance(again, Success)
    assert composer.composed == ["pot_a", "pot_b", "pot_a"]
    await again.value.release()
    assert closed == ["pot_a", "pot_b"]
    assert isinstance(await manager.shutdown(), Success)
    assert closed == ["pot_a", "pot_b", "pot_a"]


@pytest.mark.anyio
async def test_an_engine_with_a_lease_is_never_evicted() -> None:
    manager, _, _, closed = _manager(_catalog(), max_idle_engines=0)

    held = await manager.acquire(
        _acquisition(ContextSelector("explicit", "pot_a"), ALICE)
    )
    brief = await manager.acquire(
        _acquisition(ContextSelector("explicit", "pot_b"), ALICE)
    )
    assert isinstance(held, Success) and isinstance(brief, Success)
    await brief.value.release()

    assert closed == ["pot_b"]
    assert manager.cached_engine_count == 1
    await held.value.release()
    assert closed == ["pot_b", "pot_a"]
    assert isinstance(await manager.shutdown(), Success)


@pytest.mark.anyio
async def test_without_a_bound_engines_live_until_shutdown() -> None:
    manager, _, _, closed = _manager(_catalog())

    for pot in ("pot_a", "pot_b"):
        lease = await manager.acquire(
            _acquisition(ContextSelector("explicit", pot), ALICE)
        )
        assert isinstance(lease, Success)
        await lease.value.release()

    assert closed == [] and manager.cached_engine_count == 2
    assert isinstance(await manager.shutdown(), Success)
    assert sorted(closed) == ["pot_a", "pot_b"]


def test_a_negative_bound_is_refused() -> None:
    with pytest.raises(ValueError, match="max_idle_engines"):
        _manager(_catalog(), max_idle_engines=-1)


# --- commit access follows the authorized scope --------------------------------


def _scope(role: object, pot: str = "pot_a") -> AuthorizationScope:
    return AuthorizationScope(
        actor_id="alice",
        operation="commits",
        context=ContextIdentity(pot),
        attributes={"commit_access": role} if role is not None else {},
    )


@pytest.mark.parametrize(
    ("role", "levels"),
    [
        ("reader", {"read"}),
        ("writer", {"read", "write"}),
        ("admin", {"read", "write", "admin"}),
        (("read", "admin", "owner"), {"read", "admin"}),
        ("owner", set()),
        (None, set()),
    ],
)
def test_commit_access_comes_from_the_scope_role(role, levels) -> None:
    assert commit_access_for(_scope(role)) == levels


def test_no_scope_grants_no_commit_access() -> None:
    assert commit_access_for(None) == frozenset()


def _authorize(pot: str, access: str) -> None:
    asyncio.run(authorize_scoped_commit(pot, access))


def test_scoped_commit_authorization_checks_pot_and_level() -> None:
    with pytest.raises(CommitAccessDenied):
        _authorize("pot_a", "read")
    with authorization_scope(_scope("reader")):
        _authorize("pot_a", "read")
        with pytest.raises(CommitAccessDenied, match="write"):
            _authorize("pot_a", "write")
        with pytest.raises(CommitAccessDenied, match="admin"):
            _authorize("pot_a", "admin")
        with pytest.raises(CommitAccessDenied, match="pot_a"):
            _authorize("pot_b", "read")
        assert current_commit_actor() == "alice"
    with authorization_scope(_scope("admin")):
        for level in ("read", "write", "admin"):
            _authorize("pot_a", level)
    assert current_commit_actor() == "unnamed"
    with pytest.raises(ValueError):
        _authorize("pot_a", "owner")


def _journal_runtime(home: Path):
    from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
        EmbeddedGraphBackend,
    )
    from potpie_context_engine.adapters.outbound.graph.local_commit_mirror import (
        LocalCommitMirror,
    )
    from potpie_context_engine.adapters.outbound.graph.local_rollback_previews import (
        LocalRollbackPreviews,
    )
    from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
        LocalJsonGraphPlanStore,
    )
    from potpie_context_engine.api import build_graph_runtime

    return build_graph_runtime(
        EmbeddedGraphBackend(home=home),
        LocalJsonGraphPlanStore(home=home),
        commit_mirror=LocalCommitMirror(home / "graph_commits.sqlite"),
        preview_store=LocalRollbackPreviews(home / "rollback_previews.sqlite"),
        commit_host="host:test",
        commit_actor=current_commit_actor,
        commit_authorize=authorize_scoped_commit,
    )


@pytest.mark.anyio
async def test_a_reader_reads_history_but_cannot_roll_back(tmp_path: Path) -> None:
    runtime = _journal_runtime(tmp_path)
    runtime.backend.journal.activate(pot_id="pot_a", rollback_enabled=True)
    commits = runtime.workbench.commit_service

    with authorization_scope(_scope("reader")):
        status = await commits.journal_status_async(pot_id="pot_a")
        with pytest.raises(CommitAccessDenied):
            await commits.disable_rollback_async(pot_id="pot_a")
    with authorization_scope(_scope("admin")):
        disabled = await commits.disable_rollback_async(pot_id="pot_a")

    assert status["ok"] is True
    assert status["state"].rollback_enabled is True
    assert disabled["ok"] is True


# --- caller-asserted identities on a managed host ------------------------------


def _request(operation: EngineOperation, payload: object) -> EngineOperationRequest:
    return EngineOperationRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="request-1",
        operation=operation,
        selector=ContextSelector("explicit", "pot_a"),
        payload=payload,  # type: ignore[arg-type]
    )


def _apply(operation: EngineOperation, payload: object):
    return RemoteRequestPolicy().apply(_request(operation, payload), actor=ALICE)


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        (EngineOperation.PROPOSE, ProposeRequest()),
        (EngineOperation.COMMIT, CommitRequest(plan_id="plan-1")),
    ],
)
def test_an_empty_approval_stays_empty_so_the_approval_gate_still_applies(
    operation, payload
) -> None:
    outcome = _apply(operation, payload)

    assert isinstance(outcome, Success)
    assert outcome.value.payload.approved_by is None


@pytest.mark.parametrize(
    ("operation", "payload", "field_name"),
    [
        (EngineOperation.INBOX_CLAIM, InboxClaimRequest(item_id="i"), "claimed_by"),
        (
            EngineOperation.INBOX_MARK_APPLIED,
            InboxMarkAppliedRequest(item_id="i"),
            "closed_by",
        ),
        (
            EngineOperation.INBOX_MARK_REJECTED,
            InboxMarkRejectedRequest(item_id="i", closed_by="  "),
            "closed_by",
        ),
        (EngineOperation.INBOX_CLOSE, InboxCloseRequest(item_id="i"), "closed_by"),
    ],
)
def test_an_empty_attribution_is_stamped_with_the_actor(
    operation, payload, field_name
) -> None:
    outcome = _apply(operation, payload)

    assert isinstance(outcome, Success)
    assert getattr(outcome.value.payload, field_name) == "alice"


@pytest.mark.parametrize(
    ("operation", "payload", "field_name"),
    [
        (EngineOperation.PROPOSE, ProposeRequest(approved_by="mallory"), "approved_by"),
        (
            EngineOperation.COMMIT,
            CommitRequest(plan_id="p", approved_by="mallory"),
            "approved_by",
        ),
        (
            EngineOperation.INBOX_CLAIM,
            InboxClaimRequest(item_id="i", claimed_by="mallory"),
            "claimed_by",
        ),
        (
            EngineOperation.INBOX_CLOSE,
            InboxCloseRequest(item_id="i", closed_by="mallory"),
            "closed_by",
        ),
        (
            EngineOperation.INBOX_ADD,
            InboxAddRequest(summary="s", created_by={"actor": "mallory"}),
            "created_by.actor",
        ),
        (
            EngineOperation.INBOX_ADD,
            InboxAddRequest(summary="s", created_by={"actor": {"id": "alice"}}),
            "created_by.actor",
        ),
    ],
)
def test_asserting_someone_else_is_refused(operation, payload, field_name) -> None:
    outcome = _apply(operation, payload)

    assert isinstance(outcome, Failure)
    assert outcome.error.category == "authorization"
    assert outcome.error.code == "caller_assertion_mismatch"
    assert outcome.error.details == {
        "operation": operation.value,
        "field": field_name,
    }


def test_asserting_yourself_passes_unchanged() -> None:
    request = _request(
        EngineOperation.COMMIT, CommitRequest(plan_id="p", approved_by="alice")
    )

    assert RemoteRequestPolicy().apply(request, actor=ALICE) == Success(request)


def test_created_by_keeps_its_attribution_and_gains_the_actor() -> None:
    outcome = _apply(
        EngineOperation.INBOX_ADD,
        InboxAddRequest(summary="s", created_by={"surface": "cli", "harness": "x"}),
    )

    assert isinstance(outcome, Success)
    assert outcome.value.payload.created_by == {
        "surface": "cli",
        "harness": "x",
        "actor": "alice",
    }


def test_the_inbox_list_claimed_by_filter_is_not_an_assertion() -> None:
    request = _request(EngineOperation.INBOX_LIST, InboxListRequest(claimed_by="bob"))

    assert RemoteRequestPolicy().apply(request, actor=ALICE) == Success(request)


@pytest.mark.parametrize(
    ("operation", "payload", "field_name"),
    [
        (EngineOperation.EXPORT_SNAPSHOT, ExportSnapshotRequest(), "destination"),
        (
            EngineOperation.EXPORT_SNAPSHOT,
            ExportSnapshotRequest(destination="exports/pot.json", version=2),
            "destination",
        ),
        (EngineOperation.IMPORT_SNAPSHOT, ImportSnapshotRequest(), "source"),
        (
            EngineOperation.IMPORT_SNAPSHOT,
            ImportSnapshotRequest(source="exports/pot.json", version=2),
            "source",
        ),
    ],
)
def test_host_paths_are_refused(operation, payload, field_name) -> None:
    outcome = _apply(operation, payload)

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "host_path_refused"
    assert outcome.error.details == {"operation": operation.value, "field": field_name}


def test_a_version_two_snapshot_moves_as_data() -> None:
    export = _request(EngineOperation.EXPORT_SNAPSHOT, ExportSnapshotRequest(version=2))
    load = _request(
        EngineOperation.IMPORT_SNAPSHOT,
        ImportSnapshotRequest(version=2, payload={"entities": []}),
    )

    assert RemoteRequestPolicy().apply(export, actor=ALICE) == Success(export)
    assert RemoteRequestPolicy().apply(load, actor=ALICE) == Success(load)

"""A reference multi-actor managed host, built only from ``potpie.runtime.hosting``.

It is what an embedding host writes, kept small and in memory: bearer keys
mapped to actors, pots shared through per-actor grants with a role, a role
authorizer over ``operation_spec``, the ``host.*`` handler, one extension
operation, and the engine services over one real graph runtime. Contract tests
drive its :class:`OperationEndpoint` directly (``handle(body, headers)``) or
through an ``httpx`` transport, exactly as a remote client would.

The only root-Potpie import here is ``potpie.runtime.hosting``; a test keeps it
that way, so the kit's import surface is shown to be sufficient.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from potpie.runtime.hosting import (
    DEFAULT_MAX_REQUEST_BYTES,
    ActorPotCatalog,
    AgentContextService,
    AuthenticatedActor,
    AuthenticationError,
    AuthorizationError,
    AuthorizationScope,
    CatalogSelectorResolver,
    ContextResourceManager,
    DaemonBuild,
    DomainError,
    EndpointResponse,
    EngineOperation,
    EngineServices,
    Failure,
    GrantListResult,
    GrantRevokeResult,
    GraphMetadataOperationHandler,
    HostExtension,
    HostIdentity,
    HostOperation,
    OperationCoordinator,
    OperationEndpoint,
    PassThroughAuthenticator,
    PerRequestCompatibility,
    PotAggregateStatus,
    PotCounts,
    PotGrantInfo,
    PotInfo,
    PotListResult,
    PotRepoSource,
    RemoteRequestPolicy,
    RepoDefaultClearResult,
    RepoDefaultResult,
    SafetyClass,
    ServiceResourceComposer,
    SourceInfo,
    SourceListResult,
    Success,
    TypedOperationHandler,
    WhoamiResult,
    authorize_scoped_commit,
    current_actor_id,
    current_commit_actor,
    operation_spec,
)

#: Bearer keys of the reference actors. Alice administers ``pot_shop``, Bob
#: reads it, Carol holds no grant at all.
ALICE_KEY = "ref_alice_" + "a" * 32
BOB_KEY = "ref_bob_" + "b" * 32
CAROL_KEY = "ref_carol_" + "c" * 32
HOST_ID = "reference-host"
HOST_VERSION = "0.0.0+reference"
ROLE_RANK = {"reader": 1, "writer": 2, "admin": 3}

#: Operations the reference authorizer always reserves for an admin, beyond
#: every destructive one.
_ADMIN_OPERATIONS = frozenset(
    {
        HostOperation.GRANT_LIST.value,
        HostOperation.GRANT_SET.value,
        HostOperation.GRANT_REVOKE.value,
    }
)


@dataclass
class ReferenceState:
    """Everything the reference host persists, in memory."""

    keys: dict[str, str] = field(
        default_factory=lambda: {
            ALICE_KEY: "alice",
            BOB_KEY: "bob",
            CAROL_KEY: "carol",
        }
    )
    pots: dict[str, PotInfo] = field(
        default_factory=lambda: {
            "pot_shop": PotInfo(pot_id="pot_shop", name="shop"),
            "pot_lab": PotInfo(pot_id="pot_lab", name="lab"),
        }
    )
    grants: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "pot_shop": {"alice": "admin", "bob": "reader"},
            "pot_lab": {"alice": "writer"},
        }
    )
    active: dict[str, str] = field(default_factory=dict)
    repo_defaults: dict[tuple[str, str], str] = field(default_factory=dict)
    sources: dict[str, list[SourceInfo]] = field(default_factory=dict)
    handled_by: list[str | None] = field(default_factory=list)

    def role(self, actor_id: str, pot_id: str) -> str | None:
        return self.grants.get(pot_id, {}).get(actor_id)


class KeyAuthenticator:
    """``Authorization: Bearer <key>`` → one actor; nothing else is read.

    A request without the header is the anonymous actor only when the host is
    configured that way (a loopback development host with auth disabled);
    otherwise it is refused like a wrong key.
    """

    def __init__(self, state: ReferenceState, *, allow_anonymous: bool = False):
        self._state = state
        self._allow_anonymous = allow_anonymous

    async def authenticate(self, headers: Mapping[str, str]):
        header = headers.get("authorization")
        if header is None:
            if self._allow_anonymous:
                return Success(AuthenticatedActor(actor_id="anonymous"))
            return Failure(
                AuthenticationError(
                    code="authentication_required",
                    message="this host needs an API key",
                    recommended_next_action="pass the key with 'potpie host set'",
                )
            )
        scheme, _, key = header.partition(" ")
        actor_id = self._state.keys.get(key) if scheme == "Bearer" else None
        if actor_id is None:
            return Failure(
                AuthenticationError(
                    code="authentication_failed", message="the API key was refused"
                )
            )
        return Success(AuthenticatedActor(actor_id=actor_id, attributes={"org": "ref"}))


class GrantCatalog(ActorPotCatalog):
    """The pots an actor holds a grant on, and that actor's pointers."""

    def __init__(self, state: ReferenceState) -> None:
        self._state = state

    async def pots_for(self, actor):
        return [
            pot
            for pot_id, pot in self._state.pots.items()
            if self._state.role(actor.actor_id, pot_id) is not None
        ]

    async def active_for(self, actor):
        pot_id = self._state.active.get(actor.actor_id)
        if pot_id is None or self._state.role(actor.actor_id, pot_id) is None:
            return None
        return self._state.pots.get(pot_id)

    async def repo_default_for(self, actor, repo):
        return self._state.repo_defaults.get((actor.actor_id, repo))

    async def repo_sources_for(self, actor):
        return [
            PotRepoSource(
                pot_id=pot.pot_id,
                pot_name=pot.name,
                name=source.name,
                location=source.location,
            )
            for pot in await self.pots_for(actor)
            for source in self._state.sources.get(pot.pot_id, ())
            if source.kind == "repo"
        ]


class RoleAuthorizer:
    """Reader reads, writer writes, admin does destructive work and sharing.

    The actor's role also becomes the scope's ``commit_access``, which is all
    commit history and rollback consult.
    """

    def __init__(
        self, state: ReferenceState, extensions: Iterable[HostExtension] = ()
    ) -> None:
        self._state = state
        self._extensions = tuple(extensions)

    async def authorize(self, actor, operation, context):
        role = self._state.role(actor.actor_id, context.value)
        spec = operation_spec(operation, extensions=self._extensions)
        if role is None or spec is None:
            # Selection already hid pots without a grant; this is defence.
            return Failure(
                AuthorizationError(code="forbidden", message="not authorized")
            )
        if spec.destructive or operation in _ADMIN_OPERATIONS:
            needed = "admin"
        elif spec.safety is SafetyClass.SHARED_CONTEXT_READ:
            needed = "reader"
        else:
            needed = "writer"
        if ROLE_RANK[role] < ROLE_RANK[needed]:
            return Failure(
                AuthorizationError(
                    code="role_insufficient",
                    message=f"{operation} needs the {needed} role on this pot",
                    details={"operation": operation, "required_role": needed},
                )
            )
        return Success(
            AuthorizationScope(
                actor_id=actor.actor_id,
                operation=operation,
                context=context,
                attributes={"commit_access": role},
            )
        )


class ReferenceHostHandler:
    """The ``host.*`` catalog over :class:`ReferenceState`."""

    def __init__(self, state: ReferenceState) -> None:
        self._state = state

    async def handle(self, operation, request, *, actor, context):
        self._state.handled_by.append(current_actor_id())
        method = getattr(self, "_" + operation.value.removeprefix("host."))
        return method(request, actor.actor_id, context)

    def _pot_list(self, request, actor_id, context):
        pots = tuple(
            replace(pot, active=self._state.active.get(actor_id) == pot.pot_id)
            for pot_id, pot in self._state.pots.items()
            if self._state.role(actor_id, pot_id) is not None
            and (request.include_archived or not pot.archived)
        )
        status = (
            {
                pot.pot_id: PotCounts(
                    source_count=len(self._state.sources.get(pot.pot_id, ())),
                    counts={"claims": 0},
                )
                for pot in pots
            }
            if request.include_status
            else {}
        )
        return PotListResult(
            pots=pots,
            active_pot_id=self._state.active.get(actor_id),
            repo_default_pot_id=(
                self._state.repo_defaults.get((actor_id, request.repo))
                if request.repo
                else None
            ),
            status=status,
        )

    def _pot_create(self, request, actor_id, context):
        if not request.name.strip():
            return Failure(
                DomainError(code="validation_error", message="name is required")
            )
        pot_id = f"pot_{request.name}"
        created = pot_id not in self._state.pots
        self._state.pots.setdefault(pot_id, PotInfo(pot_id=pot_id, name=request.name))
        self._state.grants.setdefault(pot_id, {})[actor_id] = "admin"
        if request.repo:
            self._state.repo_defaults[(actor_id, request.repo)] = pot_id
        if request.use:
            self._state.active[actor_id] = pot_id
        return replace(self._state.pots[pot_id], active=request.use, created=created)

    def _pot_use(self, request, actor_id, context):
        self._state.active[actor_id] = context.value
        return replace(self._state.pots[context.value], active=True)

    def _repo_default_set(self, request, actor_id, context):
        self._state.repo_defaults[(actor_id, request.repo)] = context.value
        return RepoDefaultResult(repo=request.repo, pot_id=context.value)

    def _repo_default_clear(self, request, actor_id, context):
        cleared = self._state.repo_defaults.pop((actor_id, request.repo), None)
        return RepoDefaultClearResult(repo=request.repo, cleared=cleared is not None)

    def _pot_status(self, request, actor_id, context):
        return PotAggregateStatus(
            active_pot=self._state.pots[context.value],
            pot_count=1,
            sources=tuple(self._state.sources.get(context.value, ())),
            backend_ready=True,
        )

    def _source_list(self, request, actor_id, context):
        return SourceListResult(
            sources=tuple(self._state.sources.get(context.value, ()))
        )

    def _source_add(self, request, actor_id, context):
        rows = self._state.sources.setdefault(context.value, [])
        source = SourceInfo(
            source_id=f"src_{len(rows) + 1}",
            kind=request.kind,
            name=request.name or request.location,
            location=request.location,
            status="ok",
        )
        rows.append(source)
        return source

    def _whoami(self, request, actor_id, context):
        return WhoamiResult(subject=actor_id, mode="api_key", detail=HOST_ID)

    def _grant_list(self, request, actor_id, context):
        return GrantListResult(
            grants=tuple(
                PotGrantInfo(pot_id=context.value, actor_id=who, role=role)
                for who, role in sorted(self._state.grants[context.value].items())
            )
        )

    def _grant_set(self, request, actor_id, context):
        self._state.grants[context.value][request.actor_id] = request.role
        return PotGrantInfo(
            pot_id=context.value, actor_id=request.actor_id, role=request.role
        )

    def _grant_revoke(self, request, actor_id, context):
        revoked = self._state.grants[context.value].pop(request.actor_id, None)
        return GrantRevokeResult(
            pot_id=context.value, actor_id=request.actor_id, revoked=revoked is not None
        )


# --- one extension operation ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class EchoRequest:
    text: str = ""


@dataclass(frozen=True, slots=True)
class EchoResult:
    text: str = ""
    by: str = ""


async def _echo(request: EchoRequest, *, actor, context) -> EchoResult:
    del context
    return EchoResult(text=request.text, by=actor.actor_id)


ECHO = HostExtension(
    name="ext.reference.echo",
    request_type=EchoRequest,
    result_type=EchoResult,
    safety=SafetyClass.SHARED_CONTEXT_READ,
    handler=_echo,
)


# --- engine services ------------------------------------------------------------


@dataclass
class _AgentContext:
    """Resolve and record without retrieval: they echo what reached them."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def resolve(self, request):
        from potpie_context_engine.core.agent_envelope import AgentEnvelope

        self.calls.append(("resolve", request.pot_id))
        return AgentEnvelope(
            pot_id=request.pot_id,
            intent="resolve",
            items=(),
            coverage=(),
            metadata={"task": request.task or ""},
        )

    def search(self, request):
        return self.resolve(request)

    def record(self, request):
        from potpie_context_engine.core.ports.agent_context import RecordReceipt

        self.calls.append(("record", request.pot_id))
        return RecordReceipt(
            pot_id=request.pot_id,
            record_type=request.record_type,
            accepted=True,
            record_id=f"record-{len(self.calls)}",
        )


class AsyncOnlyStore:
    """A plan or inbox store that, like one over a database pool, is async only.

    It has no sync methods at all and, like a pool, belongs to the event loop
    that first used it: a call from any other loop (a fresh ``asyncio.run`` in
    a worker thread, say) is refused.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._loop: asyncio.AbstractEventLoop | None = None
        self.calls: list[str] = []

    def _enter(self, method: str) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError("this store belongs to another event loop")
        self.calls.append(method)

    async def save_async(self, *args: Any, **kwargs: Any) -> Any:
        self._enter("save")
        return self._store.save(*args, **kwargs)

    async def reserve_idempotency_async(self, *args: Any, **kwargs: Any) -> Any:
        self._enter("reserve_idempotency")
        return self._store.reserve_idempotency(*args, **kwargs)

    async def get_async(self, *args: Any, **kwargs: Any) -> Any:
        self._enter("get")
        return self._store.get(*args, **kwargs)

    async def compare_and_set_async(self, *args: Any, **kwargs: Any) -> Any:
        self._enter("compare_and_set")
        return self._store.compare_and_set(*args, **kwargs)

    async def list_async(self, *args: Any, **kwargs: Any) -> Any:
        self._enter("list")
        return self._store.list(*args, **kwargs)


def _graph_runtime(home: Path, *, plan_store: Any, inbox_store: Any):
    from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
        EmbeddedGraphBackend,
    )
    from potpie_context_engine.adapters.outbound.graph.local_commit_mirror import (
        LocalCommitMirror,
    )
    from potpie_context_engine.adapters.outbound.graph.local_rollback_previews import (
        LocalRollbackPreviews,
    )
    from potpie_context_engine.api import build_graph_runtime

    return build_graph_runtime(
        EmbeddedGraphBackend(home=home),
        plan_store,
        inbox_store,
        commit_mirror=LocalCommitMirror(home / "graph_commits.sqlite"),
        preview_store=LocalRollbackPreviews(home / "rollback_previews.sqlite"),
        commit_host=f"host:{HOST_ID}",
        commit_actor=current_commit_actor,
        commit_authorize=authorize_scoped_commit,
    )


@dataclass
class ReferenceHost:
    state: ReferenceState
    endpoint: OperationEndpoint
    manager: ContextResourceManager
    graph_runtime: Any
    agent_context: Any
    plan_store: Any
    inbox_store: Any

    @classmethod
    def build(
        cls,
        home: Path,
        *,
        allow_anonymous: bool = False,
        engine_operations: Iterable[EngineOperation] | None = None,
        serve_host_operations: bool = True,
        extensions: Iterable[HostExtension] = (ECHO,),
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_idle_engines: int | None = 8,
        async_stores: bool = False,
        agent_door: bool = False,
    ) -> ReferenceHost:
        """Build the host.

        ``async_stores`` makes its plan and inbox stores async only, as a host
        over a database pool has them. ``agent_door`` serves resolve, search
        and record through the kit's real agent door over the graph runtime
        instead of the echoing stand-in.
        """

        from potpie_context_engine.adapters.outbound.graph.inbox_stores.local_json import (
            LocalJsonGraphInboxStore,
        )
        from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
            LocalJsonGraphPlanStore,
        )

        state = ReferenceState()
        plan_store: Any = LocalJsonGraphPlanStore(home=home)
        inbox_store: Any = LocalJsonGraphInboxStore(home=home)
        if async_stores:
            plan_store = AsyncOnlyStore(plan_store)
            inbox_store = AsyncOnlyStore(inbox_store)
        runtime = _graph_runtime(home, plan_store=plan_store, inbox_store=inbox_store)
        agent_context: Any = (
            AgentContextService.for_graph_runtime(runtime)
            if agent_door
            else _AgentContext()
        )
        services = EngineServices(
            pots=None,
            agent_context=agent_context,
            graph=runtime.graph,
            graph_workbench=runtime.workbench,
            backend=runtime.backend,
            nudge=None,
        )
        extensions = tuple(extensions)
        manager = ContextResourceManager(
            resolver=CatalogSelectorResolver(GrantCatalog(state)),
            authenticator=PassThroughAuthenticator(),
            authorizer=RoleAuthorizer(state, extensions),
            composer=ServiceResourceComposer(services),
            max_idle_engines=max_idle_engines,
        )
        endpoint = OperationEndpoint(
            authenticator=KeyAuthenticator(state, allow_anonymous=allow_anonymous),
            compatibility=PerRequestCompatibility(),
            engine_handler=TypedOperationHandler(
                manager,
                coordinator=OperationCoordinator(),
                context_free_handler=GraphMetadataOperationHandler(services),
            ),
            host_handler=ReferenceHostHandler(state) if serve_host_operations else None,
            extensions=extensions,
            identity=HostIdentity(
                host_id=HOST_ID,
                version=HOST_VERSION,
                build=DaemonBuild(rev="0" * 40, dirty=False, built_at=None),
                backend_profile="embedded",
            ),
            max_request_bytes=max_request_bytes,
            request_policy=RemoteRequestPolicy(),
            engine_operations=engine_operations,
        )
        return cls(
            state=state,
            endpoint=endpoint,
            manager=manager,
            graph_runtime=runtime,
            agent_context=agent_context,
            plan_store=plan_store,
            inbox_store=inbox_store,
        )

    async def post(
        self, document: object, *, key: str | None = ALICE_KEY
    ) -> EndpointResponse:
        """Send one JSON document as a remote client would."""

        headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
        return await self.endpoint.handle(json.dumps(document).encode(), headers)

    def httpx_transport(
        self,
        *,
        rewrite_request: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        rewrite_response: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        seen: list[httpx.Request] | None = None,
    ) -> httpx.MockTransport:
        """Serve ``POST <base>/v1/operations`` for an ``httpx`` client.

        The rewrite hooks change the JSON on the wire, which is how a test
        plays a newer client or a newer host against this one.
        """

        async def serve(request: httpx.Request) -> httpx.Response:
            if seen is not None:
                seen.append(request)
            if request.method != "POST" or not request.url.path.endswith(
                "/v1/operations"
            ):
                return httpx.Response(404, json={"detail": "not found"})
            body = await request.aread()
            if rewrite_request is not None:
                body = json.dumps(rewrite_request(json.loads(body))).encode()
            answer = await self.endpoint.handle(body, dict(request.headers))
            payload = answer.body
            if rewrite_response is not None:
                payload = json.dumps(rewrite_response(json.loads(payload))).encode()
            return httpx.Response(
                answer.status, content=payload, headers=dict(answer.headers)
            )

        return httpx.MockTransport(serve)

    async def close(self) -> None:
        await self.manager.shutdown()


__all__ = [
    "ALICE_KEY",
    "AsyncOnlyStore",
    "BOB_KEY",
    "CAROL_KEY",
    "ECHO",
    "HOST_ID",
    "HOST_VERSION",
    "EchoRequest",
    "EchoResult",
    "GrantCatalog",
    "KeyAuthenticator",
    "ReferenceHost",
    "ReferenceHostHandler",
    "ReferenceState",
    "RoleAuthorizer",
]

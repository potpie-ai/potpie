"""ContextGraphComponent — exposes the context-engine agent surface under the daemon shell.

The detached daemon runs this component; its four operations (``context.resolve``/``search``/
``record``/``status``) delegate to the in-process ``HostShell.agent_context`` handed in via
``ShellContext.config["deps"]``. So the CLI path (``potpie resolve`` in-process) and the daemon
path (``context.resolve`` over the socket) terminate at the *same* ``AgentContextPort``.
"""
from __future__ import annotations
import logging
from typing import Any

from domain.ports.daemon.shell import HealthStatus
from host.daemon_runtime.context import ShellContext
from domain.ports.daemon.operations import OperationError, OperationSpec

logger = logging.getLogger("context_graph.component")


class ContextGraphComponent:
    name = "context_graph"

    def __init__(self, *, graph: str | None = None, relational: str | None = None, **extra: Any) -> None:
        self._graph_ref = graph
        self._relational_ref = relational
        self._extra = extra
        self._ctx: ShellContext | None = None
        self._host: Any = None  # HostShell, from ctx.config["deps"]
        self._graph_endpoint: str | None = None
        self._relational_url: str | None = None
        self._status: HealthStatus = HealthStatus.STOPPED

    async def on_start(self, ctx: ShellContext) -> None:
        self._ctx = ctx
        self._status = HealthStatus.STARTING
        self._host = (ctx.config or {}).get("deps")
        # Optional managed-service refs: "service:<name>" -> endpoint; any other value passes through.
        if self._graph_ref:
            self._graph_endpoint = ctx.endpoints.resolve(self._graph_ref)
        if self._relational_ref:
            self._relational_url = ctx.endpoints.resolve(self._relational_ref)
        logger.info(
            "context_graph ready (host=%s, graph=%s, relational=%s)",
            type(self._host).__name__ if self._host else None,
            self._graph_endpoint, self._relational_url,
        )
        self._status = HealthStatus.READY

    async def on_stop(self) -> None:
        self._status = HealthStatus.STOPPED

    def health(self) -> HealthStatus:
        return self._status

    # --- host access used by the ops ---------------------------------------
    @property
    def agent_context(self):
        if self._host is None:
            raise OperationError(
                "unavailable",
                "context host not wired",
                recommended_next_action="start the daemon via `potpie daemon start`",
            )
        return self._host.agent_context

    def resolve_pot_id(self, explicit: str | None) -> str:
        """Resolve an explicit pot ref → id, else the active pot. Raises if none."""
        if self._host is None:
            raise OperationError(
                "unavailable",
                "context host not wired",
                recommended_next_action="start the daemon via `potpie daemon start`",
            )
        pots = self._host.pots
        if explicit:
            for pot in pots.list_pots():
                if explicit in (pot.pot_id, pot.name):
                    return pot.pot_id
            raise OperationError("not_found", f"no pot matching {explicit!r}")
        active = pots.active_pot()
        if active is None:
            raise OperationError(
                "not_found",
                "no active pot",
                recommended_next_action="run `potpie setup` or `potpie pot create`",
            )
        return active.pot_id

    def operations(self) -> list[OperationSpec]:
        from adapters.inbound.shell_component.ops_resolve import build_op as _resolve
        from adapters.inbound.shell_component.ops_search import build_op as _search
        from adapters.inbound.shell_component.ops_record import build_op as _record
        from adapters.inbound.shell_component.ops_status import build_op as _status
        return [_resolve(self), _search(self), _record(self), _status(self)]


def register(registry) -> None:
    """Entry point for ``potpie.shell.components``: register the context_graph factory."""
    registry.register("context_graph", lambda **cfg: ContextGraphComponent(**cfg))

"""The RPC surfaces this daemon serves, declared where something can check them.

The allowlist that guards ``POST /rpc`` is not one list — it is two. The managed
service runs its own build of this daemon and keeps its own copy, and nothing
has ever compared them. They drifted: the managed deployment does not list
``resources``, so every ``potpie resource ...`` command against it fails, and
``potpie doctor`` (which asks for resource status unconditionally) was dead
there outright.

Worse than the drift was its shape. An undeclared surface was refused with
``ValueError("invalid RPC surface: resources")``, which crosses the wire as
``validation_error`` — the code the CLI reserves for *the caller got it wrong*,
rendered at exit 1 with no next action. The caller got nothing wrong. "This host
does not implement that" is a capability answer, and
:class:`~potpie_context_core.errors.CapabilityNotImplemented` is the vocabulary
the whole stack already has for it.

Two defences live here, and neither is a protocol worth the name:

* the set is a *declaration* rather than a literal buried in an HTTP module, so
  a facade surface added without an entry fails in
  ``tests/unit/test_managed_surface_contract.py`` instead of at runtime on
  someone else's machine;
* :func:`surface_contract` is served at ``GET /surfaces`` so a client can ask a
  host what it implements rather than assume its own copy applies to it.
  :mod:`potpie.daemon.negotiation` is the other end of that: the CLI reads a
  host's answer once per connection and classifies refusals against it, so
  "does this host implement that surface" stops being a guess about the wording
  of someone else's error message.

What this deliberately does *not* claim: the cross-repo half cannot be enforced
from here, because we do not control the other build and cannot fail its CI. The
enforceable substitutes are the declaration tests above plus a client that
degrades correctly against a host which does not answer — which is why
:func:`potpie.daemon.negotiation.negotiate` reports silence as ``UNKNOWN`` and
never as "implements nothing".
"""

from __future__ import annotations

from typing import Any, Final

#: Bumped only when the *shape* of :func:`surface_contract` changes, so a client
#: reading a future host can tell "I do not understand this answer" from "this
#: host implements nothing".
SURFACE_CONTRACT_VERSION: Final[int] = 1

#: Every ``HostShell`` attribute reachable through ``/rpc`` and ``/attr``, plus
#: the nested backend capability ports (``RemoteSurface._NESTED``), which are
#: addressed as a dotted path rather than as attributes of an already-resolved
#: surface.
RPC_SURFACES: Final[frozenset[str]] = frozenset(
    {
        "agent_context",
        "auth",
        "backend",
        "backend.analytics",
        "backend.claim_query",
        "backend.inspection",
        "backend.mutation",
        "backend.semantic",
        "backend.snapshot",
        "config",
        "graph",
        "graph_workbench",
        "installer",
        "ledger",
        "nudge",
        "pots",
        "resources",
        "setup",
        "skills",
    }
)

#: ``HostShell`` fields that are withheld on purpose, so that a later reader
#: cannot mistake the gap for an oversight and "fix" it by widening the
#: allowlist.
#:
#: ``daemon``
#:     Process lifecycle. The CLI drives it locally — it starts, stops and
#:     restarts the very process that would be serving the call — so routing
#:     ``daemon.stop`` through the daemon's own RPC is a foot-gun, not a
#:     feature. A managed host has no local process at all, which is why
#:     :class:`potpie.cli.hosts.StaticDiscovery` answers this family with a
#:     capability error of its own.
#: ``profile``
#:     A ``str``, not a surface. It is read over ``/attr`` as a member of
#:     ``backend``; there is nothing at ``profile.<method>`` to call.
DENIED_SURFACES: Final[frozenset[str]] = frozenset({"daemon", "profile"})


#: Surfaces where *every* member is a read, so the whole surface can run
#: concurrently. These are the backend's read-only capability ports; a write
#: reaches the graph through ``backend.mutation`` or ``graph.mutate``, never
#: through one of these.
READ_ONLY_RPC_SURFACES: Final[frozenset[str]] = frozenset(
    {
        "backend.analytics",
        "backend.claim_query",
        "backend.inspection",
    }
)

#: Per-surface members that are reads, on surfaces that also carry writes.
#:
#: **The default is exclusive.** A method that is not named here is serialized
#: exactly as every method used to be, so adding one to a service — or renaming
#: one listed here — costs throughput and never correctness. That direction is
#: the whole reason this is a declaration and not a naming convention: a
#: ``get_or_create`` matching a "reads start with get" rule would be a silent
#: lost update, and no test would fail.
#:
#: Everything listed was read for this: it either returns a projection of
#: stored state or delegates to one of the read-only ports above. The pot,
#: plan, and inbox stores publish through ``tmp.replace(path)``, so a reader
#: concurrent with a writer sees one whole version or the other — which is why
#: concurrent *reads* need no ordering at all, and why readers still exclude
#: writers below.
READ_ONLY_RPC_MEMBERS: Final[dict[str, frozenset[str]]] = {
    "agent_context": frozenset({"resolve", "search", "status"}),
    "auth": frozenset({"whoami"}),
    "backend": frozenset({"capabilities", "profile"}),
    "config": frozenset({"get", "list_public", "probe"}),
    "graph": frozenset(
        {
            "catalog",
            "catalog_async",
            "data_plane_status",
            "describe",
            "describe_async",
            "read",
            "read_async",
            "resolve",
            "resolve_async",
            "search",
            "search_async",
            "search_entities",
            "search_entities_async",
        }
    ),
    "graph_workbench": frozenset({"history", "inbox_list", "inbox_show", "quality"}),
    "installer": frozenset({"is_installed"}),
    # ``pull`` is missing on purpose: it advances the consumer cursor.
    "ledger": frozenset({"query", "sources", "status"}),
    "pots": frozenset(
        {
            "active_pot",
            "aggregate_status",
            "list_pots",
            "list_repo_defaults",
            "list_repo_sources",
            "list_sources",
            "repo_default",
            "source_status",
        }
    ),
    "resources": frozenset({"claims", "get", "index_status", "list", "status"}),
    "skills": frozenset({"list", "status"}),
}


def is_read_only(surface: str, member: str) -> bool:
    """Can this call share the daemon with another read?

    Unrecognised is not read-only. The cost of that default is a slower call;
    the cost of the other default is a race nobody would find.
    """
    if surface in READ_ONLY_RPC_SURFACES:
        return True
    return member in READ_ONLY_RPC_MEMBERS.get(surface, frozenset())


def surface_contract() -> dict[str, Any]:
    """What ``GET /surfaces`` answers: the surfaces this host actually serves.

    Constants only. This endpoint exists so a client can *ask* rather than
    assume, which is the only part of the two-repo drift this repo can fix on
    its own.
    """
    return {
        "contract": SURFACE_CONTRACT_VERSION,
        "surfaces": sorted(RPC_SURFACES),
    }


def undeclared_host_surfaces() -> tuple[str, ...]:
    """``HostShell`` fields that are neither declared nor explicitly denied.

    The failure this catches: a service added to the host facade, wired into a
    CLI command, and then refused at runtime by an allowlist nobody remembered
    to update — reported, before this module, as a caller mistake.

    ``HostShell`` is imported lazily and from inside the function: it drags the
    whole engine in, only the drift test calls this, and no CLI invocation
    should pay for a check that exists for CI.
    """
    from dataclasses import fields

    from potpie_context_engine.host.shell import HostShell

    return tuple(
        field.name
        for field in fields(HostShell)
        if field.name not in RPC_SURFACES and field.name not in DENIED_SURFACES
    )


__all__ = [
    "DENIED_SURFACES",
    "READ_ONLY_RPC_MEMBERS",
    "READ_ONLY_RPC_SURFACES",
    "RPC_SURFACES",
    "SURFACE_CONTRACT_VERSION",
    "is_read_only",
    "surface_contract",
    "undeclared_host_surfaces",
]

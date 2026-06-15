"""V2-style read views over the existing V1 read trunk (Graph V1.5 Step 2 + 6a).

A *view* is a named, V2-shaped read request (``<subgraph>.<view>``) that maps
onto a V1 include family today, so ``graph read --view`` is forward-compatible
without replacing the readers. Each view also declares a small contract —
inputs, which relations are inlined (the Retrieve axis), ranking inputs, and
whether it is a bounded traversal (the Traverse axis) — so Step 6a can wire the
three query axes against a declaration rather than ad-hoc per-command code.

``backed`` is **derived**, not hand-set: a view is backed iff its ``v1_include``
has a registered reader (``READER_BACKED_INCLUDES``). Unbacked views resolve to
an honest ``not_implemented`` rather than silent zeros. An import-time check
keeps every view's ``v1_include`` inside the advertised include vocabulary so
views and the agent surface cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    READER_BACKED_INCLUDES,
)


@dataclass(frozen=True, slots=True)
class GraphViewSpec:
    """Declarative spec for one V2-style read view."""

    name: str
    """Fully-qualified ``<subgraph>.<view>`` name (e.g. ``bugs.prior_occurrences``)."""

    subgraph: str
    view: str
    v1_include: str
    """The V1 include family this view routes to (``prior_bugs`` etc.)."""

    description: str
    backed: bool
    """True iff ``v1_include`` has a registered reader today (derived)."""

    # --- Query-axis contract (Step 6a) -------------------------------------
    inputs: tuple[str, ...] = ()
    """Scope/param names the view accepts (``service``, ``query``, ``depth`` …)."""

    inline_relations: tuple[str, ...] = ()
    """Predicates returned inline with each entity (the Retrieve/Traverse axis)."""

    ranking_inputs: tuple[str, ...] = ()
    """What the ranker weighs for this view (empty for deterministic walks)."""

    traversal: bool = False
    """True for bounded, depth-limited neighborhood walks (the Traverse axis)."""

    extra: dict[str, object] = field(default_factory=dict)

    def to_catalog_entry(self) -> dict[str, object]:
        """The shape the ``graph catalog`` advertises for this view."""
        return {
            "name": self.name,
            "subgraph": self.subgraph,
            "view": self.view,
            "v1_include": self.v1_include,
            "description": self.description,
            "backed": self.backed,
            "inputs": list(self.inputs),
            "inline_relations": list(self.inline_relations),
            "ranking_inputs": list(self.ranking_inputs),
            "traversal": self.traversal,
            "extra": dict(self.extra),
        }


def _v(
    subgraph: str,
    view: str,
    *,
    v1_include: str,
    description: str,
    inputs: tuple[str, ...] = (),
    inline_relations: tuple[str, ...] = (),
    ranking_inputs: tuple[str, ...] = (),
    traversal: bool = False,
    extra: dict[str, object] | None = None,
) -> GraphViewSpec:
    return GraphViewSpec(
        name=f"{subgraph}.{view}",
        subgraph=subgraph,
        view=view,
        v1_include=v1_include,
        description=description,
        backed=v1_include in READER_BACKED_INCLUDES,
        inputs=inputs,
        inline_relations=inline_relations,
        ranking_inputs=ranking_inputs,
        traversal=traversal,
        extra=dict(extra or {}),
    )


_VIEW_LIST: tuple[GraphViewSpec, ...] = (
    _v(
        "preferences",
        "active_preferences",
        v1_include="coding_preferences",
        description="Active coding preferences / policies that apply to a scope, "
        "ranked by relevance, strength, recency, and scope overlap.",
        inputs=("repo", "scope", "path", "query"),
        inline_relations=("POLICY_APPLIES_TO",),
        ranking_inputs=(
            "semantic_similarity",
            "strength",
            "recency",
            "scope_overlap",
            "corroboration",
        ),
    ),
    _v(
        "bugs",
        "prior_occurrences",
        v1_include="prior_bugs",
        description="Prior bug occurrences matching a symptom, with their fix / "
        "verification relations inlined.",
        inputs=("query", "service", "repo", "time_window"),
        inline_relations=("REPRODUCES", "RESOLVED", "ATTEMPTED_FIX_FAILED", "VERIFIED"),
        ranking_inputs=(
            "semantic_similarity",
            "recency",
            "scope_overlap",
            "corroboration",
        ),
    ),
    _v(
        "recent_changes",
        "timeline",
        v1_include="timeline",
        description="Recent activity (PRs, tickets, deploys) touching a scope, "
        "ordered by time.",
        inputs=("scope", "time_window", "query"),
        inline_relations=("TOUCHED", "PERFORMED", "MENTIONS"),
        ranking_inputs=("semantic_similarity", "recency", "scope_overlap"),
    ),
    _v(
        "infra_topology",
        "service_neighborhood",
        v1_include="infra_topology",
        description="Depth-bounded, direction-aware service neighborhood with "
        "environment-qualified dependency and binding edges.",
        inputs=("service", "depth", "direction", "environment"),
        inline_relations=(
            "DEPENDS_ON",
            "USES",
            "USES_ADAPTER",
            "CONFIGURES",
            "DEPLOYED_WITH",
            "DEPLOYED_TO",
            "DEFINED_IN",
            "HOSTED_ON",
            "OWNED_BY",
            "PROVIDES",
            "EXPOSES",
        ),
        traversal=True,
        extra={
            "environment_filter": {
                "default": "qualified_only",
                "include_unqualified_scope_key": "include_unqualified_environment",
                "rule": (
                    "When environment is set, only rows with the same environment "
                    "qualifier are traversed unless the scope explicitly sets "
                    "include_unqualified_environment=true."
                ),
            }
        },
    ),
    _v(
        "features",
        "provided",
        v1_include="features",
        description="Features / capabilities a repository or service provides "
        "(PROVIDES) and where they are implemented (IMPLEMENTED_IN). Answers "
        "'what does this repo do?'. Anchor with scope "
        "anchor_entity_key:<repo-or-service-key>, or omit scope for every "
        "feature in the pot.",
        inputs=("scope", "service", "query"),
        inline_relations=("PROVIDES", "IMPLEMENTED_IN"),
        ranking_inputs=("recency", "scope_overlap"),
        traversal=True,
    ),
    _v(
        "admin",
        "inspection_slice",
        v1_include="raw_graph",
        description="The whole canonical subgraph for the graph explorer "
        "(operators / visualization, not a use-case slice).",
        inputs=(),
        inline_relations=(),
    ),
    _v(
        "decisions",
        "active_decisions",
        v1_include="decisions",
        description="Active architectural decisions affecting a scope.",
        inputs=("scope", "query"),
        inline_relations=("DECIDED", "AFFECTS"),
        ranking_inputs=("semantic_similarity", "recency"),
    ),
    _v(
        "ownership",
        "owner_context",
        v1_include="owners",
        description="Who owns a service / repo and their team context.",
        inputs=("scope",),
        inline_relations=("OWNED_BY", "MEMBER_OF"),
    ),
    _v(
        "docs",
        "reference_context",
        v1_include="docs",
        description="Reference documentation pointers for a scope.",
        inputs=("scope", "query"),
        inline_relations=(),
        ranking_inputs=("semantic_similarity",),
    ),
)


GRAPH_VIEWS: dict[str, GraphViewSpec] = {spec.name: spec for spec in _VIEW_LIST}


def view_spec(name: str) -> GraphViewSpec | None:
    """Resolve a view by its ``<subgraph>.<view>`` name."""
    return GRAPH_VIEWS.get((name or "").strip())


def backed_views() -> tuple[GraphViewSpec, ...]:
    return tuple(spec for spec in _VIEW_LIST if spec.backed)


def views_for_catalog() -> list[dict[str, object]]:
    return [spec.to_catalog_entry() for spec in _VIEW_LIST]


def _check_views_coherent() -> None:
    """Import-time guard: every view's ``v1_include`` must be advertised.

    Mirrors :mod:`domain.coherence` — a view pointing at an include that the
    agent surface does not advertise is a silent dead end.
    """
    errors = [
        f"graph view {spec.name!r} routes to v1_include {spec.v1_include!r} "
        f"which is not in CONTEXT_INCLUDE_VALUES"
        for spec in _VIEW_LIST
        if spec.v1_include not in CONTEXT_INCLUDE_VALUES
    ]
    if errors:
        raise RuntimeError("graph_views incoherent:\n  - " + "\n  - ".join(errors))


_check_views_coherent()


__all__ = [
    "GRAPH_VIEWS",
    "GraphViewSpec",
    "backed_views",
    "view_spec",
    "views_for_catalog",
]

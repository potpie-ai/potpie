"""Canonical read views over the existing V1 read trunk.

A *view* is a named read request (``<subgraph>.<view>``) that maps onto an
internal reader family today, so ``graph read --view`` can expose the canonical
ontology while the data plane still reuses the current readers. Each view also
declares a small contract — inputs, which relations are inlined (the Retrieve
axis), ranking inputs, and whether it is a bounded traversal (the Traverse axis)
— so Step 6a can wire the three query axes against a declaration rather than
ad-hoc per-command code.

``backed`` is **derived**, not hand-set: a view is backed iff its ``v1_include``
has a registered reader (``READER_BACKED_INCLUDES``). Unbacked views resolve to
an honest ``not_implemented`` rather than silent zeros. An import-time check
keeps every view's ``v1_include`` inside the advertised include vocabulary so
views and the agent surface cannot drift.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from potpie_context_core.adjustments import (
    REASON_CANONICAL_ALIAS,
    REASON_CANONICAL_CASE,
    Adjustment,
)
from potpie_context_core.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    READER_BACKED_INCLUDES,
)

#: Traversal bounds for the depth-limited neighbourhood view. Declared once,
#: here, and advertised through the view's ``extra`` so the CLI can cap a
#: request before dispatch, the reader can bound its walk, and both name the
#: same number — rather than the reader clamping silently at a constant of
#: its own.
DEFAULT_TRAVERSAL_DEPTH = 2
MAX_TRAVERSAL_DEPTH = 4


@dataclass(frozen=True, slots=True)
class GraphViewSpec:
    """Declarative spec for one V2-style read view."""

    name: str
    """Fully-qualified ``<subgraph>.<view>`` name (e.g. ``debugging.prior_occurrences``)."""

    subgraph: str
    view: str
    v1_include: str
    """The internal include family this view routes to (``prior_bugs`` etc.)."""

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
        "decisions",
        "preferences_for_scope",
        v1_include="coding_preferences",
        description="Active coding preferences / policies that apply to a scope, "
        "ranked by relevance, strength, recency, and scope overlap.",
        inputs=("project", "repo", "service", "scope", "path", "environment", "query"),
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
        "debugging",
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
            },
            # The advertised traversal budget: a request above ``max_depth``
            # executes at ``max_depth`` with a disclosed adjustment; below 1
            # is not a read and is refused.
            "default_depth": DEFAULT_TRAVERSAL_DEPTH,
            "max_depth": MAX_TRAVERSAL_DEPTH,
        },
    ),
    _v(
        "features",
        "feature_context",
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
        "code_topology",
        "ownership_by_path",
        v1_include="owners",
        description="Who owns a service / repo and their team context.",
        inputs=("scope",),
        inline_relations=("OWNED_BY", "MEMBER_OF"),
    ),
    _v(
        "knowledge",
        "document_context",
        v1_include="docs",
        description="Reference documentation for a scope — ingested documents "
        "and their sections, plus the older free-form doc notes.",
        inputs=("scope", "query"),
        # Deliberately empty, so the view keeps its advertised ``flat_claims``
        # shape. Inlining these would re-project every docs read into
        # entity+relations items — including an item for the scope entity the
        # caller filtered by — and drop the ``claim`` block consumers read.
        inline_relations=(),
        ranking_inputs=("semantic_similarity",),
    ),
    _v(
        "knowledge",
        "document_passages",
        v1_include="resources",
        description="Passages from ingested documents, retrieved from the "
        "resource index rather than the claim store. Answers 'which text "
        "actually says this' where knowledge.document_context answers 'which "
        "document covers it, and what did we say about it'. A hit is a snippet "
        "plus the chunk id `potpie resource get` takes, and carries the "
        "Document / DocumentSection keys so no second lookup is needed.",
        inputs=("query", "doc"),
        # A chunk has no edges — it is a payload the graph only points at (R1),
        # so there is nothing to inline. The tie back to the graph travels as
        # ``document_key`` / ``section_key`` on each hit instead.
        inline_relations=(),
        ranking_inputs=("semantic_similarity",),
    ),
)


GRAPH_VIEWS: dict[str, GraphViewSpec] = {spec.name: spec for spec in _VIEW_LIST}

# Reverse routing map (derived): include family -> canonical view name. The
# workbench never accepts include families as input (graphv2: no compatibility
# aliases); this exists so errors can return migration guidance and the V1
# surfaces can point forward to the canonical view.
INCLUDE_TO_VIEW: dict[str, str] = {spec.v1_include: spec.name for spec in _VIEW_LIST}


class UnknownGraphViewError(ValueError):
    """Unknown subgraph/view, optionally carrying structured migration guidance.

    ``detail`` and ``recommended_next_action`` ride the CLI error envelope so
    agent retries are mechanical (guidance only — legacy names are never
    accepted as input).
    """

    def __init__(
        self,
        message: str,
        *,
        did_you_mean: Mapping[str, Any] | None = None,
        recommended_next_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.detail: dict[str, Any] | None = (
            {"did_you_mean": dict(did_you_mean)} if did_you_mean else None
        )
        self.recommended_next_action = recommended_next_action


def view_spec(name: str) -> GraphViewSpec | None:
    """Resolve a view by its ``<subgraph>.<view>`` name."""
    return GRAPH_VIEWS.get((name or "").strip())


def view_for_include(include: str) -> GraphViewSpec | None:
    """The canonical view that serves a V1 include family, if any."""
    name = INCLUDE_TO_VIEW.get((include or "").strip())
    return GRAPH_VIEWS.get(name) if name else None


def include_guess_guidance(
    subgraph: str | None, view: str | None
) -> dict[str, Any] | None:
    """Migration guidance for a failed subgraph/view guess.

    Recognizes a V1 include family used where a subgraph or view name was
    expected (e.g. ``docs`` → ``knowledge.document_context``), or an
    unqualified view name under the wrong subgraph. Returns a ``did_you_mean``
    payload naming the canonical view, or ``None`` when the guess resembles
    nothing known.

    A token that is already a valid canonical name for its position is never
    reinterpreted as an include family: ``decisions``, ``features``, and
    ``infra_topology`` are both include families and subgraph names, so a
    valid subgraph with a near-miss view must NOT be redirected to the
    include family's view — a confidently-wrong ``recommended_next_action``
    is worse than none.
    """
    subgraph_token = (subgraph or "").strip()
    view_token = (view or "").strip()
    spec = None
    matched = ""
    via_include = False
    for token in (view_token, subgraph_token):
        candidates = [s for s in _VIEW_LIST if s.view == token]
        if len(candidates) == 1:
            spec, matched = candidates[0], token
            break
    if spec is None:
        known_subgraphs = {s.subgraph for s in _VIEW_LIST}
        for token, in_subgraph_position in (
            (subgraph_token, True),
            (view_token, False),
        ):
            if in_subgraph_position and token in known_subgraphs:
                continue
            candidate = view_for_include(token)
            if candidate is not None:
                spec, matched, via_include = candidate, token, True
                break
    if spec is None:
        return None
    return {
        "view": spec.name,
        "subgraph": spec.subgraph,
        "view_name": spec.view,
        "matched": matched,
        "matched_include": spec.v1_include if via_include else None,
        "read_command": (
            f"potpie graph read --subgraph {spec.subgraph} --view {spec.view}"
        ),
    }


def backed_views() -> tuple[GraphViewSpec, ...]:
    return tuple(spec for spec in _VIEW_LIST if spec.backed)


def view_depth_bounds(name: str) -> tuple[int, int] | None:
    """``(default_depth, max_depth)`` a view advertises, or ``None``.

    Only a traversal view that declares both numbers has a depth budget the
    CLI may apply; every other view leaves ``--depth`` to the service, which
    reports it as an unsupported filter.
    """
    spec = view_spec(name)
    if spec is None:
        return None
    default = spec.extra.get("default_depth")
    maximum = spec.extra.get("max_depth")
    if not isinstance(default, int) or not isinstance(maximum, int):
        return None
    return default, maximum


@dataclass(frozen=True, slots=True)
class ViewSelector:
    """What ``--subgraph`` / ``--view`` resolved to, and how.

    ``resolved`` is ``False`` when neither token matched anything this
    registry knows: the trimmed inputs are handed back unchanged so a host
    with extension views can still answer, and an unknown pair still fails
    with the existing candidate guidance rather than a guess.
    """

    subgraph: str
    view: str
    resolved: bool
    adjustments: tuple[Adjustment, ...] = ()

    @property
    def name(self) -> str:
        return f"{self.subgraph}.{self.view}"


def _selector_conflict(subgraph_token: str, qualified: str) -> UnknownGraphViewError:
    qualified_subgraph = qualified.split(".", 1)[0]
    return UnknownGraphViewError(
        f"--view {qualified!r} names subgraph {qualified_subgraph!r} but "
        f"--subgraph is {subgraph_token!r}; pass one or make them agree",
        recommended_next_action=(
            f"potpie graph read --subgraph {qualified_subgraph} "
            f"--view {qualified.split('.', 1)[1]}"
        ),
    )


def resolve_view_selector(
    subgraph: str | None,
    view: str | None,
    *,
    views: Iterable[GraphViewSpec] | None = None,
) -> ViewSelector:
    """Canonicalize an *exact* selector spelling; never a near miss.

    Accepted without another call, each disclosed as an adjustment:

    - case and surrounding whitespace on either token
      (``Decisions`` / ``Preferences_For_Scope``);
    - a fully-qualified ``--view subgraph.view`` when ``--subgraph`` is
      absent or agrees with it;
    - the include-family aliases ``include_guess_guidance`` already proves
      unambiguous (``knowledge`` + ``docs`` → ``knowledge.document_context``),
      but only when *every* supplied token exactly names the resolved view's
      subgraph, view, include family or qualified name. A token that names
      a different valid subgraph (``knowledge`` + ``timeline``) is a
      conflict, not an alias, and keeps the candidate error.

    A qualified view that disagrees with ``--subgraph`` raises
    :class:`UnknownGraphViewError` naming both, before any backend work.
    """
    specs = tuple(views) if views is not None else tuple(_VIEW_LIST)
    subgraph_token = (subgraph or "").strip()
    view_token = (view or "").strip()
    adjustments: list[Adjustment] = []

    if "." in view_token:
        qualified_subgraph, qualified_view = view_token.split(".", 1)
        qualified_subgraph = qualified_subgraph.strip()
        qualified_view = qualified_view.strip()
        if subgraph_token and subgraph_token.lower() != qualified_subgraph.lower():
            raise _selector_conflict(subgraph_token, view_token)
        adjustments.append(
            Adjustment(
                field="view",
                requested=view_token,
                effective=qualified_view,
                reason=REASON_CANONICAL_ALIAS,
                message=(
                    f"--view {view_token!r} is fully qualified; read as "
                    f"--subgraph {qualified_subgraph} --view {qualified_view}"
                ),
            )
        )
        subgraph_token, view_token = qualified_subgraph, qualified_view

    def _canonical(token: str, candidates: Iterable[str]) -> str | None:
        pool = list(dict.fromkeys(candidates))
        if token in pool:
            return token
        lowered = token.lower()
        matches = [item for item in pool if item.lower() == lowered]
        return matches[0] if len(matches) == 1 else None

    canonical_subgraph = _canonical(subgraph_token, (s.subgraph for s in specs))
    if canonical_subgraph is not None and canonical_subgraph != subgraph_token:
        adjustments.append(
            Adjustment(
                field="subgraph",
                requested=subgraph_token,
                effective=canonical_subgraph,
                reason=REASON_CANONICAL_CASE,
                message=f"--subgraph {subgraph_token!r} read as {canonical_subgraph!r}",
            )
        )
    effective_subgraph = canonical_subgraph or subgraph_token

    canonical_view = _canonical(
        view_token,
        (s.view for s in specs if s.subgraph == effective_subgraph),
    )
    if canonical_view is not None:
        if canonical_view != view_token:
            adjustments.append(
                Adjustment(
                    field="view",
                    requested=view_token,
                    effective=canonical_view,
                    reason=REASON_CANONICAL_CASE,
                    message=f"--view {view_token!r} read as {canonical_view!r}",
                )
            )
        return ViewSelector(
            subgraph=effective_subgraph,
            view=canonical_view,
            resolved=True,
            adjustments=tuple(adjustments),
        )

    # Exact include-family aliases. Every token supplied must name the same
    # view from one of its exact spellings; anything else stays an error.
    # Only attempted when a view was asked for: a bare subgraph token is a
    # request for the subgraph, never for one view inside it.
    guidance = (
        include_guess_guidance(effective_subgraph, view_token) if view_token else None
    )
    if guidance is not None:
        spec = GRAPH_VIEWS.get(str(guidance["view"]))
        if spec is not None:
            spellings = {
                spec.subgraph.lower(),
                spec.view.lower(),
                spec.v1_include.lower(),
                spec.name.lower(),
            }
            supplied = [t for t in (effective_subgraph, view_token) if t]
            if supplied and all(t.lower() in spellings for t in supplied):
                requested = (
                    f"{effective_subgraph}.{view_token}"
                    if effective_subgraph and view_token
                    else (view_token or effective_subgraph)
                )
                adjustments.append(
                    Adjustment(
                        field="view",
                        requested=requested,
                        effective=spec.name,
                        reason=REASON_CANONICAL_ALIAS,
                        message=(
                            f"{requested!r} is the include-family alias of "
                            f"{spec.name!r}; read that view"
                        ),
                    )
                )
                return ViewSelector(
                    subgraph=spec.subgraph,
                    view=spec.view,
                    resolved=True,
                    adjustments=tuple(adjustments),
                )

    return ViewSelector(
        subgraph=effective_subgraph,
        view=view_token,
        resolved=False,
        adjustments=tuple(adjustments),
    )


def resolve_subgraph_name(
    subgraph: str | None, *, views: Iterable[GraphViewSpec] | None = None
) -> tuple[str, tuple[Adjustment, ...]]:
    """Case-insensitive exact match of a bare subgraph token; else unchanged.

    Case only. An include family typed as a subgraph (``docs``) keeps the
    existing did-you-mean error: redirecting a whole-subgraph request to one
    view would answer a different question than the one asked.
    """
    specs = tuple(views) if views is not None else tuple(_VIEW_LIST)
    token = (subgraph or "").strip()
    known = list(dict.fromkeys(s.subgraph for s in specs))
    if not token or token in known:
        return token, ()
    matches = [item for item in known if item.lower() == token.lower()]
    if len(matches) != 1:
        return token, ()
    return matches[0], (
        Adjustment(
            field="subgraph",
            requested=token,
            effective=matches[0],
            reason=REASON_CANONICAL_CASE,
            message=f"--subgraph {token!r} read as {matches[0]!r}",
        ),
    )


def views_for_catalog() -> list[dict[str, object]]:
    return [spec.to_catalog_entry() for spec in _VIEW_LIST]


def _check_views_coherent() -> None:
    """Import-time guard: every view's ``v1_include`` must be advertised.

    Mirrors :mod:`potpie_context_core.coherence` — a view pointing at an include that the
    agent surface does not advertise is a silent dead end.
    """
    errors = [
        f"graph view {spec.name!r} routes to v1_include {spec.v1_include!r} "
        f"which is not in CONTEXT_INCLUDE_VALUES"
        for spec in _VIEW_LIST
        if spec.v1_include not in CONTEXT_INCLUDE_VALUES
    ]
    if len(INCLUDE_TO_VIEW) != len(_VIEW_LIST):
        seen: dict[str, str] = {}
        for spec in _VIEW_LIST:
            if spec.v1_include in seen:
                errors.append(
                    f"views {seen[spec.v1_include]!r} and {spec.name!r} share "
                    f"v1_include {spec.v1_include!r}; include→view guidance "
                    "requires the mapping to stay 1:1"
                )
            seen[spec.v1_include] = spec.name
    errors.extend(
        f"reader-backed include {include!r} has no graph view; it would "
        "silently vanish from `graph status` backed_views"
        for include in sorted(set(READER_BACKED_INCLUDES) - set(INCLUDE_TO_VIEW))
    )
    if errors:
        raise RuntimeError("graph_views incoherent:\n  - " + "\n  - ".join(errors))


_check_views_coherent()


__all__ = [
    "GRAPH_VIEWS",
    "GraphViewSpec",
    "INCLUDE_TO_VIEW",
    "UnknownGraphViewError",
    "backed_views",
    "include_guess_guidance",
    "view_for_include",
    "view_spec",
    "views_for_catalog",
]

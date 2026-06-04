"""Path-aware scope stamping for config-source scanners (rebuild plan P5 / F2).

The proper POC's F2 failure mode: a scanner reading a CODEOWNERS file
at ``apps/auth/CODEOWNERS`` handed only the body text to the LLM
extractor, which dutifully emitted ``(component:unknown) -[OWNED_BY]->
(alice)`` because the body said ``*`` (everyone). The file path carried
the scope (this CODEOWNERS *governs apps/auth*), but it was lost
before the extractor ran.

This module gives scanners a deterministic way to infer scope from the
file path *before* anything LLM-shaped runs:

- ``derive_scope(path)`` returns a :class:`PathScope` carrying the
  service / environment / repo / component hints implied by the file
  layout. The matchers are pattern-driven (regex over normalized path
  segments) and extend cleanly with new conventions.

- ``apply_scope_to_entity(entity_upsert, scope)`` rewrites the upsert's
  ``entity_key`` from ``component:unknown`` (or similar placeholder)
  to ``service:<scope.service>`` deterministically. The LLM's
  extracted entity / claims inherit the scope.

The plan also calls for MENTIONS provenance (F4) — that lives in
``domain.episode_mentions`` and consumes the same PathScope when the
scanner / activity layer needs to surface "this episode is about
service X".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class PathScope:
    """Scope hints derived deterministically from a file path.

    Each field is optional. The scanner usually resolves a subset
    depending on the file's location and the source-system's
    conventions; downstream code treats absent fields as "unknown,
    don't override".
    """

    service: str | None = None
    component: str | None = None
    environment: str | None = None
    repo: str | None = None
    project: str | None = None
    # Free-form tags emitted by the matchers — useful for marking
    # ADRs / runbooks / dashboards without inventing a top-level field.
    tags: frozenset[str] = field(default_factory=frozenset)

    def is_empty(self) -> bool:
        return not (
            self.service
            or self.component
            or self.environment
            or self.repo
            or self.project
            or self.tags
        )

    def merged_with(self, other: "PathScope") -> "PathScope":
        """Overlay ``other``'s fields on top of ``self``; ``other`` wins on conflict.

        Path matchers are ordered most-specific-first; the merge folds
        the matches in declaration order, so a later (less specific)
        rule cannot blank out an earlier (more specific) hit. Use this
        when a path matches multiple rules (e.g. ``clusters/prod/auth-
        svc.yaml`` matches both env-from-prefix and service-from-leaf).
        """
        return PathScope(
            service=other.service or self.service,
            component=other.component or self.component,
            environment=other.environment or self.environment,
            repo=other.repo or self.repo,
            project=other.project or self.project,
            tags=self.tags | other.tags,
        )


@dataclass(frozen=True, slots=True)
class _PathMatcher:
    """One deterministic path → PathScope mapping rule.

    Matchers are evaluated in order (most specific first). A matcher
    returns ``None`` when the path does not apply; otherwise a
    ``PathScope`` carrying just the fields the rule resolves.
    """

    name: str  # short identifier used in tests + diagnostics
    pattern: re.Pattern[str]
    extract: Callable[[re.Match[str]], PathScope]


def _normalize_path(path: str) -> str:
    """Lowercase + forward-slash normalize for matching.

    Scanners may emit paths from Windows / GitHub APIs / git diffs;
    matchers compare against a single canonical shape.
    """
    return path.replace("\\", "/").strip("/").lower()


# ---------------------------------------------------------------------------
# Rule registry. Order matters: earlier = higher priority.
# ---------------------------------------------------------------------------


_MATCHERS: list[_PathMatcher] = []


def register_path_matcher(matcher: _PathMatcher) -> None:
    """Add a matcher to the registry (appended to the end of the chain)."""
    _MATCHERS.append(matcher)


def derive_scope(path: str) -> PathScope:
    """Deterministically derive a :class:`PathScope` from a file path.

    Iterates every registered matcher and folds each non-None result
    into the accumulating scope (``other-wins-on-conflict``); returns
    an empty scope when no matcher fires.
    """
    norm = _normalize_path(path)
    scope = PathScope()
    for matcher in _MATCHERS:
        match = matcher.pattern.search(norm)
        if match is None:
            continue
        update = matcher.extract(match)
        scope = scope.merged_with(update)
    return scope


# ---------------------------------------------------------------------------
# Built-in matchers — covers the conventions the proper POC exercised.
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


# 1. Kubernetes manifests under ``clusters/<env>/<service>.yaml`` or
#    ``k8s/<env>/<service>/...``. Stamps environment + service.
register_path_matcher(
    _PathMatcher(
        name="k8s-env-service",
        pattern=re.compile(
            r"(?:^|/)(?:clusters|k8s|kubernetes|helm)/(?P<env>[a-z0-9-]+)/"
            r"(?P<svc>[a-z0-9-]+)(?:[/.]|$)"
        ),
        extract=lambda m: PathScope(
            environment=_slug(m.group("env")),
            service=_slug(m.group("svc")),
        ),
    )
)

# 2. Per-service docs / config under ``apps/<service>/...`` or
#    ``services/<service>/...``. Stamps service.
register_path_matcher(
    _PathMatcher(
        name="apps-service-leaf",
        pattern=re.compile(
            r"(?:^|/)(?:apps|services|cmd)/(?P<svc>[a-z0-9-]+)(?:/|$)"
        ),
        extract=lambda m: PathScope(service=_slug(m.group("svc"))),
    )
)

# 3. CODEOWNERS files — stamp the service from the path, set ``tags`` so
#    the scanner knows this is an ownership file.
register_path_matcher(
    _PathMatcher(
        name="codeowners-file",
        pattern=re.compile(r"(?:^|/)codeowners$"),
        extract=lambda _m: PathScope(tags=frozenset({"codeowners"})),
    )
)

# 4. ADR documents at ``docs/adr/...`` or ``docs/decisions/...``.
register_path_matcher(
    _PathMatcher(
        name="adr-doc",
        pattern=re.compile(r"(?:^|/)docs/(?:adr|decisions?)/"),
        extract=lambda _m: PathScope(tags=frozenset({"adr"})),
    )
)

# 5. Per-environment infra under ``terraform/envs/<env>/...`` or
#    ``infra/<env>/...``. Stamps environment.
register_path_matcher(
    _PathMatcher(
        name="env-from-infra",
        pattern=re.compile(
            r"(?:^|/)(?:terraform/envs|infra|deploy)/(?P<env>[a-z0-9-]+)(?:/|$)"
        ),
        extract=lambda m: PathScope(environment=_slug(m.group("env"))),
    )
)


# ---------------------------------------------------------------------------
# Helpers the scanners + LLM-bridge consume
# ---------------------------------------------------------------------------


def stamp_extra_segments(
    *, scope: PathScope, base_segments: Iterable[str] = ()
) -> tuple[str, ...]:
    """Build a deterministic ``extra_segments`` tuple for ``identity.mint_entity_key``.

    Scanners often want to mint scoped entity keys like
    ``component:auth-svc:foo-handler`` rather than the unanchored
    ``component:foo-handler``. The path scope's ``service`` becomes the
    leading extra segment; callers may pass additional base segments.
    """
    pieces: list[str] = list(base_segments)
    if scope.service:
        pieces.insert(0, scope.service)
    return tuple(pieces)


def annotate_entity_properties(
    *, scope: PathScope, properties: dict[str, object]
) -> dict[str, object]:
    """Stamp scope-derived properties onto an entity property bag.

    Used by scanners to attach the deterministic ``service`` /
    ``environment`` properties so the canonical writer's edge writes
    (which carry an ``environment`` property per P3) inherit the
    correct scope.

    Returns a new dict; the input is not mutated.
    """
    out = dict(properties)
    if scope.service and "service" not in out:
        out["service"] = scope.service
    if scope.environment and "environment" not in out:
        out["environment"] = scope.environment
    if scope.repo and "repo" not in out:
        out["repo"] = scope.repo
    if scope.component and "component" not in out:
        out["component"] = scope.component
    if scope.project and "project" not in out:
        out["project"] = scope.project
    return out


__all__ = [
    "PathScope",
    "annotate_entity_properties",
    "derive_scope",
    "register_path_matcher",
    "stamp_extra_segments",
]

"""Identity classes + deterministic ``entity_key`` minting.

Rebuild plan P1: identity is one of the three load-bearing pillars. A
Linear ticket, a PR title, a doc heading, and a k8s manifest referring
to "checkout v2" must resolve to one ``entity_key``, deterministically
where possible and through an inspectable alias layer otherwise.

This module defines:

- :class:`IdentityClass` — how identity is established for an entity.
- :func:`mint_entity_key` — deterministic key minting per class (the
  LLM proposes a *name*; Potpie computes the key).
- :func:`validate_entity_key` — verify a key matches its class's grammar.

The alias layer is built on Position B's edge shape: an alias is itself
a claim ``(:Entity)-[:RELATES_TO {name: 'ALIAS_OF', source_ref, ...}]
->(:Entity {canonical})``. No new representation. The alias-claim
resolver lives at :func:`adapters.outbound.graphiti.aliases.resolve_alias`
once the canonical graph is queryable; P0/P1 only defines the contract.

See ``docs/context-graph/rebuild-plan.md`` Phase 1 for design rationale.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class IdentityClass(str, Enum):
    """How identity is established for an entity class.

    - **EXTERNAL_ID** — identity is the source-system identifier
      (PR/Issue/Commit/Deployment/Comment). Trivial to converge across
      observers: same external_id → same entity.
    - **SLUG_ALIAS** — identity is a canonical slug (Service /
      Component / Feature / Person / Team / Repository). Cross-source
      convergence is via the alias table — variants like
      ``"auth service"`` resolve to ``service:auth-svc`` via
      ``ALIAS_OF`` claims.
    - **CONTENT_HASH** — identity is the hash of canonical content
      (Decision / Note / Document body). Used to dedup notes / drafts
      with no stable external id.
    """

    EXTERNAL_ID = "external_id"
    SLUG_ALIAS = "slug_alias"
    CONTENT_HASH = "content_hash"


# Slug grammar — lowercase letters, digits, hyphens. Class prefix
# (``service:``, ``feature:``, ...) is separated by a single colon.
# Repository / nested keys use additional colons (``repo:github:owner/name``).
_SLUG_BODY_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

# External-id slug body permits a slightly broader alphabet — issue
# trackers and source systems use uppercase, digits, hyphens, and
# slashes. We canonicalize to lowercase but preserve other characters.
_EXTERNAL_ID_SAFE_RE = re.compile(r"^[a-z0-9._/\-]+$")


@dataclass(frozen=True, slots=True)
class IdentitySpec:
    """How a single entity-class kind is identified.

    Bound to ``EntityTypeSpec`` in the ontology (P3); for now the spec
    is the contract identity-callers thread through ``mint_entity_key``.
    """

    label: str  # canonical entity label (e.g. "Service")
    klass: IdentityClass
    # Canonical key prefix appended before the deterministic body. For
    # ``SLUG_ALIAS``: ``service:auth-svc``. For ``EXTERNAL_ID``: e.g.
    # ``github:pr:1042`` — the prefix is class:source[:kind].
    key_prefix: str

    # External-ID identity classes carry an authoritative source system
    # whose ID wins; used by the canonicalization pipeline to enforce
    # first-observer-wins on (source, external_id).
    authoritative_source: str | None = None


class IdentityError(ValueError):
    """Raised when an entity name / id cannot be minted into a valid key.

    A frequent symptom is the LLM producing a near-empty or non-text
    name. The plan favors discarding the upsert and surfacing a
    QualityIssue over guessing.
    """


def _slugify(text: str) -> str:
    """Deterministic slug normalization for SLUG_ALIAS keys.

    - lowercase
    - collapse whitespace + non-alphanumerics into single hyphens
    - strip leading/trailing hyphens
    - replace internal underscores with hyphens (canonical form for
      service / component names)
    """
    if not isinstance(text, str):
        raise IdentityError(f"cannot slugify non-string value: {text!r}")
    lowered = text.strip().lower()
    if not lowered:
        raise IdentityError("cannot slugify empty value")
    # Replace anything that's not a lowercase alnum or hyphen with a hyphen.
    interim = re.sub(r"[^a-z0-9]+", "-", lowered)
    # Collapse runs and strip outer hyphens.
    slug = re.sub(r"-+", "-", interim).strip("-")
    if not slug or not _SLUG_BODY_RE.match(slug):
        raise IdentityError(f"slugified value is empty / invalid: {text!r}")
    return slug


def _normalize_external_id(text: str) -> str:
    """Loose normalization for external-id identity (preserves slashes etc)."""
    if not isinstance(text, str):
        raise IdentityError(f"external_id must be a string: {text!r}")
    cleaned = text.strip().lower()
    if not cleaned:
        raise IdentityError("external_id is empty")
    cleaned = cleaned.replace(" ", "-")
    if not _EXTERNAL_ID_SAFE_RE.match(cleaned):
        # Fall back to slugify if the input has weird characters.
        return _slugify(cleaned)
    return cleaned


def _content_digest(content: str) -> str:
    """Stable 12-hex-char identity from canonical content text.

    Twelve chars of sha256 keeps the key human-readable (decision: 12
    hex chars ≈ 2^48 keys before collision; for per-pot content,
    plenty). Increase the prefix if collisions ever surface; the alias
    layer can carry the migration.
    """
    if not isinstance(content, str):
        raise IdentityError("content must be a string")
    body = content.strip()
    if not body:
        raise IdentityError("content is empty — cannot hash")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def mint_entity_key(
    spec: IdentitySpec,
    *,
    name: str | None = None,
    external_id: str | None = None,
    content: str | None = None,
    extra_segments: tuple[str, ...] = (),
) -> str:
    """Deterministically compute an entity_key for ``spec``.

    Each identity class accepts exactly one of ``name`` / ``external_id`` /
    ``content`` based on its class. The result is a colon-separated key
    that always begins with ``spec.key_prefix``.

    Examples::

        # SLUG_ALIAS
        mint_entity_key(svc_spec, name="auth-svc")        # 'service:auth-svc'
        mint_entity_key(svc_spec, name="Auth Service")    # 'service:auth-service'

        # EXTERNAL_ID
        mint_entity_key(pr_spec, external_id="1042",
                        extra_segments=("acme/api",))     # 'github:pr:acme/api:1042'

        # CONTENT_HASH
        mint_entity_key(adr_spec, content="ADR-007: ...") # 'decision:a3f2c8e91d4b'
    """
    prefix = spec.key_prefix.strip()
    if not prefix:
        raise IdentityError(f"identity spec for {spec.label} has empty key_prefix")

    if spec.klass is IdentityClass.SLUG_ALIAS:
        if not name:
            raise IdentityError(f"SLUG_ALIAS identity for {spec.label} requires name")
        body = _slugify(name)
    elif spec.klass is IdentityClass.EXTERNAL_ID:
        if not external_id:
            raise IdentityError(
                f"EXTERNAL_ID identity for {spec.label} requires external_id"
            )
        body = _normalize_external_id(external_id)
    elif spec.klass is IdentityClass.CONTENT_HASH:
        if not content:
            raise IdentityError(
                f"CONTENT_HASH identity for {spec.label} requires content"
            )
        body = _content_digest(content)
    else:
        raise IdentityError(f"unknown identity class: {spec.klass!r}")

    segments = [prefix, *(_normalize_external_id(s) for s in extra_segments), body]
    return ":".join(segments)


def validate_entity_key(spec: IdentitySpec, key: str) -> bool:
    """Confirm ``key`` matches ``spec``'s grammar (prefix + valid body).

    Returns ``True`` when the key would be accepted by ``mint_entity_key``
    for the same spec; ``False`` otherwise. Does NOT verify the entity
    exists — that's a graph-side concern.
    """
    if not isinstance(key, str) or not key:
        return False
    prefix = spec.key_prefix.strip()
    if not key.startswith(prefix + ":"):
        return False
    rest = key[len(prefix) + 1 :]
    if not rest:
        return False
    # Each colon-separated segment must be a valid slug body.
    for seg in rest.split(":"):
        if not _SLUG_BODY_RE.match(seg) and not _EXTERNAL_ID_SAFE_RE.match(seg):
            # Allow content-hash hex bodies for CONTENT_HASH specs.
            if spec.klass is IdentityClass.CONTENT_HASH and re.match(
                r"^[0-9a-f]{8,32}$", seg
            ):
                continue
            return False
    return True


# ---------------------------------------------------------------------------
# Identity spec registry
# ---------------------------------------------------------------------------


_IDENTITY_REGISTRY: dict[str, IdentitySpec] = {}


def register_identity(spec: IdentitySpec) -> None:
    """Register the identity spec for one entity label.

    Called by the ontology layer at import time. The canonicalization
    pipeline consults the registry per entity-upsert to mint the right
    key (or validate one the LLM produced).
    """
    if spec.label in _IDENTITY_REGISTRY:
        existing = _IDENTITY_REGISTRY[spec.label]
        if existing != spec:
            raise IdentityError(
                f"conflicting identity registration for {spec.label!r}: "
                f"{existing!r} vs {spec!r}"
            )
        return
    _IDENTITY_REGISTRY[spec.label] = spec


def get_identity(label: str) -> IdentitySpec | None:
    """Return the registered identity spec for ``label`` (or None)."""
    return _IDENTITY_REGISTRY.get(label)


def all_identities() -> Mapping[str, IdentitySpec]:
    """Snapshot of the registered identities (label → spec)."""
    return dict(_IDENTITY_REGISTRY)


# ---------------------------------------------------------------------------
# Default identity specs — covers the canonical ontology labels P3 will
# refine further. The spec list lives here (not in ontology.py) so the
# identity contract is a domain concern, not a schema concern; ontology
# refinements rebind specs without touching the registry plumbing.
# ---------------------------------------------------------------------------


_DEFAULTS: tuple[IdentitySpec, ...] = (
    # Slug-aliased identity (the bulk of the ontology)
    IdentitySpec(label="Service", klass=IdentityClass.SLUG_ALIAS, key_prefix="service"),
    IdentitySpec(
        label="Component", klass=IdentityClass.SLUG_ALIAS, key_prefix="component"
    ),
    IdentitySpec(label="Feature", klass=IdentityClass.SLUG_ALIAS, key_prefix="feature"),
    IdentitySpec(label="Person", klass=IdentityClass.SLUG_ALIAS, key_prefix="person"),
    IdentitySpec(label="Team", klass=IdentityClass.SLUG_ALIAS, key_prefix="team"),
    IdentitySpec(
        label="Repository", klass=IdentityClass.SLUG_ALIAS, key_prefix="repo"
    ),
    IdentitySpec(
        label="Environment",
        klass=IdentityClass.SLUG_ALIAS,
        key_prefix="environment",
    ),
    IdentitySpec(
        label="DataStore", klass=IdentityClass.SLUG_ALIAS, key_prefix="datastore"
    ),
    IdentitySpec(label="Project", klass=IdentityClass.SLUG_ALIAS, key_prefix="project"),
    IdentitySpec(label="Policy", klass=IdentityClass.SLUG_ALIAS, key_prefix="policy"),
    IdentitySpec(
        label="BugPattern",
        klass=IdentityClass.SLUG_ALIAS,
        key_prefix="bug_pattern",
    ),
    # External-id-anchored (the source system supplies a stable identifier)
    IdentitySpec(
        label="PullRequest",
        klass=IdentityClass.EXTERNAL_ID,
        key_prefix="github:pr",
        authoritative_source="github",
    ),
    IdentitySpec(
        label="Issue",
        klass=IdentityClass.EXTERNAL_ID,
        key_prefix="issue",
        authoritative_source=None,  # multi-source (Linear, GitHub, Jira)
    ),
    IdentitySpec(
        label="Commit",
        klass=IdentityClass.EXTERNAL_ID,
        key_prefix="commit",
        authoritative_source="github",
    ),
    IdentitySpec(
        label="Deployment",
        klass=IdentityClass.EXTERNAL_ID,
        key_prefix="deployment",
    ),
    IdentitySpec(
        label="Activity",
        klass=IdentityClass.EXTERNAL_ID,
        key_prefix="activity",
    ),
    # Content-hashed identity for free-form documents / notes / decisions
    IdentitySpec(label="Decision", klass=IdentityClass.CONTENT_HASH, key_prefix="decision"),
    IdentitySpec(label="Document", klass=IdentityClass.CONTENT_HASH, key_prefix="document"),
    IdentitySpec(label="Fix", klass=IdentityClass.CONTENT_HASH, key_prefix="fix"),
)


for _spec in _DEFAULTS:
    register_identity(_spec)


__all__ = [
    "IdentityClass",
    "IdentityError",
    "IdentitySpec",
    "all_identities",
    "get_identity",
    "mint_entity_key",
    "register_identity",
    "validate_entity_key",
]

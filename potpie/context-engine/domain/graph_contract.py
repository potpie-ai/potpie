"""Versioned Graph V1.5 contract constants — one home for the contract.

Before V1.5 the contract names, versions, truth vocabulary, mutation ops, and
key conventions were scattered across the ontology, the readers, and the docs.
This module is the single source of truth that the catalog, the semantic
validator, the lowerer, and the CLI all read:

- :data:`GRAPH_CONTRACT_VERSION` / :data:`ONTOLOGY_VERSION`
- :class:`TruthClass` and its map onto the ranker's evidence-strength vocabulary
- :class:`SemanticMutationOp` partitioned into applicable / review-required /
  deferred (so the catalog stays honest about what V1.5 can actually do)
- :class:`MutationRisk`
- :class:`SourceAuthority` (the supported evidence authorities)
- canonical entity-key helpers (accept the docs' readable hyphen aliases,
  normalize to the wired underscore prefixes)
- the **edge identity key**, which folds in ``environment`` when present so an
  env-qualified edge never supersedes its counterpart in another environment
  (see the plan's Query Surface section)

Import discipline: this module is a near-leaf. It owns plain constants and
enums and imports **nothing** from :mod:`domain.ontology` at module load — the
key/predicate helpers import the ontology lazily inside their bodies — so
``domain.ontology`` can mirror :data:`ONTOLOGY_VERSION` from here without an
import cycle.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

# --- Versions ---------------------------------------------------------------

GRAPH_CONTRACT_VERSION = "v1.5"
ONTOLOGY_VERSION = "2026-06-graph"

# Contract versions a V1.5 daemon will accept on an inbound mutation payload.
SUPPORTED_GRAPH_CONTRACT_VERSIONS: frozenset[str] = frozenset({"v1.5"})


def is_supported_contract_version(version: str | None) -> bool:
    """True iff ``version`` is a contract version this build can apply.

    ``None`` / empty is treated as the current version (a convenience for
    hand-written single-op payloads); any explicit value must be in the
    supported set.
    """
    if version is None or not str(version).strip():
        return True
    return str(version).strip() in SUPPORTED_GRAPH_CONTRACT_VERSIONS


# --- Truth classes ----------------------------------------------------------


class TruthClass(StrEnum):
    """The V1.5 truth vocabulary stamped on every durable claim.

    Distinct from the ontology's ``source_of_truth`` (a per-entity SoT policy):
    ``truth`` is a per-*claim* assertion about how the fact is known. The
    ranker reads ``evidence_strength``; ``truth`` rides alongside as a
    first-class field and maps onto an evidence strength below.
    """

    authoritative_fact = "authoritative_fact"
    source_observation = "source_observation"
    agent_claim = "agent_claim"
    user_decision = "user_decision"
    preference = "preference"
    timeline_event = "timeline_event"
    quality_finding = "quality_finding"


TRUTH_CLASSES: tuple[str, ...] = tuple(t.value for t in TruthClass)
DEFAULT_TRUTH_CLASS: str = TruthClass.agent_claim.value


def is_truth_class(value: str | None) -> bool:
    return bool(value) and str(value) in TRUTH_CLASSES


# Map each truth class onto the ranker's evidence-strength vocabulary
# (``domain.ranking._STRENGTH_TO_SCORE``: deterministic/attested/stated/
# inferred/speculative) so a claim's truth tier drives its rank weight without
# the ranker needing to learn a second vocabulary.
TRUTH_TO_EVIDENCE_STRENGTH: dict[str, str] = {
    TruthClass.authoritative_fact: "deterministic",
    TruthClass.source_observation: "deterministic",
    TruthClass.user_decision: "attested",
    TruthClass.preference: "attested",
    TruthClass.timeline_event: "attested",
    TruthClass.agent_claim: "stated",
    TruthClass.quality_finding: "inferred",
}


def evidence_strength_for_truth(truth: str | None) -> str:
    """Evidence strength the ranker should use for a given truth class."""
    if truth and truth in TRUTH_TO_EVIDENCE_STRENGTH:
        return TRUTH_TO_EVIDENCE_STRENGTH[truth]
    return "stated"


# Truth classes that are inherently low-authority — a durable write carrying
# one of these does NOT require evidence (it is explicitly a soft claim, not a
# fact masquerading as one).
LOW_AUTHORITY_TRUTH_CLASSES: frozenset[str] = frozenset(
    {TruthClass.agent_claim.value, TruthClass.quality_finding.value}
)

# Truth classes that assert an *objective* external/code fact and therefore must
# be grounded in evidence. Subjective/attributed classes (preference,
# user_decision, timeline_event, agent_claim, quality_finding) carry their own
# authority — the agent/user stating them — so they need no evidence ref.
EVIDENCE_REQUIRED_TRUTH_CLASSES: frozenset[str] = frozenset(
    {TruthClass.authoritative_fact.value, TruthClass.source_observation.value}
)


# --- Mutation operations ----------------------------------------------------


class SemanticMutationOp(StrEnum):
    """The public, agent-facing semantic write vocabulary.

    Never raw graph CRUD — these are *semantic* ops the validator lowers to a
    :class:`~domain.reconciliation.MutationBatch`.
    """

    upsert_entity = "upsert_entity"
    link_entities = "link_entities"
    assert_claim = "assert_claim"
    append_event = "append_event"
    end_relation_validity = "end_relation_validity"
    retract_claim = "retract_claim"
    # Review-required in V1.5 (no plan store / identity resolver yet).
    supersede_claim = "supersede_claim"
    merge_duplicate_entities = "merge_duplicate_entities"
    # Deferred to V2: V1.5 models state changes as new claims/events.
    patch_entity = "patch_entity"
    transition_state = "transition_state"


# What V1.5 can actually apply directly (still subject to risk gating).
APPLICABLE_MUTATION_OPS: tuple[str, ...] = (
    SemanticMutationOp.append_event.value,
    SemanticMutationOp.upsert_entity.value,
    SemanticMutationOp.link_entities.value,
    SemanticMutationOp.assert_claim.value,
    SemanticMutationOp.end_relation_validity.value,
    SemanticMutationOp.retract_claim.value,
)

# Always returned as ``review_required``: no V1.5 approval path exists, so
# advertising them as applicable would lie.
REVIEW_REQUIRED_OPS: tuple[str, ...] = (
    SemanticMutationOp.supersede_claim.value,
    SemanticMutationOp.merge_duplicate_entities.value,
)

# Part of the V2 vocabulary but not modeled in V1.5 (state changes are new
# claims/events, not in-place edits). Surfaced in the catalog so the absence
# is honest, not silent.
DEFERRED_OPS: tuple[str, ...] = (
    SemanticMutationOp.patch_entity.value,
    SemanticMutationOp.transition_state.value,
)

KNOWN_MUTATION_OPS: frozenset[str] = frozenset(
    APPLICABLE_MUTATION_OPS + REVIEW_REQUIRED_OPS + DEFERRED_OPS
)


def is_known_op(op: str | None) -> bool:
    return bool(op) and str(op) in KNOWN_MUTATION_OPS


# --- Risk -------------------------------------------------------------------


class MutationRisk(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


# --- Source authority -------------------------------------------------------


class SourceAuthority(StrEnum):
    """How much an evidence reference can be trusted.

    The validator accepts only known authorities; the risk policy and the
    "durable writes need evidence" rule read them.
    """

    repository_metadata = "repository_metadata"  # manifests, CODEOWNERS, OpenAPI
    authoritative_code = "authoritative_code"  # the code itself
    external_system = "external_system"  # GitHub/Linear/etc. record
    ci_run = "ci_run"  # test / build output
    user_statement = "user_statement"  # a human said so
    agent_observation = "agent_observation"  # the agent saw it in-session


SOURCE_AUTHORITIES: frozenset[str] = frozenset(a.value for a in SourceAuthority)

# Authorities strong enough to satisfy the durable-write evidence rule on their
# own (deterministic / external ground truth or an explicit user statement).
STRONG_AUTHORITIES: frozenset[str] = frozenset(
    {
        SourceAuthority.repository_metadata.value,
        SourceAuthority.authoritative_code.value,
        SourceAuthority.external_system.value,
        SourceAuthority.ci_run.value,
        SourceAuthority.user_statement.value,
    }
)


def is_source_authority(value: str | None) -> bool:
    return bool(value) and str(value) in SOURCE_AUTHORITIES


# --- Entity-key helpers -----------------------------------------------------
# DECISION (V1.5 key-prefix canonicalization): the Graph V2 docs use readable
# hyphenated prefixes (``bug-pattern``, ``api-contract``); the wired ontology,
# the identity registry, and 100+ call sites use the underscore form
# (``bug_pattern``). The underscore form is **canonical** (it is what
# ``mint_entity_key`` produces and what every reader/test expects); the
# hyphenated form is accepted as an input alias and normalized here. This makes
# the code canonical and accepts the doc/legacy prefixes without a churny
# rewrite of the identity registry.


def normalize_key_prefix(prefix: str) -> str:
    """Normalize a key prefix to canonical underscore form (``bug-pattern`` → ``bug_pattern``)."""
    return prefix.strip().replace("-", "_")


def entity_key_prefix(key: str) -> str | None:
    """Return the (normalized) prefix of an entity key, or ``None`` if unprefixed."""
    k = (key or "").strip()
    if ":" not in k:
        return None
    return normalize_key_prefix(k.partition(":")[0])


def normalize_entity_key(key: str) -> str:
    """Normalize an entity key's prefix to canonical form, preserving the body.

    Only the text before the first colon is normalized; the body is left
    verbatim because bodies legitimately contain hyphens
    (``service:payments-api`` stays ``service:payments-api``).
    """
    k = (key or "").strip()
    if ":" not in k:
        return k
    prefix, _, rest = k.partition(":")
    return f"{normalize_key_prefix(prefix)}:{rest}"


def canonical_key_prefix(entity_type: str) -> str | None:
    """Canonical key prefix for an entity-type label, from the ontology."""
    from domain.ontology import ENTITY_TYPES

    spec = ENTITY_TYPES.get(entity_type)
    return spec.key_prefix if spec else None


def entity_key_matches_type(key: str, entity_type: str) -> bool:
    """True iff ``key``'s normalized prefix matches ``entity_type``'s canonical prefix."""
    expected = canonical_key_prefix(entity_type)
    if expected is None:
        return False
    return entity_key_prefix(key) == normalize_key_prefix(expected)


# --- Edge identity ----------------------------------------------------------


def edge_identity_key(
    subject_key: str,
    predicate: str,
    object_key: str,
    *,
    environment: str | None = None,
) -> tuple[str, ...]:
    """The identity tuple for an edge / claim.

    ``(subject, predicate, object)`` normally; ``(subject, predicate, object,
    environment)`` when an environment qualifier is present. The
    singleton / supersession key derives from this tuple, so an env-qualified
    edge never supersedes its counterpart in another environment.
    """
    base = (
        normalize_entity_key(subject_key),
        predicate.strip().upper(),
        normalize_entity_key(object_key),
    )
    env = (environment or "").strip().lower()
    return (*base, env) if env else base


# --- Deterministic claim key ------------------------------------------------


def _short_hash(text: str, *, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def make_claim_key(
    *,
    pot_id: str,
    subgraph: str,
    subject_key: str,
    predicate: str,
    object_component: str,
    discriminator: str | None = None,
    environment: str | None = None,
) -> str:
    """Deterministic claim identity for idempotent writes.

    Shape (plan Step 5)::

        claim:<pot>:<subgraph>:<subject>:<predicate>:<object-or-value-hash>:<src-or-idem-hash>

    ``object_component`` is an entity key (used verbatim, normalized) or a raw
    value (hashed when it does not look like a key, to keep keys bounded).
    ``environment``, when present, is folded into the object component so an
    env-qualified claim is distinct from its cross-environment counterpart.
    ``discriminator`` is the source ref or idempotency key.
    """
    obj = object_component.strip()
    env = (environment or "").strip().lower()
    if env:
        obj = f"{obj}@{env}"
    looks_like_key = ":" in obj and " " not in obj and len(obj) <= 200
    obj_token = normalize_entity_key(obj) if looks_like_key else _short_hash(obj)
    disc = (discriminator or "").strip()
    disc_token = _short_hash(disc) if disc else "_"
    return ":".join(
        [
            "claim",
            pot_id.strip(),
            subgraph.strip(),
            normalize_entity_key(subject_key),
            predicate.strip().upper(),
            obj_token,
            disc_token,
        ]
    )


__all__ = [
    "APPLICABLE_MUTATION_OPS",
    "DEFAULT_TRUTH_CLASS",
    "DEFERRED_OPS",
    "EVIDENCE_REQUIRED_TRUTH_CLASSES",
    "GRAPH_CONTRACT_VERSION",
    "KNOWN_MUTATION_OPS",
    "LOW_AUTHORITY_TRUTH_CLASSES",
    "ONTOLOGY_VERSION",
    "REVIEW_REQUIRED_OPS",
    "SOURCE_AUTHORITIES",
    "STRONG_AUTHORITIES",
    "SUPPORTED_GRAPH_CONTRACT_VERSIONS",
    "TRUTH_CLASSES",
    "TRUTH_TO_EVIDENCE_STRENGTH",
    "MutationRisk",
    "SemanticMutationOp",
    "SourceAuthority",
    "TruthClass",
    "canonical_key_prefix",
    "edge_identity_key",
    "entity_key_matches_type",
    "entity_key_prefix",
    "evidence_strength_for_truth",
    "is_known_op",
    "is_source_authority",
    "is_supported_contract_version",
    "is_truth_class",
    "make_claim_key",
    "normalize_entity_key",
    "normalize_key_prefix",
]

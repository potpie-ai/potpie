"""Public semantic-mutation DTOs (Graph V1.5 Step 3).

This is the agent-facing *write* contract — the tier above the internal,
backend-bound :class:`~domain.reconciliation.MutationBatch`. Agents never see
``EntityUpsert`` / ``EdgeUpsert`` / Cypher; they emit semantic operations
(``link_entities``, ``assert_claim``, ``append_event`` …) that the validator
risk-classifies and the lowerer turns into a ``MutationBatch``.

The DTOs parse from plain JSON without importing any adapter or backend, so a
mutation payload can be validated and reasoned about in the domain layer alone.

The canonical payload is **batch-shaped** (``{pot_id, operations: [...]}``). A
single-operation alias (``{operation: "...", ...}``) is accepted only at the
parse boundary and normalized into ``operations=[...]`` immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from potpie.context_engine.domain.graph_contract import (
    GRAPH_CONTRACT_VERSION,
    ONTOLOGY_VERSION,
)


class SemanticMutationParseError(ValueError):
    """Raised when a mutation payload is structurally malformed (not a
    semantic/ontology error — that is the validator's job)."""


# --- Building blocks --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphEntityRef:
    """A reference to a graph entity in a semantic mutation.

    ``description`` is the agent-authored *retrieval card* (Trigger Model / R2)
    — natural-language text written for search (symptoms, synonyms, scope),
    which the local embedder indexes on write.
    """

    key: str
    type: str | None = None
    name: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    summary: str | None = None
    description: str | None = None

    @classmethod
    def parse(cls, raw: Any) -> "GraphEntityRef | None":
        if raw is None:
            return None
        if isinstance(raw, str):
            return cls(key=raw)
        if not isinstance(raw, Mapping):
            raise SemanticMutationParseError(
                f"entity ref must be an object or string, got {type(raw).__name__}"
            )
        key = str(raw.get("key") or "").strip()
        if not key:
            raise SemanticMutationParseError("entity ref is missing 'key'")
        props = raw.get("properties") or {}
        if not isinstance(props, Mapping):
            raise SemanticMutationParseError("entity ref 'properties' must be an object")
        return cls(
            key=key,
            type=_opt_str(raw.get("type")),
            name=_opt_str(raw.get("name")),
            properties=dict(props),
            summary=_opt_str(raw.get("summary")),
            description=_opt_str(raw.get("description")),
        )


@dataclass(frozen=True, slots=True)
class GraphEvidenceRef:
    """One supporting evidence pointer for a durable claim."""

    source_ref: str
    authority: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any) -> "GraphEvidenceRef":
        if isinstance(raw, str):
            return cls(source_ref=raw)
        if not isinstance(raw, Mapping):
            raise SemanticMutationParseError(
                f"evidence must be an object or string, got {type(raw).__name__}"
            )
        source_ref = str(raw.get("source_ref") or raw.get("ref") or "").strip()
        if not source_ref:
            raise SemanticMutationParseError("evidence is missing 'source_ref'")
        meta = raw.get("metadata") or {}
        if not isinstance(meta, Mapping):
            raise SemanticMutationParseError("evidence 'metadata' must be an object")
        return cls(
            source_ref=source_ref,
            authority=_opt_str(raw.get("authority")),
            metadata=dict(meta),
        )


@dataclass(frozen=True, slots=True)
class MutationActor:
    """Who authored a mutation (surface + harness + optional user)."""

    surface: str | None = None
    harness: str | None = None
    user: str | None = None

    @classmethod
    def parse(cls, raw: Any) -> "MutationActor":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise SemanticMutationParseError("created_by must be an object")
        return cls(
            surface=_opt_str(raw.get("surface")),
            harness=_opt_str(raw.get("harness")),
            user=_opt_str(raw.get("user")),
        )


# --- The operation ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticMutation:
    """One semantic operation in a batch.

    Not every field applies to every op — the validator enforces per-op
    requirements. ``raw`` keeps the original op dict for diagnostics.
    """

    op: str
    subgraph: str | None = None
    subject: GraphEntityRef | None = None
    predicate: str | None = None
    object: GraphEntityRef | None = None
    value: Any | None = None
    """Value object for ``assert_claim`` against a literal (no entity)."""

    truth: str | None = None
    confidence: float | None = None
    evidence: tuple[GraphEvidenceRef, ...] = ()
    description: str | None = None
    environment: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    observed_at: str | None = None

    # end_relation_validity / retract_claim / supersede_claim
    reason: str | None = None
    superseded_by: GraphEntityRef | None = None

    # patch_entity / transition_state
    patch: Mapping[str, Any] = field(default_factory=dict)
    expected_entity_version: str | None = None
    from_state: str | None = None
    to_state: str | None = None

    # merge_duplicate_entities
    external_ids: Mapping[str, Any] = field(default_factory=dict)

    # append_event
    verb: str | None = None
    occurred_at: str | None = None
    actor: GraphEntityRef | None = None
    targets: tuple[GraphEntityRef, ...] = ()
    mentions: tuple[GraphEntityRef, ...] = ()

    extra: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any) -> "SemanticMutation":
        if not isinstance(raw, Mapping):
            raise SemanticMutationParseError(
                f"operation must be an object, got {type(raw).__name__}"
            )
        op = str(raw.get("op") or raw.get("operation") or "").strip()
        if not op:
            raise SemanticMutationParseError("operation is missing 'op'")
        confidence = raw.get("confidence")
        if confidence is not None and not isinstance(confidence, (int, float)):
            raise SemanticMutationParseError("'confidence' must be a number")
        evidence = tuple(
            GraphEvidenceRef.parse(e) for e in _as_list(raw.get("evidence"))
        )
        patch = _parse_mapping(
            raw.get("patch")
            if "patch" in raw
            else raw.get("properties")
            if op == "patch_entity" and "properties" in raw
            else raw.get("changes")
            if "changes" in raw
            else None,
            field_name="patch",
        )
        external_ids = _parse_mapping(
            raw.get("external_ids")
            if "external_ids" in raw
            else raw.get("identity_records")
            if "identity_records" in raw
            else None,
            field_name="external_ids",
        )
        return cls(
            op=op,
            subgraph=_opt_str(raw.get("subgraph")),
            subject=GraphEntityRef.parse(raw.get("subject")),
            predicate=_opt_str(raw.get("predicate")),
            object=GraphEntityRef.parse(raw.get("object")),
            value=raw.get("value"),
            truth=_opt_str(raw.get("truth")),
            confidence=float(confidence) if confidence is not None else None,
            evidence=evidence,
            description=_opt_str(raw.get("description")),
            environment=_opt_str(raw.get("environment")),
            valid_from=_opt_str(raw.get("valid_from") or raw.get("valid_at")),
            valid_until=_opt_str(raw.get("valid_until")),
            observed_at=_opt_str(raw.get("observed_at")),
            reason=_opt_str(raw.get("reason")),
            superseded_by=GraphEntityRef.parse(raw.get("superseded_by")),
            patch=patch,
            expected_entity_version=_opt_str(
                raw.get("expected_entity_version")
                or raw.get("expected_version")
                or raw.get("entity_version")
            ),
            from_state=_opt_str(raw.get("from_state") or raw.get("expected_state")),
            to_state=_opt_str(
                raw.get("to_state") or raw.get("state") or raw.get("lifecycle_state")
            ),
            external_ids=external_ids,
            verb=_opt_str(raw.get("verb") or raw.get("verb_class")),
            occurred_at=_opt_str(raw.get("occurred_at")),
            actor=GraphEntityRef.parse(raw.get("actor")),
            targets=_parse_entity_refs(raw.get("targets")),
            mentions=_parse_entity_refs(raw.get("mentions")),
            extra=dict(raw.get("extra") or {}),
            raw=dict(raw),
        )


# --- The request ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticMutationRequest:
    """A parsed, batch-shaped semantic mutation request."""

    pot_id: str
    operations: tuple[SemanticMutation, ...]
    graph_contract_version: str = GRAPH_CONTRACT_VERSION
    idempotency_key: str | None = None
    created_by: MutationActor = field(default_factory=MutationActor)
    dry_run: bool = False
    allow_review_required: bool = False
    approved_by: str | None = None

    @classmethod
    def parse(
        cls,
        payload: Any,
        *,
        pot_id: str | None = None,
        dry_run: bool = False,
        allow_review_required: bool = False,
        approved_by: str | None = None,
    ) -> "SemanticMutationRequest":
        """Parse a JSON payload (batch-shaped or single-op alias) into a request.

        ``pot_id`` (CLI ``--pot``) overrides the payload's ``pot_id`` when given,
        so the active pot wins over a stale value baked into a file.
        """
        if not isinstance(payload, Mapping):
            raise SemanticMutationParseError(
                f"mutation payload must be an object, got {type(payload).__name__}"
            )

        ops_raw = _normalize_operations(payload)
        operations = tuple(SemanticMutation.parse(op) for op in ops_raw)

        resolved_pot = (pot_id or _opt_str(payload.get("pot_id")) or "").strip()
        if not resolved_pot:
            raise SemanticMutationParseError("mutation payload is missing 'pot_id'")

        return cls(
            pot_id=resolved_pot,
            operations=operations,
            graph_contract_version=_opt_str(payload.get("graph_contract_version"))
            or GRAPH_CONTRACT_VERSION,
            idempotency_key=_opt_str(payload.get("idempotency_key")),
            created_by=MutationActor.parse(payload.get("created_by")),
            dry_run=dry_run,
            allow_review_required=allow_review_required,
            approved_by=approved_by,
        )


# --- Validation issues ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticMutationValidationIssue:
    """One validation finding against a semantic mutation."""

    code: str
    message: str
    severity: str = "error"  # "error" | "warning"
    op_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "op_index": self.op_index,
        }

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


# --- The plan (validated + risk-classified, ready to lower) -----------------


@dataclass(frozen=True, slots=True)
class LoweredOperation:
    """One accepted operation paired with the claim keys it will produce."""

    op_index: int
    op: str
    risk: str
    claim_keys: tuple[str, ...] = ()
    subgraph: str | None = None
    truth: str | None = None
    status: str = "accepted"  # "accepted" | "review_required" | "deferred" | "rejected"


@dataclass(slots=True)
class SemanticMutationPlan:
    """Outcome of validation + risk classification + lowering.

    Carries the internal :class:`MutationBatch` (only when the plan is
    applicable), the overall risk, the per-op outcomes, and structured issues.
    """

    pot_id: str
    ok: bool
    risk: str
    decision: str = "rejected"
    """One of ``apply`` | ``review_required`` | ``rejected``."""
    issues: tuple[SemanticMutationValidationIssue, ...] = ()
    accepted_ops: tuple[LoweredOperation, ...] = ()
    review_required_ops: tuple[LoweredOperation, ...] = ()
    deferred_ops: tuple[LoweredOperation, ...] = ()
    # The lowered internal batch, set only when there is at least one applicable
    # op. Typed as Any to avoid importing the write tier into this DTO module.
    batch: Any = None
    provenance: Any = None
    warnings: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[SemanticMutationValidationIssue, ...]:
        return tuple(i for i in self.issues if i.is_error)


@dataclass(frozen=True, slots=True)
class SemanticMutationResult:
    """The response returned to the harness from ``graph mutate`` / ``record``."""

    ok: bool
    status: str  # "applied" | "validated" | "rejected" | "review_required" | "error"
    risk: str
    pot_id: str
    auto_committed: bool = False
    would_apply: bool | None = None  # set on dry-run
    mutation_id: str | None = None
    operations_accepted: int = 0
    operations_applied: int = 0
    mutations_applied: Mapping[str, int] = field(default_factory=dict)
    preview: Mapping[str, int] | None = None
    claim_keys: tuple[str, ...] = ()
    subgraphs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    issues: tuple[SemanticMutationValidationIssue, ...] = ()
    detail: str | None = None
    graph_contract_version: str = GRAPH_CONTRACT_VERSION
    ontology_version: str = ONTOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "risk": self.risk,
            "pot_id": self.pot_id,
            "graph_contract_version": self.graph_contract_version,
            "ontology_version": self.ontology_version,
            "operations_accepted": self.operations_accepted,
            "warnings": list(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }
        if self.would_apply is not None:
            out["would_apply"] = self.would_apply
            if self.preview is not None:
                out["preview"] = dict(self.preview)
        else:
            out["auto_committed"] = self.auto_committed
            out["operations_applied"] = self.operations_applied
            out["mutations_applied"] = dict(self.mutations_applied)
            if self.mutation_id:
                out["mutation_id"] = self.mutation_id
        if self.claim_keys:
            out["claim_keys"] = list(self.claim_keys)
        if self.subgraphs:
            out["subgraphs"] = list(self.subgraphs)
        if self.detail:
            out["detail"] = self.detail
        return out


# --- helpers ----------------------------------------------------------------


def _normalize_operations(payload: Mapping[str, Any]) -> list[Any]:
    """Normalize batch-shaped or single-op-alias payloads into a list of op dicts."""
    if "operations" in payload and payload["operations"] is not None:
        ops = payload["operations"]
        if not isinstance(ops, (list, tuple)):
            raise SemanticMutationParseError("'operations' must be a list")
        if not ops:
            raise SemanticMutationParseError("'operations' must not be empty")
        return list(ops)

    # Single-op alias: an ``operation`` (or top-level ``op``) names the op and
    # the remaining keys are op fields. Strip request-level keys.
    op_name = payload.get("operation") or payload.get("op")
    if op_name:
        request_keys = {
            "graph_contract_version",
            "pot_id",
            "idempotency_key",
            "created_by",
            "operation",
        }
        op_dict = {k: v for k, v in payload.items() if k not in request_keys}
        op_dict["op"] = op_name
        return [op_dict]

    raise SemanticMutationParseError(
        "mutation payload must contain 'operations' (batch) or 'operation' (single)"
    )


def _parse_entity_refs(value: Any) -> tuple[GraphEntityRef, ...]:
    out: list[GraphEntityRef] = []
    for item in _as_list(value):
        ref = GraphEntityRef.parse(item)
        if ref is not None:
            out.append(ref)
    return tuple(out)


def _parse_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SemanticMutationParseError(f"'{field_name}' must be an object")
    return dict(value)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


__all__ = [
    "GraphEntityRef",
    "GraphEvidenceRef",
    "LoweredOperation",
    "MutationActor",
    "SemanticMutation",
    "SemanticMutationParseError",
    "SemanticMutationPlan",
    "SemanticMutationRequest",
    "SemanticMutationResult",
    "SemanticMutationValidationIssue",
]

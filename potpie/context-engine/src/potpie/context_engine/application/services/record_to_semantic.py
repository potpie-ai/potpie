"""Convert a V1 ``context_record`` into a V1.5 semantic mutation (Step 8).

The V1 compatibility write (``context_record``) must use the *same* path as
``graph mutate`` — no private direct-lowering. This module is the bridge: it maps
each ``record_type`` onto a single semantic operation (op + predicate + truth +
subgraph) per the Step 8 table. ``DefaultGraphService.record`` calls this and
then ``self.mutate`` so the record flows through validation, risk classification,
and the one lowerer.

Compatibility: unknown ``record_type`` values still fall back to a free-form
claim. Known structured record types are always validated, so CLI/MCP writes
reject the same malformed payloads as the managed HTTP path instead of silently
downgrading into generic summary-derived claims.

Because a ``context_record`` *is* a deliberate write action by the agent, the
request is marked approved so medium-risk record types (decisions) auto-apply
rather than dead-ending on the approval gate. Review-required *ops* (supersede /
merge) are never produced here.
"""

from __future__ import annotations

from typing import Any, Mapping

from potpie.context_engine.domain.context_records import (
    validate_record_payload,
)
from potpie.context_engine.domain.identity import _slugify  # deterministic slug; reused for keys
from potpie.context_engine.domain.ontology import ENTITY_TYPES, record_type_spec
from potpie.context_engine.domain.ports.agent_context import RecordRequest
from potpie.context_engine.domain.semantic_mutations import (
    MutationActor,
    SemanticMutation,
    SemanticMutationRequest,
)

_APPROVED_BY = "context_record"
_PREFIX_TO_LABEL: dict[str, str] = {
    spec.key_prefix: label for label, spec in ENTITY_TYPES.items()
}


def record_to_semantic_request(
    request: RecordRequest, *, record_type: str, source_id: str
) -> SemanticMutationRequest:
    """Build the semantic mutation request for a durable record."""
    details = dict(request.details)
    validate_record_payload(
        record_type=record_type, summary=request.summary, details=details
    )

    raw_ops = _build_operation(
        record_type=record_type,
        request=request,
        details=details,
        source_id=source_id,
    )
    op_payloads = raw_ops if isinstance(raw_ops, list) else [raw_ops]
    actor = MutationActor(
        surface=str(request.metadata.get("surface") or "agent"),
        harness=_opt(request.metadata.get("harness")),
        user=_opt(request.metadata.get("user")),
    )
    return SemanticMutationRequest(
        pot_id=request.pot_id,
        operations=tuple(SemanticMutation.parse(op) for op in op_payloads),
        idempotency_key=request.idempotency_key or source_id,
        created_by=actor,
        allow_review_required=True,
        approved_by=_APPROVED_BY,
    )


# ---------------------------------------------------------------------------
# Per-record-type mapping (Step 8 table) — details-driven with safe defaults
# ---------------------------------------------------------------------------


def _build_operation(
    *,
    record_type: str,
    request: RecordRequest,
    details: Mapping[str, Any],
    source_id: str,
) -> dict:
    scope = dict(request.scope)
    summary = request.summary or ""
    evidence_payload = [{"source_ref": ref} for ref in request.source_refs]
    target = _scope_target(scope, request.pot_id)
    code_scope = _code_scope(scope)
    spec = record_type_spec(record_type)
    anchor = spec.anchor_label if spec else None

    base = {
        "evidence": evidence_payload,
        "extra": code_scope,
    }

    if record_type in ("preference", "policy"):
        prefix = (
            "policy"
            if (anchor == "Policy" or record_type == "policy")
            else "preference"
        )
        prescription = _str(details.get("prescription")) or summary
        return {
            **base,
            "op": "assert_claim",
            "subgraph": "decisions",
            "predicate": "POLICY_APPLIES_TO",
            "truth": "preference",
            "subject": {
                "key": f"{prefix}:{_slug(prescription or summary or source_id)}",
                "type": anchor or "Preference",
                "description": summary,
                "properties": _drop_empty(
                    {
                        "policy_kind": _str(details.get("policy_kind")) or "general",
                        "prescription": prescription,
                        "strength": _str(details.get("strength")) or "soft",
                        "audience": _str(details.get("audience")) or "team",
                        **code_scope,
                        **_as_str_map(details.get("code_scope")),
                    }
                ),
            },
            "object": target,
            "description": summary or prescription,
        }

    if record_type == "bug_pattern":
        symptom = _str(details.get("symptom_signature")) or summary
        return {
            **base,
            "op": "assert_claim",
            "subgraph": "debugging",
            "predicate": "REPRODUCES",
            "truth": "agent_claim",
            "subject": {
                "key": f"bug_pattern:{_slug(symptom or source_id)}",
                "type": "BugPattern",
                "description": summary,
                "properties": _drop_empty(
                    {
                        "symptom_signature": symptom,
                        "kind": _str(details.get("kind")),
                        **code_scope,
                    }
                ),
            },
            "object": target,
            "description": " • ".join(p for p in (summary, symptom) if p),
        }

    if record_type == "fix":
        symptom = _str(details.get("symptom_signature")) or summary
        bug = {
            "key": f"bug_pattern:{_slug(symptom or source_id)}",
            "type": "BugPattern",
            "description": symptom,
        }
        verification_status = _str(details.get("verification_status")) or "unverified"
        main_predicate = (
            "ATTEMPTED_FIX_FAILED"
            if verification_status.lower()
            in {"failed", "didnt_work", "did_not_work", "didn't_work"}
            else "RESOLVED"
        )
        ops = [
            {
                **base,
                "op": "assert_claim",
                "subgraph": "debugging",
                "predicate": "REPRODUCES",
                "truth": "agent_claim",
                "subject": bug,
                "object": target,
                "description": " • ".join(p for p in (summary, symptom) if p),
            },
            {
                **base,
                "op": "assert_claim",
                "subgraph": "debugging",
                "predicate": main_predicate,
                "truth": "agent_claim",
                "subject": {
                    "key": f"fix:{_slug(summary or source_id)}",
                    "type": "Fix",
                    "description": summary,
                    "properties": _drop_empty(
                        {
                            "fix_steps": _as_list(details.get("fix_steps")),
                            "verification_status": verification_status,
                            "root_cause": _str(details.get("root_cause")),
                            **code_scope,
                        }
                    ),
                },
                # RESOLVED / ATTEMPTED_FIX_FAILED are Fix -> BugPattern; derive
                # the bug from the symptom.
                "object": bug,
                "description": " • ".join(p for p in (summary, symptom) if p),
            }
        ]
        for attempt in _as_list(details.get("attempted_failed_fixes")):
            ops.append(
                {
                    **base,
                    "op": "assert_claim",
                    "subgraph": "debugging",
                    "predicate": "ATTEMPTED_FIX_FAILED",
                    "truth": "agent_claim",
                    "subject": {
                        "key": f"fix:{_slug(attempt)}",
                        "type": "Fix",
                        "description": attempt,
                        "properties": _drop_empty(
                            {
                                "fix_steps": [attempt],
                                "verification_status": "failed",
                                "resolution_status": "failed",
                                **code_scope,
                            }
                        ),
                    },
                    "object": bug,
                    "description": " • ".join(
                        p for p in (attempt, f"failed attempt for {symptom}") if p
                    ),
                }
            )
        return ops

    if record_type == "verification":
        target_ref = _str(details.get("target_ref")) or source_id
        return {
            **base,
            "op": "assert_claim",
            "subgraph": "debugging",
            "predicate": "VERIFIED",
            "truth": "timeline_event",
            "subject": {
                "key": f"activity:verification:{_slug(target_ref)}",
                "type": "Activity",
                "description": summary,
                "properties": _drop_empty(
                    {
                        "verb_class": "verified",
                        "outcome": _str(details.get("outcome")),
                    }
                ),
            },
            "object": {"key": _fix_ref(target_ref), "type": "Fix"},
            "description": summary or f"verification: {details.get('outcome')}",
        }

    if record_type == "decision":
        title = _str(details.get("title")) or summary
        decision = {
            "key": f"decision:{_slug(title or source_id)}",
            "type": "Decision",
            "description": summary,
            "properties": _drop_empty(
                {
                    "title": title,
                    "rationale": _str(details.get("rationale")),
                    "alternatives_rejected": _as_list(
                        details.get("alternatives_rejected")
                    ),
                }
            ),
        }
        ops = [
            {
                **base,
                "op": "assert_claim",
                "subgraph": "decisions",
                "predicate": "DECIDED",
                "truth": "user_decision",
                "subject": decision,
                "object": target,
                "description": " • ".join(p for p in (summary, title) if p),
            }
        ]
        for affected in _as_list(details.get("affects_refs")):
            ops.append(
                {
                    **base,
                    "op": "assert_claim",
                    "subgraph": "decisions",
                    "predicate": "AFFECTS",
                    "truth": "user_decision",
                    "subject": decision,
                    "object": _entity_ref_for_key(affected),
                    "description": " • ".join(
                        p for p in (summary, f"affects {affected}") if p
                    ),
                }
            )
        return ops

    # Free-form record (no structured schema): a generic association from a
    # Document/Observation anchor to the scope target. RELATED_TO accepts any
    # endpoints, so it always lands and surfaces via raw_graph.
    anchor_label = anchor or "Document"
    prefix = "observation" if anchor_label == "Observation" else "document"
    return {
        **base,
        "op": "assert_claim",
        "subgraph": "admin",
        "predicate": "RELATED_TO",
        "truth": "agent_claim",
        "subject": {
            "key": f"{prefix}:{_slug(summary or source_id)}",
            "type": anchor_label,
            "description": summary,
            "properties": {"record_type": record_type},
        },
        "object": target,
        "description": summary,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope_target(scope: dict, pot_id: str) -> dict:
    service = scope.get("service")
    if isinstance(service, str) and service.strip():
        return {"key": f"service:{_slug(service)}", "type": "Service"}
    repo = scope.get("repo") or scope.get("repo_name")
    if isinstance(repo, str) and repo.strip():
        return {"key": f"repo:{_slug(repo)}", "type": "Repository"}
    return {"key": f"repo:{_slug(pot_id)}", "type": "Repository"}


def _code_scope(scope: dict) -> dict[str, str]:
    keys = (
        "language",
        "framework",
        "repo",
        "service",
        "file_path",
        "audience",
        "environment",
    )
    out: dict[str, str] = {}
    for key in keys:
        val = scope.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip()
    return out


def _slug(text: str) -> str:
    try:
        return _slugify(text or "untitled")
    except Exception:
        import re

        s = re.sub(r"[^a-z0-9]+", "-", (text or "untitled").lower()).strip("-")
        return s or "untitled"


def _fix_ref(target_ref: str) -> str:
    ref = (target_ref or "").strip()
    return ref if ref.startswith("fix:") else f"fix:{_slug(ref)}"


def _entity_ref_for_key(raw: str) -> dict[str, str]:
    value = raw.strip()
    if ":" in value:
        prefix = value.partition(":")[0]
        label = _PREFIX_TO_LABEL.get(prefix)
        if label:
            return {"key": value, "type": label}
    return {"key": f"observation:{_slug(value)}", "type": "Observation"}


def _str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return []


def _as_str_map(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {
            str(k): str(v) for k, v in value.items() if isinstance(v, str) and v.strip()
        }
    return {}


def _drop_empty(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def _opt(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


__all__ = ["record_to_semantic_request"]

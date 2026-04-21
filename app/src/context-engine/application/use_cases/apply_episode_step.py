"""Apply one durable episode step (pot-scoped ordering)."""

from __future__ import annotations

from datetime import datetime, timezone

from domain.errors import ReconciliationApplyError, ReconciliationPlanValidationError
from domain.ingestion_kinds import (
    EPISODE_STEP_APPLIED,
    EPISODE_STEP_APPLYING,
    EPISODE_STEP_FAILED,
    STEP_KIND_AGENT_PLAN_SLICE,
    STEP_KIND_RAW_EPISODE,
)
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.graph_mutation_applier import GraphMutationApplierPort
from domain.ports.reconciliation_ledger import ReconciliationLedgerPort
from domain.ports.structural_graph import StructuralGraphPort
from domain.reconciliation import EpisodeDraft, MutationSummary, ReconciliationResult
from domain.reconciliation_issues import validation_line_to_issue

from adapters.outbound.context_graph_writer_adapter import DefaultContextGraphWriter
from application.use_cases.reconciliation_plan_codec import reconciliation_plan_from_dict
from application.use_cases.reconciliation_validation import validate_reconciliation_plan


def apply_episode_step_for_event(
    episodic: EpisodicGraphPort,
    structural: StructuralGraphPort,
    reco_ledger: ReconciliationLedgerPort,
    event_id: str,
    sequence: int,
    *,
    mutation_applier: GraphMutationApplierPort | None = None,
) -> ReconciliationResult:
    """Apply a single step if prior sequences for this event are already applied."""
    row = reco_ledger.get_event_by_id(event_id)
    if row is None:
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error="unknown_event",
        )
    step = reco_ledger.get_episode_step(event_id, sequence)
    if step is None:
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error="unknown_episode_step",
        )
    if step.status == EPISODE_STEP_APPLIED:
        return ReconciliationResult(ok=True, episode_uuids=[], mutation_summary=MutationSummary(), error=None)

    prev = reco_ledger.max_applied_sequence(event_id)
    expected_next = (prev or 0) + 1
    if sequence != expected_next:
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error=f"out_of_order: expected sequence {expected_next}, got {sequence}",
        )

    reco_ledger.update_episode_step_status(
        event_id,
        sequence,
        status=EPISODE_STEP_APPLYING,
        increment_attempt=True,
    )

    writer = DefaultContextGraphWriter(episodic, structural, mutation_applier)

    try:
        if step.step_kind == STEP_KIND_RAW_EPISODE:
            data = step.step_json
            ref = data.get("reference_time")
            if isinstance(ref, str):
                rt = datetime.fromisoformat(ref.replace("Z", "+00:00"))
            elif isinstance(ref, datetime):
                rt = ref
            else:
                rt = datetime.now(timezone.utc)
            draft = EpisodeDraft(
                name=str(data["name"]),
                episode_body=str(data["episode_body"]),
                source_description=str(data["source_description"]),
                reference_time=rt,
            )
            out = writer.write_raw_episode(
                row.pot_id,
                draft.name,
                draft.episode_body,
                draft.source_description,
                draft.reference_time,
            )
            uuid = out.get("episode_uuid")
            if not uuid:
                reco_ledger.update_episode_step_status(
                    event_id,
                    sequence,
                    status=EPISODE_STEP_FAILED,
                    error="episode_write_failed",
                )
                reco_ledger.record_event_failed(event_id, "episode_write_failed")
                return ReconciliationResult(
                    ok=False,
                    episode_uuids=[],
                    mutation_summary=MutationSummary(),
                    error="episode_write_failed",
                )
            reco_ledger.update_episode_step_status(event_id, sequence, status=EPISODE_STEP_APPLIED)
            _maybe_finish_event(reco_ledger, event_id)
            return ReconciliationResult(
                ok=True,
                episode_uuids=[uuid],
                mutation_summary=MutationSummary(episodes_written=1),
                error=None,
            )

        if step.step_kind != STEP_KIND_AGENT_PLAN_SLICE:
            raise ReconciliationApplyError(f"unsupported step_kind {step.step_kind}")

        plan = reconciliation_plan_from_dict(step.step_json)
        validate_reconciliation_plan(plan, row.pot_id)
        result = writer.apply_plan(plan, expected_pot_id=row.pot_id)
        if not result.ok:
            reco_ledger.update_episode_step_status(
                event_id,
                sequence,
                status=EPISODE_STEP_FAILED,
                error=result.error or "apply_failed",
            )
            reco_ledger.record_event_failed(event_id, result.error or "apply_failed")
            return result

        reco_ledger.update_episode_step_status(event_id, sequence, status=EPISODE_STEP_APPLIED)
        _maybe_finish_event(reco_ledger, event_id)
        return result
    except ReconciliationPlanValidationError as exc:
        reco_ledger.update_episode_step_status(
            event_id,
            sequence,
            status=EPISODE_STEP_FAILED,
            error=str(exc),
        )
        reco_ledger.record_event_failed(event_id, str(exc))
        if exc.structured_issues:
            errs = [dict(x) for x in exc.structured_issues]
        else:
            errs = [validation_line_to_issue(str(exc))]
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error=str(exc),
            reconciliation_errors=errs,
        )
    except ReconciliationApplyError as exc:
        reco_ledger.update_episode_step_status(
            event_id,
            sequence,
            status=EPISODE_STEP_FAILED,
            error=str(exc),
        )
        reco_ledger.record_event_failed(event_id, str(exc))
        return ReconciliationResult(
            ok=False,
            episode_uuids=[],
            mutation_summary=MutationSummary(),
            error=str(exc),
            reconciliation_errors=[validation_line_to_issue(str(exc))],
        )
    except Exception as exc:
        reco_ledger.update_episode_step_status(
            event_id,
            sequence,
            status=EPISODE_STEP_FAILED,
            error=str(exc),
        )
        reco_ledger.record_event_failed(event_id, str(exc))
        raise


def _maybe_finish_event(reco_ledger: ReconciliationLedgerPort, event_id: str) -> None:
    steps = reco_ledger.list_episode_steps(event_id)
    if steps and all(s.status == EPISODE_STEP_APPLIED for s in steps):
        reco_ledger.record_event_reconciled(event_id)

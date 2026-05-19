"""hypothesis_state_tool — conversation-scoped hypothesis persistence for the debug agent.

Provides four tools that let the debug agent persist and update hypothesis state
across a multi-turn debugging conversation:

  record_hypothesis         — create a new hypothesis record
  update_hypothesis_status  — change the status of an existing hypothesis
  append_hypothesis_evidence — add an observation to an existing hypothesis
  list_hypotheses           — retrieve all hypothesis records for the session

Persistence mirrors the todos family (todo_management_tool.py):
  - In-memory storage on a ContextVar, providing per-execution-context isolation.
  - No Redis, no DB — same lightweight, zero-dependency approach as todo/requirement tools.
  - conversation_id is NOT exposed in tool args; isolation is provided by ContextVar
    (each async agent-run context is a separate execution context).

The HypothesisStatus enum is imported from debug_hypothesis_contract.py so the
tool layer and the markdown-emission layer share exactly the same lifecycle states.
"""

from contextvars import ContextVar
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from app.modules.intelligence.agents.chat_agents.system_agents.debug_hypothesis_contract import (
    HypothesisStatus,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HypothesisRecord(BaseModel):
    """A single persisted debugging hypothesis."""

    id: str = Field(..., description="Server-assigned hypothesis id (e.g. 'hyp_1').")
    title: str = Field(..., description="Short hypothesis title; matches the markdown title the agent emits.")
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence: List[str] = Field(default_factory=list)
    validation_plan: List[str] = Field(default_factory=list)
    created_at: datetime = Field(..., description="UTC timestamp when this hypothesis was recorded.")
    updated_at: datetime = Field(..., description="UTC timestamp of last status or evidence change.")
    conversation_id: str = ""  # stamped from HypothesisStore.conversation_id at create time; production binding happens in init_managers


class RecordHypothesisInput(BaseModel):
    title: str = Field(description="Short hypothesis title matching the markdown heading")
    status: HypothesisStatus = Field(
        default=HypothesisStatus.PROPOSED,
        description="Initial lifecycle status. Defaults to 'proposed'.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Initial evidence observations (can be added to later with append_hypothesis_evidence).",
    )
    validation_plan: List[str] = Field(
        default_factory=list,
        description="Validation steps to confirm or refute this hypothesis.",
    )


class UpdateHypothesisStatusInput(BaseModel):
    hypothesis_id: str = Field(description="ID of the hypothesis to update (e.g. 'hyp_1')")
    status: HypothesisStatus = Field(description="New lifecycle status")


class AppendHypothesisEvidenceInput(BaseModel):
    hypothesis_id: str = Field(description="ID of the hypothesis to update (e.g. 'hyp_1')")
    evidence: str = Field(description="Single observation to append to the evidence list")


class ListHypothesesInput(BaseModel):
    pass  # No args — scoped by execution context


# ---------------------------------------------------------------------------
# In-memory storage — mirrors TodoStorage / RequirementManager pattern
# ---------------------------------------------------------------------------


class HypothesisStore:
    """Holds all hypothesis records for a single execution context (agent run)."""

    def __init__(self, conversation_id: str = "") -> None:
        self._records: List[HypothesisRecord] = []
        self._next_index: int = 1
        self.conversation_id: str = conversation_id

    # ---- helpers ----

    def _next_id(self) -> str:
        id_ = f"hyp_{self._next_index}"
        self._next_index += 1
        return id_

    def _find(self, hypothesis_id: str) -> Optional[HypothesisRecord]:
        for rec in self._records:
            if rec.id == hypothesis_id:
                return rec
        return None

    # ---- operations ----

    def add(
        self,
        title: str,
        status: HypothesisStatus,
        evidence: List[str],
        validation_plan: List[str],
    ) -> HypothesisRecord:
        now = datetime.now(tz=timezone.utc)
        rec = HypothesisRecord(
            id=self._next_id(),
            title=title,
            status=status,
            evidence=list(evidence),
            validation_plan=list(validation_plan),
            created_at=now,
            updated_at=now,
            conversation_id=self.conversation_id,
        )
        self._records.append(rec)
        logger.debug("hypothesis_state: recorded id={} title={!r}", rec.id, rec.title)
        return rec

    def update_status(
        self, hypothesis_id: str, status: HypothesisStatus
    ) -> HypothesisRecord | str:
        rec = self._find(hypothesis_id)
        if rec is None:
            return f"Hypothesis with ID '{hypothesis_id}' not found"
        rec.status = status
        rec.updated_at = datetime.now(tz=timezone.utc)
        logger.debug(
            "hypothesis_state: updated status id={} status={}", rec.id, status.value
        )
        return rec

    def append_evidence(self, hypothesis_id: str, evidence: str) -> HypothesisRecord | str:
        rec = self._find(hypothesis_id)
        if rec is None:
            return f"Hypothesis with ID '{hypothesis_id}' not found"
        rec.evidence.append(evidence)
        rec.updated_at = datetime.now(tz=timezone.utc)
        logger.debug(
            "hypothesis_state: appended evidence id={} total_entries={}", rec.id, len(rec.evidence)
        )
        return rec

    def list_all(self) -> List[HypothesisRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._next_index = 1


# Context variable — one HypothesisStore per async execution context (agent run).
# Mirrors _todo_storage_ctx in todo_management_tool.py.
_hypothesis_store_ctx: ContextVar[Optional[HypothesisStore]] = ContextVar(
    "_hypothesis_store_ctx", default=None
)


def get_hypothesis_store() -> HypothesisStore:
    """Return the HypothesisStore for the current execution context, creating one if needed."""
    store = _hypothesis_store_ctx.get()
    if store is None:
        store = HypothesisStore()
        _hypothesis_store_ctx.set(store)
    return store


def _reset_hypothesis_store(conversation_id: str = "") -> None:
    """Reset the hypothesis store for a new conversation/agent run."""
    _hypothesis_store_ctx.set(HypothesisStore(conversation_id=conversation_id))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def record_hypothesis(
    title: str,
    status: HypothesisStatus = HypothesisStatus.PROPOSED,
    evidence: Optional[List[str]] = None,
    validation_plan: Optional[List[str]] = None,
) -> dict:
    """Record a new debugging hypothesis.

    Returns the full HypothesisRecord including the assigned id and timestamps.
    """
    rec = get_hypothesis_store().add(
        title=title,
        status=status,
        evidence=evidence or [],
        validation_plan=validation_plan or [],
    )
    return rec.model_dump(mode="json")


def update_hypothesis_status(hypothesis_id: str, status: HypothesisStatus) -> dict:
    """Update the status of an existing hypothesis, bumping updated_at.

    Returns the updated HypothesisRecord, or an error string if the id is not found.
    """
    result = get_hypothesis_store().update_status(hypothesis_id, status)
    if isinstance(result, str):
        # Error message — mirror how todo_management_tool returns plain strings for errors
        return {"error": result}
    return result.model_dump(mode="json")


def append_hypothesis_evidence(hypothesis_id: str, evidence: str) -> dict:
    """Append a single observation to a hypothesis's evidence list, bumping updated_at.

    Returns the updated HypothesisRecord, or an error if the id is not found.
    """
    result = get_hypothesis_store().append_evidence(hypothesis_id, evidence)
    if isinstance(result, str):
        return {"error": result}
    return result.model_dump(mode="json")


def list_hypotheses() -> dict:
    """List all hypothesis records for the current conversation, in creation order."""
    records = get_hypothesis_store().list_all()
    return {"hypotheses": [r.model_dump(mode="json") for r in records]}


# ---------------------------------------------------------------------------
# SimpleTool wrapper — mirrors todo_management_tool.SimpleTool
# ---------------------------------------------------------------------------


class SimpleTool:
    """Lightweight tool descriptor compatible with ToolService's by-name registry."""

    def __init__(self, name: str, description: str, func, args_schema=None):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema


def create_hypothesis_state_tools() -> List[SimpleTool]:
    """Create hypothesis state tools for ToolService registration.

    Tool names:
      record_hypothesis, update_hypothesis_status,
      append_hypothesis_evidence, list_hypotheses
    """
    return [
        SimpleTool(
            name="record_hypothesis",
            description=(
                "Record a new debugging hypothesis. "
                "Assign it an id (e.g. 'hyp_1'), set its initial status "
                "(default: proposed), and optionally supply starting evidence "
                "and a validation plan. Returns the full record with id and timestamps. "
                "Use this when you first formulate a hypothesis about the root cause."
            ),
            func=record_hypothesis,
            args_schema=RecordHypothesisInput,
        ),
        SimpleTool(
            name="update_hypothesis_status",
            description=(
                "Update the lifecycle status of an existing hypothesis by its id. "
                "Valid statuses: proposed, debugging, needs_evidence, supported, "
                "rejected, fix_proposed, validated. "
                "Use this as evidence accumulates and the hypothesis evolves."
            ),
            func=update_hypothesis_status,
            args_schema=UpdateHypothesisStatusInput,
        ),
        SimpleTool(
            name="append_hypothesis_evidence",
            description=(
                "Append a single observation (one string) to a hypothesis's evidence list. "
                "Call once per observation; call multiple times if you have multiple "
                "new observations. The existing evidence list is never overwritten."
            ),
            func=append_hypothesis_evidence,
            args_schema=AppendHypothesisEvidenceInput,
        ),
        SimpleTool(
            name="list_hypotheses",
            description=(
                "List all hypothesis records for the current debugging session, "
                "in creation order. Use to review the current state of all hypotheses "
                "before deciding which to pursue or update."
            ),
            func=list_hypotheses,
            args_schema=None,
        ),
    ]


import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.modules.context_graph.wiring import build_container_for_user_session
from domain.actor import Actor
from domain.agent_context_port import (
    build_context_record_source_id,
    bundle_to_agent_envelope,
    context_port_manifest,
    context_recipe_for_intent,
    normalize_record_type,
)
from domain.graph_query import (
    ContextGraphGoal,
    ContextGraphQuery,
    ContextGraphScope,
    ContextGraphStrategy,
)
from domain.ingestion_event_models import IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
from domain.intelligence_models import (
    ContextBudget,
    ContextResolutionRequest,
    ContextScope,
)


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_as_of(value: Optional[str]) -> datetime | None:
    if not value or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def _clean_scope(scope: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in scope.items() if value not in (None, [])}


class ContextResolveInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id")
    query: str = Field(description="Task or question to resolve context for")
    consumer_hint: Optional[str] = Field(default=None)
    intent: Optional[str] = Field(
        default=None,
        description="Optional intent such as feature, debugging, review, operations, docs, onboarding",
    )
    repo_name: Optional[str] = None
    branch: Optional[str] = None
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    symbol: Optional[str] = None
    pr_number: Optional[int] = Field(default=None, ge=1)
    services: Optional[str] = Field(
        default=None, description="Comma-separated services"
    )
    features: Optional[str] = Field(
        default=None, description="Comma-separated features"
    )
    environment: Optional[str] = None
    ticket_ids: Optional[str] = Field(
        default=None, description="Comma-separated ticket ids"
    )
    user: Optional[str] = None
    source_refs: Optional[str] = Field(
        default=None, description="Comma-separated source refs"
    )
    include: Optional[str] = Field(
        default=None, description="Comma-separated context families"
    )
    exclude: Optional[str] = Field(
        default=None, description="Comma-separated context families"
    )
    mode: str = Field(default="fast", description="fast, balanced, deep, or verify")
    source_policy: str = Field(
        default="references_only",
        description="references_only, summary, verify, or snippets",
    )
    max_items: int = Field(default=12, ge=1, le=50)
    max_tokens: Optional[int] = Field(default=None, ge=256)
    timeout_ms: int = Field(default=4000, ge=500, le=30000)
    freshness: str = "prefer_fresh"
    as_of: Optional[str] = Field(
        default=None, description="Optional ISO-8601 timestamp"
    )


class ContextSearchInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id")
    query: str = Field(description="Specific follow-up memory search query")
    limit: int = Field(default=8, ge=1, le=50)
    node_labels: Optional[str] = Field(
        default=None, description="Comma-separated label filters"
    )
    repo_name: Optional[str] = None
    source_description: Optional[str] = None
    include_invalidated: bool = False
    as_of: Optional[str] = Field(
        default=None, description="Optional ISO-8601 timestamp"
    )


class ContextRecordInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id")
    record_type: str = Field(
        description="decision, fix, bug_pattern, investigation, diagnostic_signal, preference, workflow, feature_note, service_note, runbook_note, integration_note, incident_summary, or doc_reference"
    )
    summary: str = Field(min_length=1)
    details: Optional[str] = Field(default=None, description="Optional details text")
    repo_name: Optional[str] = None
    source_refs: Optional[str] = Field(
        default=None, description="Comma-separated source refs"
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    visibility: str = "project"
    idempotency_key: Optional[str] = None
    sync: bool = False


class ContextStatusInput(BaseModel):
    pot_id: str = Field(description="Context graph pot scope id")
    repo_name: Optional[str] = None
    source_refs: Optional[str] = Field(
        default=None, description="Comma-separated source refs"
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional intent used to return a recommended context_resolve recipe",
    )


class AgentContextTools:
    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self._container = build_container_for_user_session(sql_db, user_id)

    def _assert_pot_access(self, pot_id: str) -> None:
        if self._container.pots.resolve_pot(pot_id) is None:
            raise ValueError("Pot scope not found for user")

    async def context_resolve(
        self,
        pot_id: str,
        query: str,
        consumer_hint: Optional[str] = None,
        intent: Optional[str] = None,
        repo_name: Optional[str] = None,
        branch: Optional[str] = None,
        file_path: Optional[str] = None,
        function_name: Optional[str] = None,
        symbol: Optional[str] = None,
        pr_number: Optional[int] = None,
        services: Optional[str] = None,
        features: Optional[str] = None,
        environment: Optional[str] = None,
        ticket_ids: Optional[str] = None,
        user: Optional[str] = None,
        source_refs: Optional[str] = None,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        mode: str = "fast",
        source_policy: str = "references_only",
        max_items: int = 12,
        max_tokens: Optional[int] = None,
        timeout_ms: int = 4000,
        freshness: str = "prefer_fresh",
        as_of: Optional[str] = None,
    ) -> dict[str, Any]:
        self._assert_pot_access(pot_id)
        if not self._container.settings.is_enabled():
            return {"ok": False, "error": "context_graph_disabled"}
        if self._container.resolution_service is None:
            return {"ok": False, "error": "resolver_unavailable"}

        req = ContextResolutionRequest(
            pot_id=pot_id,
            query=query,
            consumer_hint=consumer_hint,
            intent=intent,
            scope=ContextScope(
                repo_name=repo_name,
                branch=branch,
                file_path=file_path,
                function_name=function_name,
                symbol=symbol,
                pr_number=pr_number,
                services=_split_csv(services),
                features=_split_csv(features),
                environment=environment,
                ticket_ids=_split_csv(ticket_ids),
                user=user,
                source_refs=_split_csv(source_refs),
            ),
            include=_split_csv(include),
            exclude=_split_csv(exclude),
            mode=mode,
            source_policy=source_policy,
            budget=ContextBudget(
                max_items=max_items,
                max_tokens=max_tokens,
                timeout_ms=timeout_ms,
                freshness=freshness,
            ),
            as_of=_parse_as_of(as_of),
        )
        bundle = await self._container.resolution_service.resolve(req)
        return bundle_to_agent_envelope(bundle)

    async def context_search(
        self,
        pot_id: str,
        query: str,
        limit: int = 8,
        node_labels: Optional[str] = None,
        repo_name: Optional[str] = None,
        source_description: Optional[str] = None,
        include_invalidated: bool = False,
        as_of: Optional[str] = None,
    ) -> dict[str, Any]:
        self._assert_pot_access(pot_id)
        if not self._container.settings.is_enabled():
            return {"ok": False, "error": "context_graph_disabled"}
        if self._container.context_graph is None:
            return {"ok": False, "error": "context_graph_unavailable"}
        out = await self._container.context_graph.query_async(
            ContextGraphQuery(
                pot_id=pot_id,
                query=query,
                goal=ContextGraphGoal.RETRIEVE,
                strategy=ContextGraphStrategy.SEMANTIC,
                limit=limit,
                node_labels=_split_csv(node_labels),
                scope=ContextGraphScope(repo_name=repo_name),
                source_descriptions=(
                    [source_description] if source_description else []
                ),
                include_invalidated=include_invalidated,
                as_of=_parse_as_of(as_of),
            )
        )
        rows = out.result if isinstance(out.result, list) else []
        return {
            "ok": out.error is None,
            "answer": {"summary": f"Found {len(rows)} context search result(s)."},
            "evidence": rows,
            "source_refs": [],
            "coverage": {
                "status": "complete" if rows else "empty",
                "available": ["semantic_search"] if rows else [],
                "missing": [] if rows else ["semantic_search"],
                "missing_reasons": {} if rows else {"semantic_search": "empty_result"},
            },
            "freshness": {
                "status": "unknown",
                "last_graph_update": None,
                "last_source_verification": None,
                "stale_refs": [],
                "needs_verification_refs": [],
            },
            "fallbacks": [],
            "recommended_next_actions": [],
            "error": out.error,
            "meta": out.meta,
        }

    async def context_record(
        self,
        pot_id: str,
        record_type: str,
        summary: str,
        details: Optional[str] = None,
        repo_name: Optional[str] = None,
        source_refs: Optional[str] = None,
        confidence: float = 0.7,
        visibility: str = "project",
        idempotency_key: Optional[str] = None,
        sync: bool = False,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._context_record_sync,
            pot_id,
            record_type,
            summary,
            details,
            repo_name,
            source_refs,
            confidence,
            visibility,
            idempotency_key,
            sync,
        )

    def _context_record_sync(
        self,
        pot_id: str,
        record_type: str,
        summary: str,
        details: Optional[str],
        repo_name: Optional[str],
        source_refs: Optional[str],
        confidence: float,
        visibility: str,
        idempotency_key: Optional[str],
        sync: bool,
    ) -> dict[str, Any]:
        self._assert_pot_access(pot_id)
        if not self._container.settings.is_enabled():
            return {"ok": False, "error": "context_graph_disabled"}
        normalized_type = normalize_record_type(record_type)
        scope = _clean_scope(
            {"repo_name": repo_name, "source_refs": _split_csv(source_refs)}
        )
        refs = _split_csv(source_refs)
        source_id = build_context_record_source_id(
            record_type=normalized_type,
            summary=summary,
            scope=scope,
            source_refs=refs,
            idempotency_key=idempotency_key,
        )
        req = IngestionSubmissionRequest(
            pot_id=pot_id,
            ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
            source_channel="system",
            source_system="agent",
            event_type="context_record",
            action=normalized_type,
            source_id=source_id,
            repo_name=repo_name,
            artifact_refs=tuple(refs),
            idempotency_key=idempotency_key,
            actor=Actor(
                user_id=self.user_id,
                surface="system",
                client_name="potpie-agent",
                auth_method="system",
            ),
            payload={
                "record": {
                    "type": normalized_type,
                    "summary": summary,
                    "details": {"text": details} if details else {},
                    "source_refs": refs,
                    "confidence": confidence,
                    "visibility": visibility,
                },
                "scope": scope,
            },
        )
        try:
            receipt = self._container.ingestion_submission(self.sql_db).submit(
                req,
                sync=sync,
            )
            self.sql_db.commit()
        except Exception:
            self.sql_db.rollback()
            raise
        return {
            "ok": receipt.error is None,
            "status": "duplicate" if receipt.duplicate else receipt.status,
            "event_id": receipt.event_id,
            "job_id": receipt.job_id,
            "record_type": normalized_type,
            "source_id": source_id,
            "fallbacks": [
                {
                    "code": "record_queued",
                    "message": "The context record was accepted and queued for reconciliation.",
                    "impact": "It may not appear in graph reads until the worker applies it.",
                }
            ]
            if receipt.status == "queued" and not receipt.duplicate
            else [],
            "error": receipt.error,
        }

    def context_status(
        self,
        pot_id: str,
        repo_name: Optional[str] = None,
        source_refs: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> dict[str, Any]:
        resolved = self._container.pots.resolve_pot(pot_id)
        if resolved is None:
            raise ValueError("Pot scope not found for user")
        refs = _split_csv(source_refs)
        gaps: list[dict[str, str]] = []
        if not self._container.settings.is_enabled():
            gaps.append(
                {
                    "code": "context_graph_disabled",
                    "message": "Context graph is disabled for this server.",
                }
            )
        if self._container.resolution_service is None:
            gaps.append(
                {
                    "code": "resolver_unavailable",
                    "message": "Context resolution service is not configured.",
                }
            )
        if not resolved.repos:
            gaps.append(
                {
                    "code": "no_repositories",
                    "message": "This pot has no attached repositories.",
                }
            )
        return {
            "ok": not gaps,
            "pot": {
                "id": resolved.pot_id,
                "name": resolved.name,
                "ready": resolved.ready,
                "repos": [asdict(repo) for repo in resolved.repos],
            },
            "scope": _clean_scope({"repo_name": repo_name, "source_refs": refs}),
            "coverage": {
                "status": "partial" if gaps else "complete",
                "available": ["pot", "repositories"] if resolved.repos else ["pot"],
                "missing": [gap["code"] for gap in gaps],
                "missing_reasons": {gap["code"]: gap["message"] for gap in gaps},
            },
            "freshness": {
                "status": "unknown",
                "last_graph_update": None,
                "last_source_verification": None,
                "stale_refs": [],
                "needs_verification_refs": refs,
            },
            "agent_port": context_port_manifest(),
            "recommended_recipe": context_recipe_for_intent(intent),
            "fallbacks": gaps,
            "recommended_next_actions": []
            if gaps
            else [
                {
                    "action": "resolve",
                    "intent": context_recipe_for_intent(intent)["intent"],
                    "include": context_recipe_for_intent(intent)["include"],
                    "mode": context_recipe_for_intent(intent)["mode"],
                    "source_policy": context_recipe_for_intent(intent)["source_policy"],
                    "reason": "Gather a bounded context wrap for the task scope.",
                }
            ],
        }


def create_agent_context_tools(sql_db: Session, user_id: str) -> list[StructuredTool]:
    instance = AgentContextTools(sql_db, user_id)
    return [
        StructuredTool.from_function(
            coroutine=instance.context_resolve,
            name="context_resolve",
            description=(
                "Primary context graph tool. Resolve a bounded task context wrap with answer, facts, evidence, "
                "source refs, coverage, freshness, quality, fallbacks, and recommended next actions."
            ),
            args_schema=ContextResolveInput,
        ),
        StructuredTool.from_function(
            coroutine=instance.context_search,
            name="context_search",
            description=(
                "Narrow follow-up memory search after context_resolve. Use only when the needed entity or phrase is already known."
            ),
            args_schema=ContextSearchInput,
        ),
        StructuredTool.from_function(
            coroutine=instance.context_record,
            name="context_record",
            description=(
                "Record durable project memory such as decisions, fixes, preferences, workflows, feature notes, and incidents."
            ),
            args_schema=ContextRecordInput,
        ),
        StructuredTool.from_function(
            func=instance.context_status,
            name="context_status",
            description=(
                "Cheap pot readiness and trust check. Returns source coverage, freshness gaps, agent port manifest, and recommended context_resolve recipe."
            ),
            args_schema=ContextStatusInput,
        ),
    ]

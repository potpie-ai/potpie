"""Benchmark target runners."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Protocol

from benchmarks.models import BenchmarkDataset


class BenchmarkRunner(Protocol):
    pot_id: str
    repo_name: str | None

    async def seed_episode(self, episode: dict[str, Any]) -> dict[str, Any]: ...
    async def seed_record(self, record: dict[str, Any]) -> dict[str, Any]: ...
    async def ingest_pr(self, pr_number: int, repo_name: str | None = None) -> dict[str, Any]: ...
    async def query(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def status(self, intent: str | None = None) -> dict[str, Any]: ...


class ApiRunner:
    def __init__(self, *, base_url: str, api_key: str, pot_id: str, repo_name: str | None) -> None:
        from adapters.outbound.http.potpie_context_api_client import PotpieContextApiClient

        self.client = PotpieContextApiClient(base_url, api_key, timeout=120.0)
        self.pot_id = pot_id
        self.repo_name = repo_name

    async def seed_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        body = _episode_body(self.pot_id, episode)
        loop = asyncio.get_running_loop()
        status_code, payload = await loop.run_in_executor(
            None, lambda: self.client.ingest(body, sync=True)
        )
        return {"status_code": status_code, **payload}

    async def seed_record(self, record: dict[str, Any]) -> dict[str, Any]:
        body = _record_body(self.pot_id, record, self.repo_name)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.record(body, sync=True))

    async def ingest_pr(self, pr_number: int, repo_name: str | None = None) -> dict[str, Any]:
        body = {"pot_id": self.pot_id, "pr_number": pr_number}
        if repo_name or self.repo_name:
            body["repo_name"] = repo_name or self.repo_name
        loop = asyncio.get_running_loop()

        def _call() -> dict[str, Any]:
            response = self.client.post_context("/ingest-pr", json_body=body)
            self.client._raise_for_status(response)
            out = response.json()
            return out if isinstance(out, dict) else {}

        return await loop.run_in_executor(None, _call)

    async def query(self, body: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.context_graph_query(body))

    async def status(self, intent: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"pot_id": self.pot_id}
        if intent:
            body["intent"] = intent
        if self.repo_name:
            body["scope"] = {"repo_name": self.repo_name}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.status(body))


class HttpE2ERunner:
    def __init__(self, *, pot_id: str, repo_name: str | None) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from adapters.inbound.http.api.v1.context.router import create_context_router
        from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
        from adapters.outbound.intelligence.mock import MockIntelligenceProvider
        from application.services.context_resolution import ContextResolutionService
        from bootstrap.container import ContextEngineContainer
        from bootstrap.http_projects import ExplicitPotResolution
        from domain.ports.jobs import NoOpJobEnqueue
        from scripts.context_engine_lab import _InMemoryEpisodicGraph, _InMemoryStructuralGraph, _LabSettings

        self.pot_id = pot_id
        self.repo_name = repo_name
        episodic = _InMemoryEpisodicGraph()
        structural = _InMemoryStructuralGraph()
        provider = MockIntelligenceProvider()
        resolution_service = ContextResolutionService(provider)
        context_graph = GraphitiContextGraphAdapter(
            episodic=episodic,
            structural=structural,
            resolution_service=resolution_service,
        )
        container = ContextEngineContainer(
            settings=_LabSettings(),
            episodic=episodic,
            structural=structural,
            pots=ExplicitPotResolution({pot_id: repo_name or ""}),
            source_for_repo=lambda _repo_name: None,
            intelligence_provider=provider,
            resolution_service=resolution_service,
            reconciliation_agent=None,
            jobs=NoOpJobEnqueue(),
            context_graph=context_graph,
        )
        app = FastAPI()
        app.include_router(
            create_context_router(
                require_auth=lambda: {"user_id": "context-engine-benchmark"},
                get_container=lambda: container,
                get_db=lambda: None,
                get_db_optional=lambda: None,
                enforce_pot_access=False,
            ),
            prefix="/context",
        )
        self._client = TestClient(app)

    def _post(self, path: str, body: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._client.post(path, json=body, params=params)
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"{exc}; body={response.text}") from exc
        out = response.json() if response.content else {}
        return out if isinstance(out, dict) else {}

    async def seed_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        body = _episode_body(self.pot_id, episode)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._post("/context/ingest", body, {"sync": "true"})
        )

    async def seed_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {"status": "skipped", "reason": "http-e2e lab container has no reconciliation agent", "type": record.get("type")}

    async def ingest_pr(self, pr_number: int, repo_name: str | None = None) -> dict[str, Any]:
        # The in-process lab client intentionally has no source-control adapter.
        # Seed PR fixtures as built PR episodes for deterministic no-network runs.
        return {"status": "skipped", "reason": "http-e2e uses PR episode fixtures", "pr_number": pr_number}

    async def query(self, body: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._post("/context/query/context-graph", body)
        )

    async def status(self, intent: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"pot_id": self.pot_id}
        if intent:
            body["intent"] = intent
        if self.repo_name:
            body["scope"] = {"repo_name": self.repo_name}
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._post("/context/status", body))


class MockRunner:
    def __init__(self, *, pot_id: str, repo_name: str | None) -> None:
        from adapters.outbound.intelligence.mock import MockIntelligenceProvider
        from application.services.context_resolution import ContextResolutionService
        from application.use_cases.resolve_context import resolve_context
        from domain.intelligence_models import ContextBudget, ContextResolutionRequest, ContextScope

        self.pot_id = pot_id
        self.repo_name = repo_name
        self.provider = MockIntelligenceProvider()
        self.service = ContextResolutionService(self.provider)
        self._resolve_context = resolve_context
        self._ContextResolutionRequest = ContextResolutionRequest
        self._ContextScope = ContextScope
        self._ContextBudget = ContextBudget

    async def seed_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        return {"status": "mock_seeded", "name": episode.get("name")}

    async def seed_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {"status": "mock_recorded", "type": record.get("type")}

    async def ingest_pr(self, pr_number: int, repo_name: str | None = None) -> dict[str, Any]:
        return {"status": "mock_pr_ingested", "pr_number": pr_number, "repo_name": repo_name or self.repo_name}

    async def query(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("goal") == "answer":
            return await self._resolve(body)
        hits = await self.provider.search_context(
            self.pot_id,
            body.get("query") or "",
            limit=int(body.get("limit") or 8),
            node_labels=body.get("node_labels"),
        )
        return {"kind": "semantic", "goal": "retrieve", "strategy": body.get("strategy", "semantic"), "result": hits}

    async def status(self, intent: str | None = None) -> dict[str, Any]:
        from domain.agent_context_port import context_recipe_for_intent

        return {
            "ok": True,
            "pot_id": self.pot_id,
            "status": "ready",
            "recommended_recipe": context_recipe_for_intent(intent),
        }

    async def _resolve(self, body: dict[str, Any]) -> dict[str, Any]:
        from domain.agent_context_port import bundle_to_agent_envelope

        scope = dict(body.get("scope") or {})
        req = self._ContextResolutionRequest(
            pot_id=self.pot_id,
            query=body.get("query") or "",
            intent=body.get("intent"),
            include=list(body.get("include") or []),
            mode=body.get("mode") or "fast",
            source_policy=body.get("source_policy") or "references_only",
            scope=self._ContextScope(**scope),
            budget=self._ContextBudget(**dict(body.get("budget") or {})),
        )
        bundle = await self._resolve_context(self.service, req)
        return bundle_to_agent_envelope(bundle)


def make_runner(
    mode: str,
    dataset: BenchmarkDataset,
    *,
    pot_id: str | None = None,
    repo_name: str | None = None,
) -> BenchmarkRunner:
    pid = pot_id or dataset.pot_id
    repo = repo_name or dataset.repo_name
    if mode == "api":
        from adapters.inbound.cli.credentials_store import get_active_pot_id
        from adapters.inbound.cli.potpie_api_config import resolve_potpie_api_base_url, resolve_potpie_api_key

        api_key = resolve_potpie_api_key()
        if not api_key:
            raise RuntimeError("No Potpie API key configured. Run `potpie login` or set POTPIE_API_KEY.")
        return ApiRunner(
            base_url=resolve_potpie_api_base_url(),
            api_key=api_key,
            pot_id=pot_id or get_active_pot_id() or dataset.pot_id,
            repo_name=repo,
        )
    if mode == "http-e2e":
        return HttpE2ERunner(pot_id=pid, repo_name=repo)
    if mode == "mock":
        return MockRunner(pot_id=pid, repo_name=repo)
    raise ValueError(f"unknown benchmark mode: {mode}")


def _episode_body(pot_id: str, episode: dict[str, Any]) -> dict[str, Any]:
    reference_time = episode.get("reference_time") or datetime.now(timezone.utc).isoformat()
    if isinstance(reference_time, datetime):
        reference_time = reference_time.isoformat()
    return {
        "pot_id": pot_id,
        "name": episode["name"],
        "episode_body": episode.get("episode_body") or episode.get("body") or "",
        "source_description": episode.get("source_description") or episode.get("source") or "benchmark",
        "reference_time": reference_time,
        "idempotency_key": episode.get("idempotency_key"),
    }


def _record_body(pot_id: str, record: dict[str, Any], repo_name: str | None) -> dict[str, Any]:
    # Omit repo_name from scope to avoid repo_not_in_pot when the benchmark repo
    # is not explicitly attached to the target pot. The server falls back to the
    # pot's primary repo automatically.
    scope: dict[str, Any] = {}
    if record.get("scope"):
        scope = dict(record["scope"])
    body: dict[str, Any] = {
        "pot_id": pot_id,
        "record": {
            "type": record["type"],
            "summary": record["summary"],
            "details": record.get("details") or {},
            "source_refs": record.get("source_refs") or [],
            "confidence": record.get("confidence"),
        },
        "scope": scope,
        "idempotency_key": record.get("idempotency_key"),
    }
    return body

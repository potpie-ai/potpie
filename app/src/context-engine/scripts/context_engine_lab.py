#!/usr/bin/env python3
"""Context-engine local lab and smoke-test harness.

The script has two modes:

- ``mock-e2e`` runs the agent context flows in-process with deterministic mock data.
- ``http-e2e`` mounts the context HTTP router in-process with no API key.
- ``api-smoke`` exercises a configured Potpie ``/api/v2/context`` server.

Run from repo root with:

    uv run python app/src/context-engine/scripts/context_engine_lab.py mock-e2e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from adapters.inbound.http.api.v1.context.router import create_context_router  # noqa: E402
from adapters.outbound.cli_auth.credentials_store import get_active_pot_id  # noqa: E402
from adapters.outbound.cli_auth.potpie_api_config import (  # noqa: E402
    resolve_potpie_api_base_url,
    resolve_potpie_api_key,
)
from adapters.outbound.http.potpie_context_api_client import (  # noqa: E402
    IngestRejectedError,
    PotpieContextApiClient,
    PotpieContextApiError,
)
from adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore  # noqa: E402
from application.services.envelope_builder import envelope_to_dict  # noqa: E402
from application.services.read_orchestrator import ReadOrchestrator  # noqa: E402
from bootstrap.ingestion_server import IngestionServerContainer  # noqa: E402
from bootstrap.http_projects import ExplicitPotResolution  # noqa: E402
from domain.agent_context_port import (  # noqa: E402
    build_context_record_source_id,
    context_recipe_for_intent,
    normalize_record_type,
)
from domain.ports.context_graph_job_queue import NoOpContextGraphJobQueue  # noqa: E402

DEFAULT_DATA = Path(__file__).with_name("mock_context_data.json")
DEFAULT_REPORT = PACKAGE_ROOT / ".tmp" / "context-engine-lab-report.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample-data", help="Print bundled mock context data.")
    sample.add_argument("--data", type=Path, default=DEFAULT_DATA)

    mock = sub.add_parser("mock-e2e", help="Run deterministic in-process E2E flows.")
    mock.add_argument("--data", type=Path, default=DEFAULT_DATA)
    mock.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    mock.add_argument("--print-json", action="store_true")

    http = sub.add_parser(
        "http-e2e",
        help="Run the context HTTP router in-process with no API key or Potpie server.",
    )
    http.add_argument("--data", type=Path, default=DEFAULT_DATA)
    http.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    http.add_argument("--print-json", action="store_true")

    api = sub.add_parser(
        "api-smoke", help="Run smoke tests against Potpie /api/v2/context."
    )
    api.add_argument("--data", type=Path, default=DEFAULT_DATA)
    api.add_argument("--pot-id", default=None)
    api.add_argument("--repo-name", default=None)
    api.add_argument(
        "--write", action="store_true", help="Also ingest sample episodes."
    )
    api.add_argument(
        "--record",
        action="store_true",
        help="Also call context_record for sample records. Implies --write.",
    )
    api.add_argument(
        "--sync", action="store_true", help="Use sync=true for ingest/record."
    )
    api.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    api.add_argument("--print-json", action="store_true")

    args = parser.parse_args()
    if args.command == "sample-data":
        print_json(_load_data(args.data))
        return 0
    if args.command == "mock-e2e":
        return _run_mock_e2e(args)
    if args.command == "http-e2e":
        return _run_http_e2e(args)
    if args.command == "api-smoke":
        return _run_api_smoke(args)
    parser.error("unknown command")
    return 2


def _run_mock_e2e(args: argparse.Namespace) -> int:
    data = _load_data(args.data)
    result = asyncio.run(_mock_e2e(data))
    _write_report(args.report, result)
    if args.print_json:
        print_json(result)
    else:
        _print_summary(result)
        print(f"report={args.report}")
    return 0 if result["ok"] else 1


def _run_http_e2e(args: argparse.Namespace) -> int:
    data = _load_data(args.data)
    result = _http_e2e(data)
    _write_report(args.report, result)
    if args.print_json:
        print_json(result)
    else:
        _print_summary(result)
        print(f"report={args.report}")
    return 0 if result["ok"] else 1


async def _mock_e2e(data: dict[str, Any]) -> dict[str, Any]:
    # Mock mode drives the single read trunk over an (empty) in-memory claim
    # store: each query resolves to an AgentEnvelope — backed includes come
    # back empty, planned ones surface as unsupported_include.
    orchestrator = ReadOrchestrator(claim_query=InMemoryClaimQueryStore())
    pot_id = str(data["pot_id"])
    repo_name = str(data["repo_name"])
    flows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for item in data["queries"]:
        recipe = context_recipe_for_intent(item.get("intent"))
        include = item.get("include") or recipe["include"]
        env = orchestrator.resolve(
            pot_id=pot_id,
            intent=item.get("intent"),
            query=item["query"],
            include=include,
            scope={"repo": repo_name, "service": "context-engine", "environment": "staging"},
            max_items=8,
        )
        envelope = envelope_to_dict(env)
        flow = {
            "name": item["name"],
            "ok": _validate_envelope(envelope),
            "intent": envelope["intent"],
            "confidence": envelope["overall_confidence"],
            "items": len(envelope["items"]),
            "unsupported": [u["name"] for u in envelope["unsupported_includes"]],
        }
        if not flow["ok"]:
            failures.append(
                {
                    "flow": item["name"],
                    "error": "envelope failed required-field validation",
                }
            )
        flows.append(flow)

    record_results = []
    for record in data["records"]:
        try:
            record_type = normalize_record_type(record["type"])
        except ValueError as exc:
            failures.append({"flow": "record", "error": str(exc)})
            continue
        source_id = build_context_record_source_id(
            record_type=record_type,
            summary=record["summary"],
            scope={"repo_name": repo_name},
            source_refs=list(record.get("source_refs") or []),
            idempotency_key=None,
        )
        record_results.append(
            {
                "type": record_type,
                "source_id": source_id,
                "summary": record["summary"],
            }
        )

    return {
        "ok": not failures,
        "mode": "mock-e2e",
        "pot_id": pot_id,
        "repo_name": repo_name,
        "flows": flows,
        "records": record_results,
        "failures": failures,
    }


def _http_e2e(data: dict[str, Any]) -> dict[str, Any]:
    pot_id = str(data["pot_id"])
    repo_name = str(data["repo_name"])
    client = _build_in_process_client(pot_id, repo_name)
    result: dict[str, Any] = {
        "ok": True,
        "mode": "http-e2e",
        "pot_id": pot_id,
        "repo_name": repo_name,
        "steps": [],
        "failures": [],
    }

    _http_step(
        result,
        "status",
        client,
        "POST",
        "/context/status",
        json={
            "pot_id": pot_id,
            "intent": "debugging",
            "scope": {
                "repo_name": repo_name,
                "source_refs": ["lab:episode:repo-timeout-fix"],
            },
        },
    )

    for episode in data["episodes"]:
        _http_step(
            result,
            f"ingest:{episode['name']}",
            client,
            "POST",
            "/context/ingest",
            params={"sync": "true"},
            json={
                "pot_id": pot_id,
                "name": episode["name"],
                "episode_body": episode["body"],
                "source_description": episode["source"],
                "reference_time": datetime.now(timezone.utc).isoformat(),
                "idempotency_key": episode.get("idempotency_key"),
            },
        )

    _http_step(
        result,
        "search",
        client,
        "POST",
        "/context/query/context-graph",
        json={
            "pot_id": pot_id,
            "query": "repository ingestion timeout",
            "goal": "retrieve",
            "strategy": "semantic",
            "limit": 5,
            "scope": {"repo_name": repo_name} if repo_name else {},
        },
    )

    for query in data["queries"]:
        recipe = context_recipe_for_intent(query.get("intent"))
        _http_step(
            result,
            f"resolve:{query['name']}",
            client,
            "POST",
            "/context/query/context-graph",
            json={
                "pot_id": pot_id,
                "query": query["query"],
                "goal": "answer",
                "strategy": "hybrid",
                "intent": query.get("intent"),
                "scope": {
                    "repo_name": repo_name,
                    "services": ["context-engine"],
                    "environment": "staging",
                },
                "include": query.get("include") or recipe["include"],
                "source_policy": recipe["source_policy"],
                "budget": {"max_items": 8, "timeout_ms": 3000},
            },
        )

    for record in data["records"]:
        _record_step(result, record, repo_name)

    _http_step(
        result,
        "reset",
        client,
        "POST",
        "/context/reset",
        json={"pot_id": pot_id, "skip_ledger": True},
    )

    result["ok"] = not result["failures"]
    return result


def _run_api_smoke(args: argparse.Namespace) -> int:
    data = _load_data(args.data)
    pot_id = args.pot_id or get_active_pot_id() or data.get("pot_id")
    if not pot_id:
        raise SystemExit(
            "pot_id missing: pass --pot-id or run `potpie pot use`"
        )
    repo_name = args.repo_name or data.get("repo_name")
    client = PotpieContextApiClient(
        resolve_potpie_api_base_url(),
        resolve_potpie_api_key(),
    )
    result = {
        "ok": True,
        "mode": "api-smoke",
        "pot_id": pot_id,
        "repo_name": repo_name,
        "steps": [],
        "failures": [],
    }

    _api_step(result, "health", lambda: {"status_code": client.get_health()[0]})
    _api_step(
        result,
        "status",
        lambda: client.status(
            {
                "pot_id": pot_id,
                "intent": "debugging",
                "scope": {"repo_name": repo_name} if repo_name else {},
            }
        ),
    )

    if args.write or args.record:
        for episode in data["episodes"]:
            body = {
                "pot_id": pot_id,
                "name": episode["name"],
                "episode_body": episode["body"],
                "source_description": episode["source"],
                "reference_time": datetime.now(timezone.utc),
                "idempotency_key": episode.get("idempotency_key"),
            }
            _api_step(
                result,
                f"ingest:{episode['name']}",
                lambda body=body: client.ingest(body, sync=args.sync)[1],
            )

    _api_step(
        result,
        "search",
        lambda: client.context_graph_query(
            {
                "pot_id": pot_id,
                "query": "repository ingestion timeout context-engine",
                "goal": "retrieve",
                "strategy": "semantic",
                "limit": 5,
                "scope": {"repo_name": repo_name} if repo_name else {},
            }
        ),
    )

    for query in data["queries"]:
        recipe = context_recipe_for_intent(query.get("intent"))
        _api_step(
            result,
            f"resolve:{query['name']}",
            lambda query=query, recipe=recipe: client.context_graph_query(
                {
                    "pot_id": pot_id,
                    "query": query["query"],
                    "goal": "answer",
                    "strategy": "hybrid",
                    "intent": query.get("intent"),
                    "scope": {
                        key: value
                        for key, value in {
                            "repo_name": repo_name,
                            "services": ["context-engine"],
                            "environment": "staging",
                        }.items()
                        if value
                    },
                    "include": query.get("include") or recipe["include"],
                    "source_policy": recipe["source_policy"],
                    "budget": {"max_items": 8, "timeout_ms": 3000},
                },
            ),
        )

    if args.record:
        for record in data["records"]:
            _api_step(
                result,
                f"record:{record['type']}",
                lambda record=record: client.record(
                    {
                        "pot_id": pot_id,
                        "record": record,
                        "scope": {"repo_name": repo_name} if repo_name else {},
                        "idempotency_key": f"lab:record:{record['type']}",
                    },
                    sync=args.sync,
                ),
            )

    result["ok"] = not result["failures"]
    _write_report(args.report, result)
    if args.print_json:
        print_json(result)
    else:
        _print_summary(result)
        print(f"report={args.report}")
    return 0 if result["ok"] else 1


def _build_in_process_client(pot_id: str, repo_name: str) -> TestClient:
    container = IngestionServerContainer(
        settings=_LabSettings(),
        graph_writer=_InMemoryGraphWriter(),
        pots=ExplicitPotResolution({pot_id: repo_name}),
        reconciliation_agent=None,
        jobs=NoOpContextGraphJobQueue(),
    )

    app = FastAPI()
    app.include_router(
        create_context_router(
            require_auth=lambda: {"user_id": "context-engine-lab"},
            get_container=lambda: container,
            get_db=lambda: None,
            get_db_optional=lambda: None,
        ),
        prefix="/context",
    )
    return TestClient(app)


def _http_step(
    result: dict[str, Any],
    name: str,
    client: TestClient,
    method: str,
    path: str,
    **kwargs: Any,
) -> None:
    try:
        response = client.request(method, path, **kwargs)
        payload = response.json()
    except Exception as exc:
        result["failures"].append({"step": name, "error": str(exc)})
        result["steps"].append({"name": name, "ok": False})
        return

    ok = 200 <= response.status_code < 300
    result["steps"].append(
        {
            "name": name,
            "ok": ok,
            "status_code": response.status_code,
            "summary": _summarize(payload),
        }
    )
    if not ok:
        result["failures"].append(
            {
                "step": name,
                "status_code": response.status_code,
                "detail": payload.get("detail", payload),
            }
        )


def _record_step(
    result: dict[str, Any], record: dict[str, Any], repo_name: str
) -> None:
    try:
        record_type = normalize_record_type(record["type"])
        source_id = build_context_record_source_id(
            record_type=record_type,
            summary=record["summary"],
            scope={"repo_name": repo_name},
            source_refs=list(record.get("source_refs") or []),
            idempotency_key=None,
        )
    except Exception as exc:
        result["failures"].append(
            {"step": f"record:{record.get('type')}", "error": str(exc)}
        )
        result["steps"].append({"name": f"record:{record.get('type')}", "ok": False})
        return
    result["steps"].append(
        {
            "name": f"record:{record_type}",
            "ok": True,
            "summary": {"record_type": record_type, "source_id": source_id},
        }
    )


def _api_step(result: dict[str, Any], name: str, fn: Any) -> None:
    try:
        payload = fn()
    except IngestRejectedError as exc:
        result["failures"].append(
            {
                "step": name,
                "status_code": 422,
                "detail": exc.body,
            }
        )
        result["steps"].append({"name": name, "ok": False})
        return
    except PotpieContextApiError as exc:
        result["failures"].append(
            {
                "step": name,
                "status_code": exc.status_code,
                "detail": exc.detail,
            }
        )
        result["steps"].append({"name": name, "ok": False})
        return
    except Exception as exc:
        result["failures"].append({"step": name, "error": str(exc)})
        result["steps"].append({"name": name, "ok": False})
        return
    result["steps"].append({"name": name, "ok": True, "summary": _summarize(payload)})


def _validate_envelope(envelope: dict[str, Any]) -> bool:
    required = {
        "pot_id",
        "intent",
        "items",
        "coverage",
        "unsupported_includes",
        "overall_confidence",
    }
    return required.issubset(envelope)


def _summarize(payload: Any) -> Any:
    if isinstance(payload, tuple):
        return [_summarize(item) for item in payload]
    if isinstance(payload, list):
        return {"items": len(payload)}
    if isinstance(payload, dict):
        keys = [
            "ok",
            "status",
            "status_code",
            "event_id",
            "job_id",
            "coverage",
            "freshness",
            "quality",
            "fallbacks",
            "recommended_recipe",
        ]
        summary = {key: payload.get(key) for key in keys if key in payload}
        for key in ("coverage", "freshness", "quality"):
            if isinstance(summary.get(key), dict):
                summary[key] = {"status": summary[key].get("status")}
        if isinstance(summary.get("fallbacks"), list):
            summary["fallbacks"] = len(summary["fallbacks"])
        if isinstance(summary.get("recommended_recipe"), dict):
            summary["recommended_recipe"] = {
                "intent": summary["recommended_recipe"].get("intent"),
                "mode": summary["recommended_recipe"].get("mode"),
                "source_policy": summary["recommended_recipe"].get("source_policy"),
            }
        return summary
    return payload


def _load_data(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def print_json(payload: Any) -> None:
    print(json.dumps(_json_safe(payload), indent=2))


def _print_summary(result: dict[str, Any]) -> None:
    print(f"mode={result['mode']} ok={result['ok']} pot_id={result.get('pot_id')}")
    for flow in result.get("flows", []):
        quality = flow.get("quality", {})
        coverage = flow.get("coverage", {})
        print(
            f"flow={flow['name']} ok={flow['ok']} "
            f"coverage={coverage.get('status')} quality={quality.get('status')}"
        )
    for step in result.get("steps", []):
        print(f"step={step['name']} ok={step['ok']} summary={step.get('summary')}")
    for failure in result.get("failures", []):
        print(f"failure={failure}")


class _LabSettings:
    def is_enabled(self) -> bool:
        return True

    def neo4j_uri(self) -> str | None:
        return None

    def neo4j_user(self) -> str | None:
        return None

    def neo4j_password(self) -> str | None:
        return None

    def backfill_max_prs_per_run(self) -> int:
        return 0


class _InMemoryGraphWriter:
    """Lab-only :class:`GraphWriterPort` fake — counts mutations, no storage."""

    def __init__(self) -> None:
        self._pots: set[str] = set()

    @property
    def enabled(self) -> bool:
        return True

    async def ensure_indexes(self) -> bool:
        return True

    async def upsert_entities(
        self, pot_id: str, items: list[Any], provenance: Any
    ) -> int:
        self._pots.add(pot_id)
        return len(items)

    async def upsert_edges(
        self, pot_id: str, items: list[Any], provenance: Any
    ) -> int:
        self._pots.add(pot_id)
        return len(items)

    async def delete_edges(
        self, pot_id: str, items: list[Any], provenance: Any
    ) -> int:
        return len(items)

    async def invalidate(
        self, pot_id: str, items: list[Any], provenance: Any
    ) -> int:
        return len(items)

    async def reset_pot(self, pot_id: str) -> dict[str, Any]:
        self._pots.discard(pot_id)
        return {"ok": True}



if __name__ == "__main__":
    raise SystemExit(main())

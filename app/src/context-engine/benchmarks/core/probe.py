"""Engine pre-flight probe.

Catches the failure modes that today cost 10+ minutes of bench runtime to
expose:

- Engine not reachable / auth wrong.
- Connector kind the bench needs is not registered on the live engine.
- Reconciliation queue doesn't drain — events submit fine but stall in
  ``queued`` because no worker is consuming them.
- Pot-creation path is broken.

Usage from the CLI::

    python -m benchmarks probe [--connectors github,linear,...] [--terminal-timeout 15]

Returns 0 if everything is ready to run, 1 if any check failed. Output
is one line per check, so it is greppable.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from adapters.outbound.http.potpie_context_api_client import (
    PotpieContextApiClient,
    PotpieContextApiError,
)

from benchmarks.core.lifecycle import EphemeralPot, create_ephemeral_pot, reset_pot

logger = logging.getLogger(__name__)


@dataclass
class ProbeCheck:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0


@dataclass
class ProbeReport:
    engine_url: str
    checks: list[ProbeCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_url": self.engine_url,
            "passed": self.passed,
            "checks": [c.__dict__ for c in self.checks],
        }


# A minimal envelope shape used to test one event through reconciliation
# per connector kind. The reconciliation agent doesn't need rich payload
# content for a "did the queue drain?" check.
_MINIMAL_PAYLOADS: dict[str, dict[str, Any]] = {
    "linear": {
        "event_type": "Issue",
        "action": "create",
        "payload": {
            "action": "create",
            "type": "Issue",
            "data": {
                "id": "probe-issue-{ts}",
                "identifier": "PROBE-{ts}",
                "title": "bench probe",
                "team": {"key": "PROBE"},
                "state": {"name": "Backlog", "type": "unstarted"},
            },
        },
    },
    "github": {
        "event_type": "pull_request",
        "action": "closed",
        "payload": {
            "action": "closed",
            "pull_request": {
                "number": -1,
                "id": -1,
                "title": "bench probe",
                "merged": True,
                "merged_at": "2026-05-20T00:00:00.000Z",
                "user": {"login": "bench-probe"},
            },
            "repository": {"full_name": "acme/sandbox"},
        },
    },
    "repo_docs": {
        "event_type": "Document",
        "action": "create",
        "payload": {
            "action": "create",
            "type": "Document",
            "data": {"id": "probe-doc-{ts}", "title": "probe doc", "content": "probe"},
        },
    },
    "slack": {
        "event_type": "Message",
        "action": "create",
        "payload": {
            "action": "create",
            "type": "Message",
            "data": {"id": "probe-msg-{ts}", "text": "probe", "channel": "probe"},
        },
    },
    "alerting": {
        "event_type": "Alert",
        "action": "alert_fire",
        "payload": {
            "action": "alert_fire",
            "type": "Alert",
            "data": {"id": "probe-alert-{ts}", "title": "probe", "severity": "info"},
        },
    },
    "deploy": {
        "event_type": "Deployment",
        "action": "deploy_success",
        "payload": {
            "action": "deploy_success",
            "type": "Deployment",
            "data": {"id": "probe-deploy-{ts}", "service": "probe", "environment": "dev"},
        },
    },
    "notion": {
        "event_type": "notion_page",
        "action": "create",
        "payload": {
            "page": {"id": "probe-page-{ts}", "title": "probe page"},
        },
    },
}


def _check_engine_reachable(client: PotpieContextApiClient) -> ProbeCheck:
    """Reach for *any* engine response, including 4xx — that proves the
    process is up and the auth is correct. A connection error or 401/403
    is the failure case worth surfacing here.
    """
    started = time.monotonic()
    try:
        resp = client.post_context("/status", json_body={})
    except Exception as exc:  # noqa: BLE001
        return ProbeCheck(
            name="engine_reachable",
            passed=False,
            detail=f"transport failure: {exc}",
            elapsed_s=time.monotonic() - started,
        )
    code = resp.status_code
    if code in (401, 403):
        return ProbeCheck(
            name="engine_reachable",
            passed=False,
            detail=f"auth rejected ({code}): check POTPIE_BENCH_API_KEY",
            elapsed_s=time.monotonic() - started,
        )
    # Any other status (200, 4xx validation, 5xx) means the engine
    # accepted the request — we are talking to it.
    return ProbeCheck(
        name="engine_reachable",
        passed=True,
        detail=f"HTTP {code} (engine accepting requests)",
        elapsed_s=time.monotonic() - started,
    )


def _check_pot_creation(client: PotpieContextApiClient) -> tuple[ProbeCheck, EphemeralPot | None]:
    started = time.monotonic()
    try:
        pot = create_ephemeral_pot(client, scenario_id="probe", attach_repo=True)
    except Exception as exc:  # noqa: BLE001
        return (
            ProbeCheck(
                name="pot_create",
                passed=False,
                detail=f"create_ephemeral_pot failed: {exc}",
                elapsed_s=time.monotonic() - started,
            ),
            None,
        )
    return (
        ProbeCheck(
            name="pot_create",
            passed=True,
            detail=f"pot_id={pot.pot_id} repo={pot.repo_name}",
            elapsed_s=time.monotonic() - started,
        ),
        pot,
    )


def _check_connector_registry(client: PotpieContextApiClient, pot_id: str) -> tuple[ProbeCheck, set[str]]:
    started = time.monotonic()
    try:
        resp = client.post_context("/status", json_body={"pot_id": pot_id})
        if resp.status_code >= 400:
            return (
                ProbeCheck(
                    name="connector_registry",
                    passed=False,
                    detail=f"/context/status -> HTTP {resp.status_code} {resp.text[:200]}",
                    elapsed_s=time.monotonic() - started,
                ),
                set(),
            )
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return (
            ProbeCheck(
                name="connector_registry",
                passed=False,
                detail=f"/context/status failed: {exc}",
                elapsed_s=time.monotonic() - started,
            ),
            set(),
        )
    kinds = {c.get("kind") for c in (data.get("connectors") or []) if c.get("kind")}
    return (
        ProbeCheck(
            name="connector_registry",
            passed=bool(kinds),
            detail=f"registered: {sorted(kinds) or '(none)'}",
            elapsed_s=time.monotonic() - started,
        ),
        kinds,
    )


def _check_event_drains(
    client: PotpieContextApiClient,
    pot_id: str,
    pot_repo: str | None,
    kind: str,
    terminal_timeout_s: float,
) -> ProbeCheck:
    """Submit one probe envelope for ``kind`` and wait briefly for terminal."""
    started = time.monotonic()
    template = _MINIMAL_PAYLOADS.get(kind)
    if template is None:
        return ProbeCheck(
            name=f"drain:{kind}",
            passed=False,
            detail=f"no probe template for connector kind '{kind}'",
            elapsed_s=time.monotonic() - started,
        )
    ts = uuid.uuid4().hex[:10]
    body = {
        "pot_id": pot_id,
        "ingestion_kind": "agent_reconciliation",
        "source_system": kind,
        "event_type": template["event_type"],
        "action": template["action"],
        "source_id": f"probe:{kind}:{ts}",
        "occurred_at": "2026-05-20T00:00:00.000Z",
        "payload": json.loads(json.dumps(template["payload"]).replace("{ts}", ts)),
        "repo_name": pot_repo or "",
    }
    try:
        resp = client.post_context("/events/reconcile", json_body=body)
        if resp.status_code >= 400:
            return ProbeCheck(
                name=f"drain:{kind}",
                passed=False,
                detail=f"submit HTTP {resp.status_code} {resp.text[:200]}",
                elapsed_s=time.monotonic() - started,
            )
        submission = resp.json()
    except Exception as exc:  # noqa: BLE001
        return ProbeCheck(
            name=f"drain:{kind}",
            passed=False,
            detail=f"submit failed: {exc}",
            elapsed_s=time.monotonic() - started,
        )
    event_id = submission.get("event_id") or submission.get("id")
    if not event_id:
        return ProbeCheck(
            name=f"drain:{kind}",
            passed=False,
            detail=f"no event_id in response: {submission!r}",
            elapsed_s=time.monotonic() - started,
        )
    # Poll for terminal until ``terminal_timeout_s`` — but we only really
    # care whether a worker *picked it up*, not whether reconciliation
    # produced a clean graph. So any non-``queued`` lifecycle is a win.
    deadline = time.monotonic() + terminal_timeout_s
    last_status = "queued"
    while time.monotonic() < deadline:
        try:
            ev = client.get_event(str(event_id))
        except PotpieContextApiError:
            time.sleep(0.5)
            continue
        last_status = str(ev.get("status") or "").lower()
        lc_status = str(ev.get("lifecycle_status") or "").lower()
        stage = ev.get("stage")
        if last_status in {"done", "reconciled", "error", "failed", "rejected"}:
            return ProbeCheck(
                name=f"drain:{kind}",
                passed=True,
                detail=f"status={last_status} stage={stage}",
                elapsed_s=time.monotonic() - started,
            )
        if lc_status not in {"queued", ""} or stage:
            return ProbeCheck(
                name=f"drain:{kind}",
                passed=True,
                detail=f"picked up (lifecycle={lc_status} stage={stage})",
                elapsed_s=time.monotonic() - started,
            )
        time.sleep(0.5)
    return ProbeCheck(
        name=f"drain:{kind}",
        passed=False,
        detail=f"stuck at status={last_status} for {terminal_timeout_s:.0f}s — worker likely dead",
        elapsed_s=time.monotonic() - started,
    )


def run_probe(
    client: PotpieContextApiClient,
    *,
    expected_connectors: tuple[str, ...] = ("github", "linear"),
    terminal_timeout_s: float = 15.0,
) -> ProbeReport:
    """Run the full probe sequence. Always returns a ``ProbeReport``."""
    report = ProbeReport(engine_url=getattr(client, "_base", ""))

    reach = _check_engine_reachable(client)
    report.checks.append(reach)
    if not reach.passed:
        return report

    pot_check, pot = _check_pot_creation(client)
    report.checks.append(pot_check)
    if not pot_check.passed or pot is None:
        return report

    try:
        registry, kinds = _check_connector_registry(client, pot.pot_id)
        report.checks.append(registry)

        missing = [k for k in expected_connectors if k not in kinds]
        if missing:
            report.checks.append(
                ProbeCheck(
                    name="connectors_present",
                    passed=False,
                    detail=f"missing expected kinds: {sorted(missing)} — restart the engine with the bench-stub registrations",
                )
            )
        else:
            report.checks.append(
                ProbeCheck(name="connectors_present", passed=True, detail="all expected kinds registered")
            )

        for kind in expected_connectors:
            if kind in kinds:
                report.checks.append(
                    _check_event_drains(client, pot.pot_id, pot.repo_name, kind, terminal_timeout_s)
                )
    finally:
        if pot is not None:
            try:
                reset_pot(client, pot)
            except Exception:  # noqa: BLE001
                logger.debug("probe: reset_pot best-effort failure")

    return report


def render_probe(report: ProbeReport) -> str:
    """Human-readable, greppable one-line-per-check rendering."""
    lines = [f"engine: {report.engine_url}", ""]
    width = max((len(c.name) for c in report.checks), default=0)
    for c in report.checks:
        mark = "OK" if c.passed else "FAIL"
        elapsed = f"{c.elapsed_s:.2f}s" if c.elapsed_s > 0 else ""
        lines.append(f"  [{mark:<4}] {c.name.ljust(width)}  {elapsed:>7}  {c.detail}")
    lines.append("")
    lines.append("status: " + ("READY" if report.passed else "BLOCKED"))
    return "\n".join(lines)

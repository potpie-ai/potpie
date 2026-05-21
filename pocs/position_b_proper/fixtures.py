"""
Acme Corp synthetic universe for the Position B proper POC.

Mirrors bench-plan.md §5.1 in miniature: a small but realistic multi-source
fixture set we can ingest end-to-end via LLM extraction, then query through
agent-shaped scenarios across PREF/INFRA/TIME/BUG.

Cross-source identity stress is built in: the same "auth-svc" appears under
five different names across sources. The POC's job is to converge them.

Distractors are intentional: ~25% of events are noise (different services,
unrelated features, near-miss timestamps) — they MUST be excluded from
scenario answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal


@dataclass(frozen=True)
class FixtureEvent:
    """One ingestable raw event. Mirrors what a real source connector would
    produce after webhook normalization — body text + structured metadata."""

    event_id: str  # stable id for citation assertions
    source: str   # "github" | "linear" | "slack" | "notion" | "repo-docs" | "k8s-scanner" | "codeowners-scanner" | "alerting" | "deploy"
    kind: str     # "pr_merged", "issue_closed", "incident_opened", "doc_created", "manifest_scanned", "discussion", etc.
    body: str     # free-text content the LLM extracts from
    actor: str | None = None    # alice / bob / carol / system
    occurred_at_offset_days: float = 0.0  # negative offsets = past; "now" = 0
    role: Literal["signal", "distractor", "universe"] = "signal"
    structured_facts: dict[str, str] = field(default_factory=dict)  # facts the
    # deterministic activity layer would parse (used as ground truth, not
    # given to the LLM)


# ---------------------------------------------------------------------------
# Universe seed (the "as of -365d" state, no LLM extraction needed)
# ---------------------------------------------------------------------------


UNIVERSE_SEED: list[FixtureEvent] = [
    FixtureEvent(
        event_id="seed/services",
        source="repo-docs",
        kind="doc_created",
        body=(
            "ACME PLATFORM SERVICES\n\n"
            "We run the following backend services in production:\n"
            "- auth-svc: handles authentication, owned by alice. Stack: FastAPI, Postgres.\n"
            "- inventory-svc: stock + warehouse, owned by bob. Stack: FastAPI, Postgres, Kafka consumer.\n"
            "- checkout-api: order placement, owned by alice. Stack: FastAPI, calls auth-svc + inventory-svc.\n"
            "Environments: dev, staging, prod. Deployment via ArgoCD."
        ),
        actor=None,
        occurred_at_offset_days=-365,
        role="universe",
    ),
    FixtureEvent(
        event_id="seed/codeowners",
        source="codeowners-scanner",
        kind="manifest_scanned",
        body=(
            "# CODEOWNERS file (apps/auth/CODEOWNERS)\n"
            "* @alice\n"
            "# (the auth-service team is just alice for now)\n"
        ),
        actor=None,
        occurred_at_offset_days=-365,
        role="universe",
    ),
    FixtureEvent(
        event_id="seed/team",
        source="repo-docs",
        kind="doc_created",
        body=(
            "TEAM\n\n"
            "- alice (alice@acme.com, github: @alice-acme) — backend lead, auth + checkout.\n"
            "- bob (bob@acme.com, github: @bob-acme) — backend lead, inventory.\n"
            "- carol (carol@acme.com, github: @carol-acme) — SRE on-call rotation."
        ),
        actor=None,
        occurred_at_offset_days=-365,
        role="universe",
    ),
]


# ---------------------------------------------------------------------------
# Signal events — the things scenarios are *supposed* to find
# ---------------------------------------------------------------------------


SIGNAL_EVENTS: list[FixtureEvent] = [
    # --- PREF: project preferences (3) ---
    FixtureEvent(
        event_id="adr/007",
        source="repo-docs",
        kind="doc_created",
        body=(
            "# ADR-007: HTTP handlers must declare response_model\n\n"
            "## Context\n"
            "We've had three incidents in the last quarter where a FastAPI endpoint\n"
            "returned an untyped dict and the schema drifted from what clients expected.\n\n"
            "## Decision\n"
            "All new HTTP handlers in app/api/ MUST declare an explicit `response_model`\n"
            "on the route decorator. No bare dict returns. This applies to every endpoint\n"
            "regardless of complexity.\n\n"
            "## Consequences\n"
            "PR reviewers should reject any new handler that lacks response_model.\n"
        ),
        actor="alice",
        occurred_at_offset_days=-180,
        role="signal",
    ),
    FixtureEvent(
        event_id="pr/review/881",
        source="github",
        kind="pr_review_comment",
        body=(
            "[PR #881 review by alice on file app/services/auth/refresh.py]\n"
            "Don't raise bare `Exception` here — use AcmeError or one of its subclasses\n"
            "(AcmeAuthError, AcmeValidationError, etc.). It lets the global error\n"
            "middleware route to the right log channel and status code. This is a\n"
            "convention across the whole platform — please update before merge."
        ),
        actor="alice",
        occurred_at_offset_days=-90,
        role="signal",
    ),
    FixtureEvent(
        event_id="doc/conventions/logging",
        source="repo-docs",
        kind="doc_created",
        body=(
            "# Logging conventions\n\n"
            "Use `structlog` for all log emissions. Each log line MUST include an\n"
            "`event=` key naming the event type (e.g. `event=auth.login.success`).\n"
            "Do NOT use f-strings or `%`-formatting in log messages — pass values as\n"
            "keyword args so the structured logger can index them.\n\n"
            "Bad:  logger.info(f\"user {uid} logged in\")\n"
            "Good: logger.info(\"login.success\", event=\"auth.login.success\", user_id=uid)\n"
        ),
        actor=None,
        occurred_at_offset_days=-150,
        role="signal",
    ),

    # --- INFRA: deployment topology (4) ---
    FixtureEvent(
        event_id="k8s/auth/prod",
        source="k8s-scanner",
        kind="manifest_scanned",
        body=(
            "# k8s manifest scan: clusters/prod/auth-svc.yaml@a3f1c\n"
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata: { name: auth-svc, namespace: prod }\n"
            "spec:\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "        - name: auth\n"
            "          env:\n"
            "            - name: DATABASE_URL\n"
            "              valueFrom: { secretKeyRef: { name: auth-pg-prod, key: url } }\n"
            "            - name: REDIS_URL\n"
            "              valueFrom: { secretKeyRef: { name: auth-redis-prod, key: url } }\n"
            "---\n"
            "# Resolved: auth-svc in prod uses postgres-auth-prod + redis-auth-prod."
        ),
        actor=None,
        occurred_at_offset_days=-30,
        role="signal",
        structured_facts={
            "service": "auth-svc",
            "environment": "prod",
            "depends_on": "postgres-auth-prod,redis-auth-prod",
        },
    ),
    FixtureEvent(
        event_id="k8s/auth/staging",
        source="k8s-scanner",
        kind="manifest_scanned",
        body=(
            "# k8s manifest scan: clusters/staging/auth-service.yaml@a3f1c\n"
            "# NOTE: this file calls it 'auth-service' for legacy reasons; same service.\n"
            "kind: Deployment\n"
            "metadata: { name: auth-service, namespace: staging }\n"
            "spec:\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "        - name: auth\n"
            "          env:\n"
            "            - name: DATABASE_URL\n"
            "              valueFrom: { secretKeyRef: { name: auth-pg-staging, key: url } }\n"
            "---\n"
            "# Resolved: the auth service in staging uses postgres-auth-staging."
        ),
        actor=None,
        occurred_at_offset_days=-30,
        role="signal",
        structured_facts={
            "service": "auth-svc",  # ground truth: same service as above
            "environment": "staging",
            "depends_on": "postgres-auth-staging",
        },
    ),
    FixtureEvent(
        event_id="adr/inventory-kafka",
        source="notion",
        kind="doc_created",
        body=(
            "# Inventory streaming architecture\n\n"
            "The Inventory Service consumes from the `orders-topic` Kafka topic to\n"
            "decrement stock when an order is placed. The orders-topic is produced\n"
            "by checkout-api on order placement. This is the only producer.\n\n"
            "Inventory's consumer group is `inventory-consumer-group`. Retention: 7d."
        ),
        actor="bob",
        occurred_at_offset_days=-200,
        role="signal",
        structured_facts={
            "service": "inventory-svc",
            "depends_on": "orders-topic",
            "producer": "checkout-api",
        },
    ),
    FixtureEvent(
        event_id="codeowners/auth",
        source="codeowners-scanner",
        kind="manifest_scanned",
        body=(
            "# apps/auth/CODEOWNERS file scan\n"
            "* @alice-acme\n"
            "# auth-service team primary owner\n"
        ),
        actor=None,
        occurred_at_offset_days=-200,
        role="signal",
        structured_facts={
            "service": "auth-svc",
            "owner": "alice",
        },
    ),

    # --- TIME: timeline events around auth-svc (4) ---
    FixtureEvent(
        event_id="github/pr/1042/opened",
        source="github",
        kind="pr_opened",
        body=(
            "[PR #1042 opened by alice in acme/platform]\n"
            "Title: Add rate limiting to auth-svc /login endpoint\n\n"
            "Adds a redis-backed rate limiter (10 req/s per IP) to the login endpoint\n"
            "in app/api/v1/auth.py. Closes AUTH-42."
        ),
        actor="alice",
        occurred_at_offset_days=-5,
        role="signal",
    ),
    FixtureEvent(
        event_id="github/pr/1042/merged",
        source="github",
        kind="pr_merged",
        body=(
            "[PR #1042 merged by alice]\n"
            "Approved by carol after rate-limit math review.\n"
            "Files changed: app/api/v1/auth.py, app/services/auth/rate_limit.py."
        ),
        actor="alice",
        occurred_at_offset_days=-3,
        role="signal",
    ),
    FixtureEvent(
        event_id="deploy/auth/staging/v2.1.4",
        source="deploy",
        kind="deployment",
        body=(
            "[ArgoCD deploy] auth-svc v2.1.4 → staging. Triggered by PR #1042 merge.\n"
            "Deployed at 2026-05-17 14:30 UTC by argo-bot. Rollout completed in 4min."
        ),
        actor="argo-bot",
        occurred_at_offset_days=-3,
        role="signal",
        structured_facts={
            "service": "auth-svc",
            "environment": "staging",
            "version": "v2.1.4",
        },
    ),
    FixtureEvent(
        event_id="linear/AUTH-42/closed",
        source="linear",
        kind="issue_closed",
        body=(
            "[Linear AUTH-42 closed by alice]\n"
            "Title: Rate limit /login to prevent brute force\n"
            "Resolution: Implemented in PR #1042. Verified 10rps/IP in staging.\n"
            "Status: Done."
        ),
        actor="alice",
        occurred_at_offset_days=-2,
        role="signal",
    ),

    # --- BUG: incident + fix + postmortem (4) ---
    FixtureEvent(
        event_id="alert/ops-218",
        source="alerting",
        kind="incident_opened",
        body=(
            "[Datadog alert + PagerDuty incident OPS-218]\n"
            "auth-svc P95 latency spiked to 1800ms at 03:00 UTC on 2026-04-12.\n"
            "Database connection pool exhaustion suspected — pool of 20 connections\n"
            "saturated, queue depth > 100. Backend timeouts cascade to login failures.\n"
            "On-call carol acknowledged."
        ),
        actor="carol",
        occurred_at_offset_days=-38,
        role="signal",
    ),
    FixtureEvent(
        event_id="github/pr/998/merged",
        source="github",
        kind="pr_merged",
        body=(
            "[PR #998 merged by carol]\n"
            "Title: Increase auth-svc pgbouncer pool size to 50; add pool_pre_ping\n\n"
            "Mitigates OPS-218. Pool exhaustion under burst login traffic; raised\n"
            "from 20 to 50 connections. Added pool_pre_ping to detect dead conns\n"
            "before checkout. Tested in staging soak for 24h.\n"
            "Files: apps/auth/db.py, helm/auth/values-prod.yaml"
        ),
        actor="carol",
        occurred_at_offset_days=-35,
        role="signal",
    ),
    FixtureEvent(
        event_id="notion/postmortem/2026-04-12",
        source="notion",
        kind="doc_created",
        body=(
            "# Postmortem: OPS-218 auth-svc latency spike (2026-04-12)\n\n"
            "## Root cause\n"
            "PgBouncer pool of 20 was undersized for peak login traffic. Connection\n"
            "queue saturated, p95 latency spiked to 1800ms.\n\n"
            "## Fix\n"
            "PR #998 raised pool to 50 and added pre-ping. Deployed prod 2026-04-15.\n\n"
            "## Policy decision\n"
            "Going forward: NEVER deploy connection-pool-changing PRs without a 24h\n"
            "staging soak with realistic load. This becomes a hard rule for the\n"
            "platform team. Add to deployment checklist."
        ),
        actor="carol",
        occurred_at_offset_days=-32,
        role="signal",
    ),
    FixtureEvent(
        event_id="slack/eng-platform/pool-pattern",
        source="slack",
        kind="discussion",
        body=(
            "[Slack #eng-platform thread, 2026-04-20]\n"
            "carol: BTW since OPS-218 we've seen the same pattern emerge in inventory-svc\n"
            "  last week — pgbouncer pool of 20 was undersized once order volume spiked.\n"
            "  Same fix shape: bump pool to 50, add pre-ping. Worth treating as a BugPattern.\n"
            "bob: agree. We should add 'pgbouncer pool sizing' to the runbook.\n"
            "alice: filed as recurring pattern. Let's also bump checkout-api proactively."
        ),
        actor="carol",
        occurred_at_offset_days=-26,
        role="signal",
    ),
]


# ---------------------------------------------------------------------------
# Distractor events — noise the engine MUST exclude from scenario answers
# ---------------------------------------------------------------------------


DISTRACTOR_EVENTS: list[FixtureEvent] = [
    # Distractor PRs in unrelated services (TIME scenario must exclude these)
    FixtureEvent(
        event_id="github/pr/1037/merged",
        source="github",
        kind="pr_merged",
        body=(
            "[PR #1037 merged by bob]\n"
            "Title: Bump inventory-svc Kafka consumer batch size to 100\n"
            "Files: apps/inventory/consumer.py"
        ),
        actor="bob",
        occurred_at_offset_days=-4,
        role="distractor",
    ),
    FixtureEvent(
        event_id="github/pr/1038/merged",
        source="github",
        kind="pr_merged",
        body=(
            "[PR #1038 merged by alice]\n"
            "Title: checkout-api: add idempotency-key header to order placement\n"
            "Files: apps/checkout/api/orders.py — has nothing to do with auth-svc."
        ),
        actor="alice",
        occurred_at_offset_days=-6,
        role="distractor",
    ),
    FixtureEvent(
        event_id="github/pr/1040/merged",
        source="github",
        kind="pr_merged",
        body=(
            "[PR #1040 merged by bob]\n"
            "Title: inventory-svc: fix off-by-one in stock decrement\n"
            "Files: apps/inventory/services/stock.py"
        ),
        actor="bob",
        occurred_at_offset_days=-3,
        role="distractor",
    ),
    # Distractor Linear tickets on other features
    FixtureEvent(
        event_id="linear/INV-92/closed",
        source="linear",
        kind="issue_closed",
        body=(
            "[Linear INV-92 closed by bob]\n"
            "Title: inventory consumer batch size tuning\n"
            "Resolution: Implemented in PR #1037."
        ),
        actor="bob",
        occurred_at_offset_days=-4,
        role="distractor",
    ),
    # Distractor deploys of unrelated services
    FixtureEvent(
        event_id="deploy/inventory/staging/v3.2.1",
        source="deploy",
        kind="deployment",
        body=(
            "[ArgoCD deploy] inventory-svc v3.2.1 → staging. Triggered by PR #1037."
        ),
        actor="argo-bot",
        occurred_at_offset_days=-4,
        role="distractor",
    ),
    # Near-miss discussion (same channel, different topic)
    FixtureEvent(
        event_id="slack/eng-platform/redis-eviction",
        source="slack",
        kind="discussion",
        body=(
            "[Slack #eng-platform thread, 2026-05-10]\n"
            "carol: noticed redis cluster eviction rate ticked up in prod overnight.\n"
            "alice: ttl tuning? not the pool issue from OPS-218 — different system."
        ),
        actor="carol",
        occurred_at_offset_days=-10,
        role="distractor",
    ),
    # Adversarial: an OLD ADR that's now superseded by ADR-007 — agent should
    # surface ADR-007 (the current rule), NOT this older version.
    FixtureEvent(
        event_id="adr/003-superseded",
        source="repo-docs",
        kind="doc_created",
        body=(
            "# ADR-003: Allow untyped dict returns for prototypes\n\n"
            "## Status: SUPERSEDED by ADR-007 (2026-04 retired this stance).\n\n"
            "Original (now-incorrect) guidance: handlers prototyping new endpoints may\n"
            "return bare dicts during initial development. Promote to response_model\n"
            "before merging to main."
        ),
        actor="alice",
        occurred_at_offset_days=-300,
        role="distractor",
    ),
]


# ---------------------------------------------------------------------------
# Scenarios — agent queries with expected outcomes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioExpectation:
    """Ground truth for one scenario, used to score the engine's response.

    `must_surface_event_ids` is the recall set: these event_ids must appear in
    `source_refs` or evidence of the response.

    `must_not_surface_event_ids` is the precision set: any of these in the
    response is a precision regression.

    `must_mention_phrases` is a coarse synthesis check: does the answer text
    contain these key phrases? Lowercase substring match.

    `must_not_mention_phrases` is a hallucination check.

    `expected_confidence` is the coarse label we'd expect from D3 derivation:
    high/medium/low/unknown.
    """

    must_surface_event_ids: set[str]
    must_not_surface_event_ids: set[str]
    must_mention_phrases: list[str]
    must_not_mention_phrases: list[str] = field(default_factory=list)
    expected_confidence: str = "medium"


@dataclass
class Scenario:
    scenario_id: str
    dimension: Literal["PREF", "INFRA", "TIME", "BUG"]
    query_intent: str
    query_scope: dict[str, str | int]
    query_text: str
    expected: ScenarioExpectation


SCENARIOS: list[Scenario] = [
    Scenario(
        scenario_id="pref/auth-handler",
        dimension="PREF",
        query_intent="feature",
        query_scope={
            "repo": "acme/platform",
            "file_path": "app/api/v1/auth.py",
            "language": "python",
        },
        query_text="I'm writing a new FastAPI handler for the /sessions endpoint in app/api/v1/auth.py.",
        expected=ScenarioExpectation(
            must_surface_event_ids={
                "adr/007",            # response_model rule
                "pr/review/881",      # AcmeError rule
                "doc/conventions/logging",  # structlog rule
            },
            must_not_surface_event_ids={
                "adr/003-superseded",  # the OLD rule that should not surface
                "github/pr/1037/merged",  # inventory PR — unrelated
                "github/pr/1040/merged",  # inventory PR — unrelated
            },
            must_mention_phrases=["response_model", "AcmeError", "structlog"],
            must_not_mention_phrases=["bare dict"],  # the old superseded guidance
            expected_confidence="high",
        ),
    ),
    Scenario(
        scenario_id="infra/auth-prod-deps",
        dimension="INFRA",
        query_intent="debugging",
        query_scope={"service": "auth-svc", "environment": "prod"},
        query_text="What does auth-svc depend on in prod?",
        expected=ScenarioExpectation(
            must_surface_event_ids={
                "k8s/auth/prod",       # prod manifest
                "codeowners/auth",     # owner info
            },
            must_not_surface_event_ids={
                "k8s/auth/staging",    # staging — wrong environment
                "adr/inventory-kafka", # inventory — wrong service
                "github/pr/1037/merged",  # unrelated PR
            },
            must_mention_phrases=["postgres-auth-prod"],
            must_not_mention_phrases=["postgres-auth-staging", "orders-topic"],
            expected_confidence="high",
        ),
    ),
    Scenario(
        scenario_id="time/auth-recent-changes",
        dimension="TIME",
        query_intent="debugging",
        query_scope={"service": "auth-svc"},
        query_text="What changed in auth-svc in the last 7 days?",
        expected=ScenarioExpectation(
            must_surface_event_ids={
                "github/pr/1042/opened",
                "github/pr/1042/merged",
                "deploy/auth/staging/v2.1.4",
                "linear/AUTH-42/closed",
            },
            must_not_surface_event_ids={
                # Distractor PRs in OTHER services
                "github/pr/1037/merged",
                "github/pr/1038/merged",
                "github/pr/1040/merged",
                "deploy/inventory/staging/v3.2.1",
                # Out-of-window events
                "alert/ops-218",
                "github/pr/998/merged",
                "notion/postmortem/2026-04-12",
            },
            must_mention_phrases=["PR #1042", "rate limit"],
            must_not_mention_phrases=["inventory", "checkout-api"],
            expected_confidence="high",
        ),
    ),
    Scenario(
        scenario_id="bug/auth-latency-recurrence",
        dimension="BUG",
        query_intent="debugging",
        query_scope={"service": "auth-svc"},
        query_text=(
            "auth-svc latency suddenly spiked at 03:00 UTC, looks like database "
            "connection pool exhaustion. Have we seen this before, and what was "
            "the fix?"
        ),
        expected=ScenarioExpectation(
            must_surface_event_ids={
                "alert/ops-218",
                "github/pr/998/merged",
                "notion/postmortem/2026-04-12",
                "slack/eng-platform/pool-pattern",
            },
            must_not_surface_event_ids={
                "slack/eng-platform/redis-eviction",  # near-miss distractor
                "github/pr/1037/merged",
                "adr/007",                            # PREF-only, not BUG
            },
            must_mention_phrases=["pool", "OPS-218", "PR #998"],
            must_not_mention_phrases=["redis eviction"],
            expected_confidence="high",
        ),
    ),
]


def all_events() -> list[FixtureEvent]:
    """All events to ingest, ordered by occurred_at (oldest first).

    Universe first (so identity-resolution has the canonical names),
    then signal + distractors interleaved by time (the realistic case)."""
    universe = sorted(UNIVERSE_SEED, key=lambda e: e.occurred_at_offset_days)
    others = sorted(
        SIGNAL_EVENTS + DISTRACTOR_EVENTS, key=lambda e: e.occurred_at_offset_days
    )
    return universe + others

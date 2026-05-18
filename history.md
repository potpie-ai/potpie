# Logging Audit — History

Running log of work done on the logging audit. Newest entries at the bottom.

## 2026-05-15 — Audit kickoff

**Goal:** Audit the backend logging — volume, whether logs are meaningful or
noise, what structure exists, and which libraries/mechanisms are in use.

**Setup done:**
- Switched backend (`/Users/yashkrishan/Desktop/workspace/test/potpie`) and
  frontend (`.../potpie/potpie-ui/potpie-ui`) to `main`.
- Discarded a local `uv.lock` change to allow the backend switch (user-approved).

**Initial recon:**
- Logging-related deps in `pyproject.toml`: `loguru>=0.7.3`,
  `logfire[litellm]>=3.2.0`, `sentry-sdk[fastapi]>=2.43.0`.
- 361 Python files under `app/`.
- Central logger module: `app/modules/utils/logger.py` (loguru-based).
  - Sensitive-data redaction via regex (`SENSITIVE_PATTERNS`) applied in both
    dev and prod sinks.
  - Stdlib `logging` intercepted and routed through loguru
    (`InterceptHandler`).
  - Prod sink emits flat JSONL to stdout; dev sink is colorized human format.
  - Per-library level map (uvicorn/sqlalchemy/httpx/etc. dialed down).
  - `log_context()` context manager + `setup_logger(name)` factory.
- Observability stack: `logfire` (tracing) + `sentry-sdk` (errors), both
  initialized in `app/main.py`.

**Status:** Infrastructure mapped. Next: quantify volume and assess quality.

## 2026-05-15 — Findings

### Volume (app/, ~361 .py files)

~3,018 explicit `logger.*` calls + 20 stdlib `logging.*` + 10 `print()`.

| Level | Count | Share |
| --- | --- | --- |
| info | 1296 | 43% |
| warning | 606 | 20% |
| error | 473 | 16% |
| exception | 318 | 11% |
| debug | 325 | 11% |
| critical | 0 | 0% |

Top emitters: `parsing/graph_construction/parsing_helper.py` (163),
`integrations/integrations_router.py` (119),
`conversations/conversation/conversation_service.py` (102),
`repo_manager/repo_manager.py` (97), `integrations/integrations_service.py` (97).

Logger sourcing: 158 files use the central `setup_logger` (good adoption),
17 use stdlib `logging.getLogger` directly, 1 imports loguru directly.

### Infrastructure (solid — above average)

- Single well-designed loguru module (`app/modules/utils/logger.py`):
  regex secret redaction in both sinks, stdlib interception, JSONL in prod,
  colorized dev format, per-library level map, `log_context()` +
  `setup_logger()`.
- `configure_logging()` runs at import time in `app/main.py:53` (before app /
  Sentry / logfire init — correct ordering). Sentry = errors, logfire =
  tracing.

### Quality problems (the real issues — usage, not infra)

1. **Structure largely bypassed.** 748 f-string log calls (~25%) vs only ~73
   structured-kwarg (~2.4%) and 29 `%`-lazy. Prod sink is JSON, but f-strings
   pre-render the message into an opaque blob — cannot query/aggregate by
   field. Biggest "structure" gap.
2. **Correlation IDs not propagated.** `log_context()` used in only 7 files /
   7 call sites. The other ~3,000 log points have no
   conversation_id/run_id/user_id, so a single run can't be traced
   end-to-end — directly undercuts the debuggability goals in
   `docs/queued-task-flow.md`.
3. **Debug-grade noise ships at INFO in prod.** ≥14 lines like
   `logger.info("DEBUG: ...")`, `DEBUGNEO4J:`, `=== ... DEBUG ===`. Prod sink
   level is INFO so these always ship (clearest "random/doesn't-make-sense"
   finding). Files: `users/user_service.py`,
   `parsing/knowledge_graph/inference_service.py`,
   `parsing/graph_construction/parsing_service.py`,
   `intelligence/agents/chat_agents/pydantic_agent.py`,
   `integrations/integrations_router.py`.
4. **PII in logs.** ≥18 log lines interpolate user emails; redaction patterns
   cover credentials/tokens but NOT emails or user IDs. Emitted at INFO.
   Compliance smell.
5. **Double-logging on error paths.** ~168 sites do
   `logger.error/exception(...)` immediately followed by `raise`; the same
   error is logged again upstream / by Sentry. Explains the high
   error+exception ratio (~26%). Should log-or-raise, not both.
6. **`info` used where `debug` belongs.** debug only 11% vs info 43%; many
   info logs are step traces ("Found SSO user", "After update project
   status") that inflate prod volume.
7. Minor: 45 emoji-decorated messages (cosmetic, can break some parsers);
   10 `print()` (mostly benign alembic/CLI; `user_service` "Dummy user" and
   `parse_webhook_helper` Slack errors should be logged); 17 stdlib
   `getLogger` files are still intercepted via root but lose `bind(name)` so
   the `logger` field is the module path, not the intended name.

### Net assessment

Infrastructure is good; **usage is the problem**. "How much": a lot (~3k
sites, INFO-heavy). "Do they make sense": infra yes, usage inconsistent —
debug noise and PII at INFO, double-logged errors. "Structure": exists but
~25% are f-string blobs and field-level structure is almost never used.
"What we use": loguru (app) + logfire (tracing) + sentry (errors).

**Status:** First-pass audit complete. No code changes made yet.

## 2026-05-15 — Critical Sentry / observability deep-dive

Scope: backend only. Question raised: Sentry code exists but seems unwired —
should we keep Sentry for backend, or is there something better; what's left.

### Sentry: effectively non-functional today

- **Production-only gate.** `setup_sentry()` runs only when
  `ENV == "production"` (`app/main.py:76`). Staging/dev have zero error
  monitoring. (Celery queue prefix defaults to `staging`, so much real
  traffic is staging.)
- **Web-process only.** Sentry inits in `MainApp.__init__` (FastAPI). Celery
  workers (`app/celery/celery_app.py`) call `configure_logging()` + logfire
  but **never init Sentry**, and there is **no `CeleryIntegration`** anywhere.
  → background agent/parsing/regenerate failures (the bulk of real work per
  `docs/queued-task-flow.md`) are invisible to Sentry.
- **No loguru→Sentry bridge.** Sentry's `LoggingIntegration` taps stdlib
  `logging`; the app logs via loguru. The only bridge is stdlib→loguru
  (`InterceptHandler`), not loguru→Sentry. So the 473 `logger.error` + 318
  `logger.exception` deliberate calls never become Sentry events/breadcrumbs.
  Sentry only sees unhandled web exceptions (`FastApiIntegration`) + ~20 raw
  stdlib calls.
- **DSN unset in-repo.** `dsn=os.getenv("SENTRY_DSN")`; no `.env`/infra/k8s
  sets it (no k8s dir in this repo at all). `sentry_sdk.init(dsn=None)` is a
  silent no-op → Sentry likely does literally nothing today unless the deploy
  platform injects DSN out-of-repo.
- **`profiles_sample_rate=1.0`** → 100% profiling overhead in prod (cost/perf
  smell). `traces_sample_rate=0.25`.
- Note: `sentry_oauth_v2.py` / integrations Sentry code is a **product
  feature** (users linking their own Sentry org), unrelated to app
  self-monitoring — do not conflate.

### What's left / additional gaps

- **No log aggregation defined in-repo.** Prod emits clean JSONL to stdout
  (good) but nothing says where it goes — no Datadog/Fluentbit/Vector/Loki/
  CloudWatch/OTel-collector config, no k8s manifests here. The ship→index→
  retain→search→alert half of "structure" is undefined. JSONL to stdout is
  inert without a consumer.
- **Celery correlation gap.** `LoggingContextMiddleware`
  (`app/modules/utils/logging_middleware.py`) auto-injects
  request_id/user_id — but it is ASGI middleware, **web requests only**.
  Celery tasks get no automatic correlation context; only the 7 manual
  `log_context()` sites. Background runs (where debugging matters most) have
  the weakest correlation.
- **logfire not bridged to loguru.** `logfire.configure()` runs in both web
  and Celery, but `logfire.instrument_loguru()` is never called → loguru logs
  don't reach logfire either. logfire only gets LiteLLM + Pydantic AI spans.

### Verdict

Sentry is the right tool for *errors*, the wrong tool for *logs* — and is
currently functioning as neither. The core problem is **three half-wired
tools (loguru, logfire, Sentry) with no bridges between them**, not the choice
of Sentry. Two coherent paths recorded for the user:

- **Path A (low churn, recommended):** Sentry = errors only — init in Celery
  too (+`CeleryIntegration`), add a loguru ERROR+ sink → `capture_exception`,
  ungate from prod-only, require/validate `SENTRY_DSN`, drop
  `profiles_sample_rate` to ~0.05. Logs = the existing JSONL→stdout; pick &
  configure the aggregator (the actually-missing piece). logfire = traces
  (+optional `instrument_loguru`).
- **Path B (consolidate):** make logfire (already a paid OTel-native dep) the
  single logs+traces backend via `instrument_loguru()`, drop or thin Sentry.
  Fewer parts, one correlation story; weaker error-grouping/alerting than
  Sentry.

**Status:** Critical deep-dive complete. No code changes. Awaiting decision on
Path A vs B before any remediation.

## 2026-05-15 — Hexagonal context-engine logging pattern

User asked to model the logging "module" on the hexagonal context folder.

### Location / branch reality

- Two candidates: `app/modules/context_graph` (the old module) and
  `app/src/context-engine` (the hexagonal one: adapters/application/domain/
  ports/bootstrap).
- **Neither exists on `main`** — only stale `__pycache__`. The full
  hexagonal source (172 .py) lives on branch **`feat/context-engine`**.
  Inspected via `git show`, did not switch branches.

### How context-engine actually does logging

- **No `LoggerPort`, no logging adapter, no logging module of its own.** The
  `domain/ports` list has no logger port; there is zero logging-config code
  anywhere in the tree (no `basicConfig`/`dictConfig`/`configure_logging`/
  `setup_logger`/handlers).
- Every layer (use_cases, application services, in/outbound adapters, http)
  uses plain stdlib: `import logging` → `logger = logging.getLogger(__name__)`.
- Formatting is consistent **%-style lazy** —
  `logger.exception("%s failed: %s", name, exc)`,
  `logger.warning("reset_pot structural: %s", exc)` — i.e. structure-
  preserving, the opposite of the monolith's 748 f-string logs.
- The core (domain/application) depends only on the **stdlib `logging`**
  facility — zero coupling to loguru/logfire/sentry.

### Interpretation

This is an intentional hexagonal stance: logging is an **ambient
cross-cutting capability, not a domain port**, so it is *not* wrapped in a
port/adapter (that would be over-engineering). The **composition root owns
the sink**: inside the monolith, `configure_logging()`'s root
`InterceptHandler` (installed via `logging.basicConfig(force=True)`) captures
these stdlib logs and routes them into loguru → JSON/redaction. So
"setup logging like the hexagonal repo" = **don't build a logging
module/port; standardize on `getLogger(__name__)` + %-lazy formatting +
configure-at-the-edge.**

### The one real gap

context-engine has no logging bootstrap of its own, so it only gets proper
sink/format/redaction *because the monolith configures the root logger*. Run
**standalone** (its CLI / `standalone_container.py`) nothing configures
logging → Python default (root=WARNING, no JSON, no redaction); INFO/DEBUG
silently dropped. A true standalone hexagon needs a tiny logging-bootstrap
call in its standalone composition root (`bootstrap/standalone_container.py`
or `cli/main.py`) — not in the core.

### Recommendation (ties back to the earlier audit)

Adopt context-engine's convention monolith-wide: `logging.getLogger(__name__)`
+ %-lazy everywhere, keep `configure_logging()` as the single
composition-root adapter owning sink/format/redaction/Sentry-bridge. This
simultaneously fixes the earlier findings — kills f-string structure loss,
decouples app code from loguru, and makes the loguru→Sentry bridge a single
edge concern instead of 3,000 scattered ones.

**Status:** Investigation complete. No code changes.

## 2026-05-15 — Requirement clarified: portable logging library

Key context shift from the user:

- The current codebase is **mid-refactor and may be discarded** — do not
  invest in monolith-coupled logging.
- Goal is a **standalone, reusable logging module pluggable into any
  codebase/service** (refactored app, context-engine standalone, new
  services).

This overrides the earlier "flat in-app package" recommendation. For a
*reusable library* (vs in-app module), exactly **one abstraction seam** is
justified — a `Sink` Protocol — but not a full hexagon.

Agreed design principles (recorded for continuity):

1. Standalone installable package — own `pyproject.toml`, versioned, **no
   `app.*` imports**. Portability comes from packaging + dep hygiene, not
   folder layout.
2. Core targets **stdlib `logging`** (ambient `getLogger(__name__)`), not
   loguru. loguru/structlog/logfire become optional sink adapters.
3. **One port**: a `Sink`/handler `Protocol` + registry. Everything else =
   plain functions.
4. Explicit `LoggingConfig` dataclass at composition root; env-loading is one
   optional adapter.
5. Framework integrations (FastAPI/Celery/Sentry/loguru) = optional extras,
   lazily imported. Core has near-zero deps.
6. Composition profiles per runtime: monolith / celery / standalone /
   lib-default.

Proposed package skeleton: `obslog/` (frozen public API in `__init__`:
`get_logger`, `configure`, `log_context`; `sink.py` = the one Protocol;
`sinks/` plug-ins; `integrations/` optional; `profiles.py` wiring).

**Open decisions blocking scaffold:** (1) distribution model — separate repo
+ private index vs monorepo path-package vs git submodule; (2) core backend —
stdlib-only core w/ loguru optional sink (recommended) vs loguru as hard core
dep.

**Status:** Design agreed in principle; awaiting the two decisions above
before scaffolding. No code changes.

## 2026-05-15 — Constraints locked: in-repo, unify logs+Sentry+logfire

User decisions:

- **In-repo, not a separate repo.** Lives inside the current repo as a
  self-contained path-package. Reconciliation with the "survives the rewrite"
  requirement: it must stay **`app.*`-free** so it can be lifted wholesale
  into the refactored codebase / other services.
- **One module owns all three:** application logging + **Sentry** (errors) +
  **logfire** (tracing). This is an *observability* module, not just logging.
  It subsumes today's scattered `logfire_tracer.py` + `main.py` Sentry init +
  `app/modules/utils/logger.py`.

Design implications:
- stdlib-core + single `Sink` Protocol is now clearly correct: ≥3 backends
  (loguru sink, Sentry sink/bridge, logfire log+trace) must coexist behind
  one seam.
- Public surface stays tiny and frozen: `get_logger()`, `configure()`,
  `log_context()` (correlation IDs + active trace/span binding).
- One composition root per runtime initializes loguru sink + Sentry
  (web+celery+standalone, with CeleryIntegration) + logfire instrumentation —
  fixing the audit's prod-only / web-only / no-bridge gaps in one place.
- Recommended placement: `app/src/observability/` (mirrors
  `app/src/context-engine` — signals "standalone src package", physically
  liftable).

**Remaining micro-decision:** stdlib-only core with loguru/Sentry/logfire as
optional sink-adapters (recommended) vs loguru as a hard core dep (makes
Sentry/logfire second-class). Proceeding on the recommendation unless
redirected.

**Status:** Constraints locked. Next: confirm placement + scaffold the
`app/src/observability/` package skeleton (contract only, no behavior).

## 2026-05-15 — Plan agreed, broken into tickets, reviewed

5-phase plan agreed (scaffold → port current → add Sentry/Celery/standalone →
migrate callers → cutover/cleanup). Placement `app/src/observability/`
(own pyproject, app.*-free, liftable). Scaffold dirs created then halted at
user request to plan first; **no code/files written** (empty dirs only,
populated only post-approval).

Plan converted to tickets POT-1250(?)–1254 (+ unrelated POT-1228 blog post).
Reviewed — flagged: (1) POT-1228 doesn't belong; (2) first ticket unnumbered;
(3) priority inversion — POT-1252 (the actual Sentry/Celery fix, the
audit's core value) is Medium but should be High vs the High plumbing
tickets; (4) POT-1253 (migrate ~158 callers, biggest blast radius) given only
3 days after a Medium dependency — underscoped, widen/split; (5) **no ticket**
for the original audit remediation (PII/DEBUG-at-INFO/log-and-raise/f-strings,
compliance-relevant) nor for the log-aggregation-backend decision — both
untracked gaps not solved by this package.

**Open blocker for Phase 1:** target branch not yet decided.

**Status:** Awaiting ticket fixes + branch decision before Phase 1 scaffold.

## 2026-05-15 — Phase 1 scaffold DONE (contract-only)

Branch: **`feat/observability-package`** (off `main`; chosen since user said
"start" without specifying — reversible, keeps main clean).

Created `app/src/observability/` — pyproject.toml + 16 contract-only modules
(public API, config dataclasses, Sink Protocol, stubbed redaction/intercept/
tracing/sinks/integrations/profiles). Verified: zero `app.*` imports; all 16
modules import; **no loguru/sentry_sdk/logfire/starlette/celery pulled at
import** (lazy, correct); public API re-exports work; stubs raise
NotImplementedError; `profiles_sample_rate` defaulted to 0.05 (fixes audit
cost smell).

### Edge cases found & encoded in the contract (EC1–EC4 in __init__.py)

- **EC1 stdlib rejects kwargs.** stdlib loggers only accept exc_info/
  stack_info/stacklevel/extra. 73+ existing `logger.info("m", k=v)` sites
  would break on migration → `get_logger` returns a `StructuredLogger`
  adapter mapping `**fields -> extra`. Load-bearing contract decision.
- **EC2 configure() idempotent + fork-safe.** No `basicConfig(force=True)`
  (nukes handlers, breaks idempotency) → tag-based handler replace. Network
  sinks (Sentry/logfire) MUST init in Celery `worker_process_init`, not at
  import (prefork breaks pre-fork sockets).
- **EC3 contextvars don't cross hops.** Correlation IDs lost across
  threadpool/process/Celery boundaries unless re-bound per integration —
  this is the *root cause* of the audit's weak queued-run correlation.
- **EC4 pre-configure logs.** Phase 2 must install a minimal safety handler
  so logs before configure() aren't dropped; Phase 1 remains contract-only.
- **Module shadowing** (caught during impl): `loguru.py`/`sentry.py`/
  `logfire.py` would shadow the real packages → renamed `*_sink.py`.
- **Sentry double-capture**: our Sentry sink + Sentry's LoggingIntegration →
  init with `event_level=None`.
- **logfire single configure()**: tracing.py and logfire_sink.py must share
  one `logfire.configure()`.
- **Disabled-state**: missing DSN/token must emit ONE visible notice, never
  the audit's silent no-op.
- **Streaming/SSE**: request context must stay bound for the whole stream
  (codebase streams chat) — middleware can't unbind at header flush.
- **Verification bug, fixed**: public re-exports were under `TYPE_CHECKING`
  only → runtime ImportError; moved to real imports (no cycle).

### Gaps surfaced (no ticket — flagged earlier, still untracked)

- Where prod JSONL ships (aggregation backend) — not solved here.
- Audit remediation (email/PII at INFO, DEBUG-at-INFO, 168 log-and-raise,
  748 f-strings) — separate effort.
- Email/PII redaction pattern: keep deferred to Phase 2 (noisy vs caller
  responsibility) — decision pending.
- Root `pyproject` `packages.find include=["app*"]`: safe because `app/` has
  no `__init__.py`, so setuptools won't traverse into `app/src`; nested
  package won't clash with root build. Recommend an explicit exclude later
  for defensiveness.

**Status:** Phase 1 complete & verified on `feat/observability-package`. No
existing code touched. Ready for Phase 2 (port current components) — pending
go-ahead + the ticket fixes flagged previously.

## 2026-05-15 — Phase 1 PR opened

Committed 18 files (17 scaffold + history.md, +964) as
`feat(observability): scaffold liftable observability package (Phase 1)`,
pushed `feat/observability-package`, opened **PR #785** (base `main`):
https://github.com/potpie-ai/potpie/pull/785

Pre-existing unrelated untracked files (.tmp-*, DATA_STORAGE_INVENTORY.md,
mermaid_sanitizer.py, vulnerability-assessment-report.md, docs/, tests/unit/)
deliberately excluded — staged only `app/src/observability` + `history.md`.

**Status:** PR #785 open for review. Phase 2 pending go-ahead + ticket fixes.

## 2026-05-18 — PR #785 review + critical cleanup

Fetched PR #785 locally as `pr-785-review`; it points at the same commit as
`feat/observability-package` (`15588d2e`).

### Review findings

- **Critical: PR scope drift.** The PR diff included unrelated root
  `pyproject.toml`, `uv.lock`, and `scripts/run_tests.sh` changes despite the
  recorded Phase 1 intent being contract-only observability scaffold +
  `history.md`. The lockfile change pulled in unrelated dependency metadata
  (`coverage`, `pytest-cov`, extra `torch` wheels), widening review and merge
  risk for no observability value.
- **Contract doc mismatch.** `observability/__init__.py` claimed the package
  installed a pre-configure safety handler at import, but Phase 1 is explicitly
  stub/contract-only. Fixed the contract wording so Phase 2 owns that behavior
  instead of implying existing runtime behavior.
- **False local Python failure.** macOS `python3` is 3.9.6 and fails on
  `dataclass(slots=True)`, but the repo-managed interpreter is Python 3.13.9
  via `uv run`; root project requires `>=3.10,<3.14`, so no package change was
  needed.

### Fixes made

- Restored `pyproject.toml` and `uv.lock` to the PR merge-base
  (`4cc4dfdd...`) to remove the accidental root dependency/coverage churn from
  PR #785.
- Deleted unrelated `scripts/run_tests.sh`.
- Updated `app/src/observability/observability/__init__.py` docs:
  behavioral public API vs data/typing re-exports is now explicit, EC4 no
  longer falsely says a handler is installed in Phase 1, and
  `StructuredLogger.process` no longer says the kwarg mapping is already
  implemented.

### Verification

- `PYTHONPATH=app/src/observability uv run python -m compileall -q app/src/observability/observability`
  passed.
- Imported public API + all sink/integration modules successfully under
  `uv run python`.
- Verified lazy optional deps: importing the scaffold did **not** import
  `loguru`, `sentry_sdk`, `logfire`, `starlette`, or `celery`.
- `rg -n "app\\." app/src/observability/observability app/src/observability/pyproject.toml`
  found only documentation text, no `app.*` imports.

**Status:** Critical PR-scope issue fixed locally; scaffold contract review
passes. Remaining known gaps are still Phase 2+ / separate tickets:
aggregation backend decision, audit remediation (PII, DEBUG-at-INFO,
log-and-raise, f-string blobs), and actual implementation of the contract.

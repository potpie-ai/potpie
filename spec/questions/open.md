---
id: SPEC-QUESTIONS
title: Context Runtime Open Questions
kind: open-question-log
owners:
  - team:potpie
---

# Context Runtime Open Questions

Three questions remain intentionally deferred and are not referenced by active
behavior nodes. Nine implementation-readiness questions are resolved by
ADR-0007 through ADR-0009. Each deferred question becomes a separate decision
before an implementation commit depends on its answer.

OQ-CE-API-001 [resolved]: What exact operation groups and synchronous or asynchronous surfaces should the public ContextEngine façade expose?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0008
  owner: team:potpie
  option: asynchronous façade with explicit sync adapters outside the engine
  option: paired synchronous and asynchronous façade methods
  option: operation objects with one canonical execution model
  decision-trigger: before proposing the public ContextEngine API implementation contract
  resolution: ADR-0008

OQ-RM-LIFETIME-001 [resolved]: What reuse, eviction, and idle-shutdown policy should govern Resource Manager engine and resource lifetimes?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0009
  owner: team:potpie
  option: retain compatible engines until daemon shutdown
  option: reference-count and close when the last request releases the context
  option: bounded cache with explicit idle eviction
  decision-trigger: before implementing engine reuse, eviction, or idle shutdown; borrowed and transferred ownership are already fixed
  resolution: ADR-0009

OQ-DAEMON-IMPLEMENTATION-001 [resolved]: Which existing daemon stack supplies reusable implementation and which stack is removed first?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0007
  owner: team:potpie
  option: replace reflection in the active launcher and remove the candidate runtime
  option: complete the candidate runtime and migrate the launcher and client to it
  option: extract minimal shared primitives and replace both stacks
  decision-trigger: before the first daemon implementation commit
  resolution: ADR-0007

OQ-DAEMON-CONTROLLER-001 [resolved]: What exact controller API and operating-system supervisor integration should implement daemon creation, observation, status, and failure reporting?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0009
  owner: team:potpie
  option: Potpie controller directly starts and observes the foreground process
  option: Potpie controller delegates physical creation to a selected operating-system supervisor adapter
  option: one controller interface with direct-process and supervisor-backed implementations
  decision-trigger: before implementing CLI-driven daemon startup or operating-system supervisor integration
  resolution: ADR-0009

OQ-DAEMON-TRANSPORT-001 [resolved]: Which local transport and platform fallback should carry typed daemon operations?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0009
  owner: team:potpie
  option: Unix domain socket with loopback TCP fallback
  option: loopback TCP on every platform
  option: platform-native local IPC adapters behind one client contract
  decision-trigger: before defining discovery fields or transport adapters
  resolution: ADR-0009

OQ-DAEMON-PROTOCOL-001 [resolved]: What versioned request, response, handshake, and error envelope should the typed daemon protocol use?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0009
  owner: team:potpie
  option: one versioned envelope with discriminated operation payloads
  option: per-operation HTTP routes with shared version and error headers
  option: generated local RPC schema with explicit compatibility rules
  decision-trigger: before implementing the typed client or daemon operation registry
  resolution: ADR-0009

OQ-DAEMON-CANCEL-001 [deferred]: What cancellation, timeout, disconnect, and unknown-outcome semantics should each operation category expose?
  > decision [active]: decision:ADR-0006
  owner: team:potpie
  option: best-effort cancellation with explicit unknown mutation outcome
  option: cancellable reads and queued work with non-cancellable committed mutations
  option: operation handles with separate cancellation and status requests
  decision-trigger: before exposing cancellation or timeout in the typed protocol
  resolution: null

OQ-DAEMON-IDEMPOTENCY-001 [deferred]: Which operation categories are idempotent and what retry identity should safe mutations use?
  > decision [active]: decision:ADR-0006
  owner: team:potpie
  option: explicit per-mutation idempotency keys
  option: operation-specific natural identity where available
  option: no automatic mutation retry until each operation is separately contracted
  decision-trigger: before implementing client or daemon mutation retries
  resolution: null

OQ-DAEMON-CONCURRENCY-001 [resolved]: Which typed operation-safety classifications and conflict keys determine concurrency for handlers, contexts, resources, and daemon lifecycle?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0009
  owner: team:potpie
  option: declarative shared-read, context-mutation, resource-lifecycle, and daemon-lifecycle safety metadata on each typed handler
  option: explicit conflict keys derived from typed operation, context identity, and resource identity
  option: one central compatibility table keyed by typed operation identifiers
  decision-trigger: before implementing replacement daemon request coordination or removing the global RPC mutex
  resolution: ADR-0009

OQ-DESTRUCTIVE-INTENT-001 [resolved]: What exact type, fields, lifetime, signing, and replay protection should represent the already-fixed destructive-intent assertion and validation flow?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0009
  owner: team:potpie
  option: short-lived signed intent bound to actor operation and context
  option: authenticated request field validated against operation and resolved context
  option: operation-specific confirmation types without one universal token
  decision-trigger: before implementing a destructive typed operation
  resolution: ADR-0009

OQ-CLI-JSON-001 [deferred]: What exact machine JSON envelope and complete numeric exit-code mapping should the CLI publish?
  > decision [active]: decision:ADR-0006
  owner: team:potpie
  option: one shared success and error envelope across all commands
  option: typed per-command result payloads with one shared error envelope
  option: versioned machine contract independent of human command presentation
  decision-trigger: before changing existing machine output or exit mappings
  resolution: null

OQ-COMPAT-001 [resolved]: What sequence and temporary shims should perform the required removal of Context Core, HostShell, reflection RPC, the unused daemon stack, and duplicate discovery formats while preserving supported callers?
  > decision [superseded]: decision:ADR-0006
  > decision [active]: decision:ADR-0007
  owner: team:potpie
  option: staged compatibility shims removed within the same pull request
  option: additive typed boundary followed by one deprecation release
  option: coordinated breaking release with import and command migration guidance
  decision-trigger: before the first compatibility or deletion commit; the deletion end state is not deferred
  resolution: ADR-0007

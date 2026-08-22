---
id: SPEC-GLOSSARY
title: Context Runtime Glossary
kind: glossary
revision: 1
maturity: accepted
owners:
  - team:potpie
depends_on: []
change_ref: SPEC-CHANGE-0002
---

# Context Runtime Glossary

## Purpose

These definitions give the Context Engine, Potpie Resource Manager, daemon, and
CLI contracts one vocabulary. A more specific contract can narrow a term for
its boundary but cannot silently change its canonical meaning.

Only the behavior nodes in this document are normative. The term descriptions
explain those behaviors and the contracts that depend on them.

## Terms

### Context

The logically isolated body of source-derived knowledge, state, and domain
operations addressed through one context identity.

### Context identity

The logical, stable identity bound to one Context Engine instance. It is not a
pot display name, current CLI selection, storage handle, process identifier, or
daemon instance identity.

### Context Engine

The importable library that owns context-domain semantics and invariants while
depending on explicitly supplied host capabilities.

### `ContextEngine`

The finite, explicit public façade through which a compatible host invokes
context-domain use cases for one bound context identity. It is not a service
locator or a product-runtime object graph.

### Host

Software that constructs Context Engine, supplies its dependencies, and owns
concerns outside the context-domain boundary.

### Host-composed dependency

A capability, adapter, resource handle, policy, or configuration value selected
and supplied explicitly by a host rather than discovered or provisioned
implicitly by Context Engine.

### Borrowed dependency

A host-composed dependency whose ownership remains with the host. Context
Engine can use and detach from it but does not close or release the underlying
host resource.

### Transferred dependency

A host-composed dependency whose ownership is explicitly transferred through a
typed construction contract. Transfer is never inferred from the presence of a
cleanup method.

### Pot

Potpie's product-level record for selecting and managing a context. A pot can
carry names, source associations, and resource configuration without becoming
the Context Engine identity-selection mechanism.

### Context-selection request

Typed product input containing explicit selectors and permitted environmental
hints that Potpie resolves to one context identity before engine access.

### Authorized context lease

A scoped Potpie capability containing one resolved context identity, a
context-bound `ContextEngine`, validated authorization scope, and explicit
resource-ownership information. It grants access only for its declared scope
and has an explicit release operation. This is a conceptual contract term, not
a frozen Python class or wire type.

### Potpie Resource Manager

The logical Potpie subsystem that resolves context selection, applies
authorization policy, acquires context resources, constructs or reuses a
context-bound engine, and returns an authorized context lease. It is not a
product-operation dispatcher or service locator. The name describes a focused
responsibility and does not require one class or package.

### Potpie daemon subsystem

The Potpie-owned process boundary consisting of a narrow controller or launcher
outside the daemon process and one foreground daemon runtime.

### Daemon controller

The external control role outside the daemon process that causes creation and
observes the foreground runtime. The canonical CLI-managed path fulfills this
role with the Potpie controller; physical process creation can be delegated to
a selected operating-system supervisor adapter. The controller does not
establish readiness or execute product-domain operations.

### Daemon runtime

The long-lived foreground Potpie process that hosts typed operation handlers
and one logical Resource Manager.

### Daemon instance identity

A unique value generated for one daemon boot and returned by its live
handshake. It is distinct from process identifiers and context identities.

### Discovery record

Atomically published connection metadata that claims to identify a daemon
instance in one Potpie runtime scope. The claim can be stale; record presence
does not establish liveness, readiness, or authorization.

### Authenticated readiness handshake

A live request that authenticates the caller, negotiates compatible protocol
semantics, and returns the expected daemon instance identity and ready state.

### Typed daemon client

The client-side Potpie API that exposes a finite set of typed control and
product operations without manufacturing methods through reflection.

### Typed operation

An explicitly named operation with defined request, result, and error types.
Its transport encoding is an implementation concern unless separately
contracted.

### Human CLI mode

The interactive presentation mode that can prompt, show progress, render
readable output, and explain remediation.

### Machine CLI mode

The non-prompting presentation mode intended for automation, with one complete
machine-readable result on standard output and diagnostics kept separate.

### Human confirmation

An explicit affirmative response or explicit non-interactive flag showing that
a person requested a destructive action. Confirmation is not authentication or
authorization.

### Destructive-intent assertion

Typed, initially untrusted request metadata recording an explicitly confirmed
destructive operation and target-selection input. It is a safety assertion, not
a credential or proof of authorization. Potpie validates it against the
authenticated actor, authorized operation, and resolved context before domain
execution.

### Authentication

The process of establishing the caller's identity.

### Authorization

The process of deciding whether an authenticated caller can perform a selected
operation on a resolved context.

### Domain result

A transport-neutral typed value produced by Context Engine domain behavior.

### SelectionError

A typed Potpie failure to resolve one context, including distinct unknown,
ambiguous, and administratively unavailable selection-record variants. Failure
to acquire resources for an already resolved context is instead a
`ResourceLifecycleError`.

### AuthenticationError

A typed rejection of caller identity or credentials. A malformed authentication
exchange is instead a `ProtocolTransportError`.

### AuthorizationError

A typed denial because an authenticated caller lacks permission for the
operation and resolved context.

### ResourceLifecycleError

A typed Potpie failure while composing, acquiring, validating, retaining, or
releasing host-managed resources, including a configured daemon process.

### DomainError

A typed Context Engine failure caused by domain input, state, invariant, or
capability semantics.

### DependencyError

A typed Context Engine boundary failure reported when an engine-owned port
cannot complete an operation. A failure while Potpie acquires the resource
before engine invocation is instead a `ResourceLifecycleError`.

### EngineLifecycleError

A typed Context Engine failure caused by use outside the valid lifetime of an
engine instance.

### ProtocolTransportError

A typed failure in connection, authentication-exchange framing, protocol
compatibility, request framing, or response delivery between a client and the
daemon runtime.

### DaemonInternalError

A typed, redacted failure caused by an unexpected defect within the daemon
runtime rather than a domain, dependency, resource, authentication,
authorization, or protocol outcome.

### PresentationError

A CLI-local failure while parsing input, prompting, formatting output, or
mapping a typed result to process behavior.

### Engine lifetime

The interval from construction of one context-bound Context Engine instance
through release of its engine-owned state.

### Resource lifetime

The interval during which Potpie-managed resources and dependency bindings are
available for one or more compatible engine lifetimes.

### Process lifetime

The interval from one daemon boot through its stopped state. Failed is a
non-ready cleanup state that transitions to stopped; it is not a reusable ready
runtime.

### Binding contract

An exact accepted contract revision identified by contract ID, integer revision,
and immutable Git ref.

### Conformance record

An immutable artifact that pins a binding contract and implementation ref,
records per-behavior claims and verification results, and cites reproducible
evidence.

## Normative Requirements

GLOSS-001 [active]: Contracts in this specification set MUST use accepted glossary terms with the meanings defined by this revision unless they explicitly introduce a narrower term.
  > authority [active]: user:dsantra

GLOSS-002 [active]: A context identity MUST remain logical and immutable for one Context Engine instance.
  > authority [active]: user:dsantra
  @ GLOSS-001

GLOSS-003 [active]: Human confirmation MUST be treated as evidence that a destructive action was explicitly requested.
  > authority [active]: user:dsantra
  @ GLOSS-001

GLOSS-004 [active]: A discovery record MUST be treated only as a connection claim.
  > authority [active]: user:dsantra
  @ GLOSS-001

GLOSS-005 [active]: A fully usable Context Engine MUST be publicly constructible and operable with explicit host-composed dependencies.
  > authority [active]: user:dsantra
  @ GLOSS-001

GLOSS-006 [active]: Destructive-intent assertion, authentication, authorization, and explicit destructive domain execution MUST be treated as separate concepts.
  > authority [active]: user:dsantra
  @ GLOSS-001
  @ GLOSS-003

GLOSS-007 [active]: SelectionError, AuthenticationError, AuthorizationError, ResourceLifecycleError, DomainError, DependencyError, EngineLifecycleError, ProtocolTransportError, DaemonInternalError, and PresentationError MUST remain distinct error categories.
  > authority [active]: user:dsantra
  @ GLOSS-001

GLOSS-008 [active]: A context identity MUST NOT be treated as a pot display name, current CLI selection, storage handle, process identifier, or daemon instance identity.
  > authority [active]: user:dsantra
  @ GLOSS-002

GLOSS-009 [active]: Human confirmation MUST NOT be treated as authentication.
  > authority [active]: user:dsantra
  @ GLOSS-003

GLOSS-010 [active]: Human confirmation MUST NOT be treated as authorization.
  > authority [active]: user:dsantra
  @ GLOSS-003

GLOSS-011 [active]: A discovery record MUST NOT be treated as evidence of daemon readiness.
  > authority [active]: user:dsantra
  @ GLOSS-004

GLOSS-012 [active]: A fully usable Context Engine MUST NOT imply automatic provisioning of host-managed machine resources.
  > authority [active]: user:dsantra
  @ GLOSS-005

## Acceptance Criteria

This revision is ready for acceptance when every contract uses the terms
consistently, the identity categories cannot be confused, the ten error
categories remain distinct, and confirmation cannot be mistaken for security
authority.

## Rationale

Most boundary failures in the current architecture come from one concept
standing in for several responsibilities. Precise terms assign those
responsibilities without inventing overlapping runtime layers.

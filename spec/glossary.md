---
id: SPEC-GLOSSARY
title: Potpie Glossary
kind: glossary
status: proposed
version: 0.1.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-PROCESS
related_specs:
  - SPEC-PRODUCT
  - SPEC-SYSTEM
  - SPEC-PACKAGE-BOUNDARY
related_decisions:
  - ADR-0002
affects:
  - SPEC-PRODUCT
  - SPEC-SYSTEM
  - SPEC-PACKAGE-BOUNDARY
open_questions: []
verification:
  code_status: not_applicable
  verified_commit: null
  verified_at: null
  verified_by: null
  behavior_scope: []
  evidence: []
  cross_spec_status: passed
  cross_spec_checked_against:
    - SPEC-PACKAGE-BOUNDARY
  drift_status: unverified
---

# Potpie Glossary

These terms are canonical within the spec tree.

## Terms

### Product distribution

The installable `potpie` package used by end users. It owns user-facing
processes, product settings, product workflows, presentation, and product
integrations.

### Context engine

The reusable `potpie-context-engine` distribution, imported through the
`potpie_context_engine` Python namespace. It owns context and graph behavior but
does not own product processes or user experience.

### Product service

A root-Potpie capability whose behavior depends on the installed product,
including account authentication, integration authentication, setup, skills,
installation, daemon lifecycle, product status, and product telemetry.

### Engine operation

A typed, transport-independent context-engine request and result exposed through
`EngineClient`.

### PotpieRuntime

The root product composition object. It exposes an `engine` client and sibling
product services without flattening them into one facade.

### EngineClient

The asynchronous public protocol implemented by both an in-process context
engine and the root-owned daemon client.

### LocalEngineClient

The root runtime adapter that invokes a `ContextEngine` in the current process.

### DaemonEngineClient

The root runtime adapter that invokes the same engine operations through the
versioned daemon RPC protocol.

### Product settings

Root-owned configuration covering product paths, persisted user choices,
runtime mode, credentials, presentation, and product integrations.

### Engine configuration

Explicit library configuration describing engine behavior and injected
capabilities. It does not infer product paths or read product credentials.

### Persistent engine

An engine created with an explicit data directory whose state outlives the
current process.

### In-memory engine

An engine whose state is process-local and whose construction does not access
the user's home directory.

### Compatibility alias

An old import, command, RPC method, facade, or executable retained only to route
to a replacement. The package-boundary migration deliberately does not provide
these aliases.

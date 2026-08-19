---
id: SPEC-PRODUCT
title: Context Runtime Product Contract
kind: product-spec
revision: 1
maturity: accepted
owners:
  - team:potpie
depends_on:
  - SPEC-GLOSSARY
change_ref: SPEC-CHANGE-0003
---

# Context Runtime Product Contract

## Purpose

Potpie provides a managed local product experience around an independently
usable Context Engine. This contract defines the product outcomes that the
system and module boundaries preserve during migration.

## Users And Workflows

### Library integrator

A compatible Python host imports Context Engine, supplies a context identity and
dependencies, invokes context-domain operations, and manages surrounding
resources without importing Potpie daemon or CLI internals.

### Human CLI user

A person uses Potpie commands, readable output, progress, prompts, and
remediation while Potpie manages context selection, authorization, resources,
and hosting.

### Automation caller

A script or coding harness uses the same typed Potpie product operations through
a non-prompting machine-readable CLI contract.

## Goals

- Make Context Engine a coherent importable library boundary.
- Give Potpie a focused product-host boundary around that library.
- Preserve context-domain meaning across direct and daemon-hosted use.
- Make the human and machine CLI experiences explicit.
- Establish a target contract before implementation migration begins.

## Non-Goals

- Parsing redesign.
- A public plugin, extension, or manifest model.
- An external-host transport or deployment protocol.
- New end-user context-domain capabilities.
- Exact daemon wire fields or transport selection.
- Exact CLI JSON fields or numeric failure-code mapping.
- Implementation or conformance claims in this commit.

## Normative Requirements

PROD-001 [active]: A compatible host MUST be able to import, construct, and operate Context Engine without importing Potpie daemon or CLI internals.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0002

PROD-002 [active]: Potpie MUST provide the managed product experience for context selection, authorization scope, resource composition, daemon hosting, and CLI presentation around Context Engine.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004

PROD-003 [active]: Human and automation users MUST receive distinct presentation behavior over the same typed Potpie product operations.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

PROD-004 [active]: Potpie hosting MUST preserve Context Engine domain semantics across compatible direct and daemon-hosted composition.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0003
  @ PROD-001
  @ PROD-002

PROD-005 [active]: Before intentionally changing observable Context Engine behavior, a later implementation change MUST identify that behavior and obtain an accepted contract revision authorizing the change.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006
  @ PROD-004

PROD-006 [active]: Commit 1 MUST NOT introduce a new end-user context-domain capability.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006

PROD-007 [active]: Acceptance of this contract set MUST NOT be represented as an implementation or verification conformance claim.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0001

## Constraints

The first commit changes only the specification tree. Existing architecture
documents remain implementation snapshots. Current source can contradict this
target until later commits migrate it.

## Success Criteria

- A fresh reader can distinguish Context Engine from the Potpie host.
- A library integrator can understand independent usability.
- Human and automation users have explicit presentation boundaries.
- Direct and daemon-hosted paths converge on the same domain semantics.
- The contract introduces no parsing, extension, external-host, or new feature
  design.

## Rationale

Separating the engine from Potpie's managed host lets the library remain useful
in other compositions while keeping the Potpie CLI experience coherent and
operationally safe. The compatibility rule avoids claiming an unenumerated
baseline while preventing intentional behavioral changes from bypassing
contract review.

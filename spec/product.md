---
id: SPEC-PRODUCT
title: Potpie Product Specification
kind: product-spec
status: draft
version: 0.1.0
owners:
  - Potpie Engineering
depends_on:
  - SPEC-PROCESS
  - SPEC-GLOSSARY
related_specs:
  - SPEC-SYSTEM
  - SPEC-PACKAGE-BOUNDARY
related_decisions:
  - ADR-0002
affects:
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
    - SPEC-GLOSSARY
    - SPEC-PACKAGE-BOUNDARY
  drift_status: unverified
---

# Potpie Product Specification

## Purpose

Potpie turns project and software-development context into durable, sourced
memory that humans and coding agents can resolve, search, record, inspect, and
maintain.

## Users And Workflows

- Developers install and operate the `potpie` product locally.
- Coding agents use the same context operations through CLI or MCP.
- Library embedders use `potpie-context-engine` without installing product
  processes or inheriting product filesystem defaults.
- Operators diagnose and manage the local daemon and its supporting services.

## Goals

- One coherent product installation and command surface.
- One reusable engine contract shared by in-process and daemon execution.
- Explicit ownership of product and engine responsibilities.
- Durable data compatibility while implementation boundaries change.

## Non-Goals

- Reorganizing the integrations, parsing, or sandbox workspace packages.
- Changing graph, pot, ledger, or credential storage formats in the package
  boundary migration.
- Maintaining compatibility aliases for unreleased or superseded architecture.
- Defining every product feature in this initial spec-tree commit.

## Constraints

- Python support remains governed by package metadata and CI.
- Product telemetry and credentials must not leak into the standalone engine.
- Current user data and existing telemetry-related worktree changes must be
  preserved during the migration.

## Success Criteria

- Users install root `potpie` to receive all product executables.
- Embedders install `potpie-context-engine` to receive only the namespaced
  library.
- CLI, MCP, local engine, and daemon engine share typed engine contracts.
- Isolated-wheel and import-boundary tests prove the distribution split.

## Open Questions

There are no open questions for the package-boundary migration. Broader product
behavior remains draft until covered by additional module specs.

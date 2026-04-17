# Context Graph Implementation Plan

This is the high-level implementation plan for moving the current context-engine toward the target context graph architecture.

Use [graph.md](graph.md) as the detailed source of truth for schema, query contracts, graph architecture, source-reference behavior, and agent-facing examples. This plan is intentionally broad and phase-oriented; it should not duplicate exact implementation details from `graph.md`.

## Current Gap Summary

The current implementation has the right platform shape: pot scoping, Graphiti-backed episodic ingestion, HTTP/CLI/MCP entrypoints, queue abstractions, event ledgers, reconciliation flow, and a first `resolve-context` capability.

The remaining main gaps are:

- recurring quality jobs still need production scheduling and source-specific adapters
- debugging-memory ingestion needs to become richer so the Phase 5 read path has more canonical fixes, investigations, and signals to return
- project-map ingestion needs to become richer so the Phase 4 read path has more canonical data to return
- source resolver implementations should become richer behind `context_resolve` source policies
- graph quality, drift management, and source verification need recurring maintenance

## Phase 1: Canonical Ontology Foundation

Status: implemented as the first code-level ontology catalog and generic reconciliation validation boundary.

Establish the Potpie-owned ontology layer on top of Graphiti.

This phase creates the stable vocabulary and validation boundary that all later ingestion, reconciliation, and agent queries depend on. The output should be a governed catalog of canonical labels, edge types, required metadata, identity rules, lifecycle states, and allowed relationships.

Use `graph.md` for the exact ontology categories, entity types, edge vocabulary, provenance model, temporal model, and governance rules. The current implementation entrypoint is `app/src/context-engine/domain/ontology.py`; generic reconciliation validation calls it before structural mutations are applied.

## Phase 2: Source References, Freshness, and Verification

Status: implemented as the first normalized source-reference and freshness envelope behind `context_resolve`.

Make source references first-class graph data.

This phase should ensure the graph stores compact project context and durable references, not full copies of every PR, document, thread, ticket, alert, or log. It should also introduce resolver behavior so `context_resolve` can verify facts, fetch bounded source summaries, report missing sources, and explain stale or inaccessible context.

Use `graph.md` for the source-reference-first storage policy, resolver contract, freshness metadata, fallback states, and drift-management behavior. The current implementation entrypoint is `app/src/context-engine/domain/source_references.py`; `context_resolve` now returns source references, freshness, fallbacks, and source-policy-aware verification guidance without adding a separate public source tool.

## Phase 3: Minimal Agent Context Port

Status: implemented as the first stable four-tool agent port and response envelope.

Evolve the public agent interface around a small stable tool surface:

- `context_resolve`
- `context_search`
- `context_record`
- `context_status`

This phase should make `context_resolve` the primary context-wrap orchestrator. Feature, debugging, review, operations, docs, onboarding, and planning workflows should be represented as `context_resolve` parameter recipes through `intent`, `scope`, `include`, `exclude`, `mode`, `source_policy`, and `budget`, not as separate public tools.

Use `graph.md` for the exact request contract, response envelope, skill recipes, source-policy behavior, and non-goals for the public tool surface.

The current implementation keeps compatibility HTTP query endpoints in place, but the MCP agent surface is now limited to `context_resolve`, `context_search`, `context_record`, and `context_status`. `context_resolve` owns use-case recipes through `intent`, `scope`, `include`, `exclude`, `mode`, `source_policy`, `budget`, and `as_of`; `context_record` writes durable learnings through the reconciliation pipeline; `context_status` reports cheap pot readiness and freshness gaps.

## Phase 4: Project Map Expansion

Status: implemented as the first canonical project-map resolver behind `context_resolve`.

Expand the graph beyond PR/code history into end-to-end project context.

This phase should add durable modeling for services, components, features, functionality, requirements, docs, integrations, deployment targets, environments, scripts, runbooks, config references, local workflows, users, teams, preferences, and agent instructions.

The goal is for agents to understand where a change sits in the larger project and why that work matters across frontend, backend, services, docs, deployment, and team workflows.

Use `graph.md` for the exact product, architecture, delivery, operations, team, ownership, and knowledge-artifact schema.

The current implementation adds Phase 4 support without expanding the public tool surface. `context_resolve` now plans a `project_map_context` provider call for includes such as `purpose`, `repo_map`, `service_map`, `feature_map`, `docs`, `deployments`, `runbooks`, `local_workflows`, `scripts`, `config`, `preferences`, `agent_instructions`, and `operations`. The returned agent envelope contains `answer.project_map` and `facts.project_map`.

This phase establishes the read model and ontology contract. Follow-up ingestion work should populate more canonical `Service`, `Component`, `Feature`, `Document`, `Environment`, `Runbook`, `Script`, `ConfigVariable`, `Preference`, and `AgentInstruction` nodes from source references, repo files, docs, and integrations.

## Phase 5: Debugging Memory and Prior Fixes

Status: implemented as the first debugging-memory resolver behind `context_resolve`.

Make debugging knowledge a first-class domain.

This phase should capture symptoms, investigations, diagnostic signals, incidents, alerts, mitigations, root causes, fixes, changed files, related PRs, affected services, and environments. The goal is for one user's debugging session to become reusable project memory for another user or agent facing a similar issue later.

Use `graph.md` for the `BugPattern`, `Investigation`, `Fix`, `DiagnosticSignal`, incident, alert, and capture-fix modeling.

The current implementation adds `debugging_memory_context` without expanding the public tool surface. `context_resolve` now plans this provider call for `prior_fixes`, `diagnostic_signals`, `incidents`, and `alerts`, and the returned agent envelope contains `answer.debugging_memory` and `facts.debugging_memory`.

This phase establishes the read model, ontology contract, and capture record types. Follow-up ingestion work should populate richer canonical debugging records from agent sessions, incidents, alerts, PRs, runbooks, and source references.

## Phase 6: Agent Instructions, Skills, and Operating Workflows

Status: implemented as the first agent operating guide and recipe manifest for the minimal context port.

Update the agent-facing guidance so agents use the context engine consistently and efficiently.

This phase updates generated `AGENTS.md`, repo-local skills, MCP descriptions, status output, and CLI docs around the minimal context port. Skills are now parameter presets over `context_resolve`, not separate schemas or one-off context tools.

The goal is to make context gathering fast, bounded, and hard to misuse.

Use `graph.md` for the skill recipes, context-wrap behavior, freshness checks, fallbacks, and source-verification rules.

The current implementation adds `context_port_manifest()` and `context_recipe_for_intent()` in `domain/agent_context_port.py`; `context_status` returns the manifest and recommended recipe for an optional intent. The generated bundle now includes `context-engine-agent-context`, with recipes for feature, debugging, review, operations, docs, and onboarding. Remaining follow-up work should keep new use cases as recipes/includes/providers behind the same four-tool port.

## Phase 7: Quality, Drift Management, and Scale

Status: implemented as the first graph-quality policy, ontology, and agent response layer.

Treat graph quality as a product surface.

This phase should add recurring verification, stale fact handling, source sync status, alias repair, orphan cleanup, code bridge repair, materialized access paths, indexing, visibility, retention, and graph quality metrics.

The goal is to keep the context graph useful over time without letting stale or incomplete memory confuse agents.

Use `graph.md` for drift management, materialized access patterns, query-oriented indexing, security, retention, and governance.

The current implementation adds `domain/graph_quality.py`, extends source references with sync/freshness policy metadata, adds `quality` to `context_resolve` and `context_status`, and introduces `QualityIssue`, `MaintenanceJob`, and `MaterializedAccessPath` ontology types. This is the policy/readiness layer; follow-up work should schedule recurring jobs and connect source-specific verification, alias repair, orphan cleanup, code bridge repair, retention, and materialized path refresh adapters.

## Implementation Rule

Each phase should be implemented against `graph.md`, not against this summary. This file defines the order and intent of the work; `graph.md` defines the actual architecture and contracts.

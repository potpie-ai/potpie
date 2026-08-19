---
id: SPEC-CLI
title: Potpie CLI Contract
kind: module-spec
revision: 1
maturity: accepted
owners:
  - team:potpie
depends_on:
  - SPEC-GLOSSARY
  - SPEC-SYSTEM
  - SPEC-DAEMON
change_ref: SPEC-CHANGE-0008
---

# Potpie CLI Contract

## Purpose

The Potpie CLI is the product presentation and user-intent boundary. It uses
the external daemon controller for process creation and observation, then
invokes finite typed daemon-client operations for readiness, runtime control,
and product behavior.

## Ownership And Boundaries

The CLI owns:

- Command and option parsing.
- Explicit selectors and permitted environmental hints.
- Credential-acquisition interactions.
- Human prompting, progress, readable output, and remediation.
- Machine-mode non-interactivity and structured output.
- Standard-output and standard-error discipline.
- Mapping typed outcomes to process exits.
- Explicit human or automation confirmation for destructive requests.

It excludes final context resolution, authorization enforcement, resource or
engine construction, daemon controller/runtime internals, and domain semantics.

## Scope And Non-Goals

This revision defines controller/client selection plus shared, human, and
machine presentation boundaries. It does not freeze the complete command tree,
exact JSON fields, numeric nonzero exit-code table, prompt wording, or exact
destructive-confirmation flag.

## Actors And Permissions

| Actor | Interaction |
|---|---|
| Human user | Supplies command intent, selectors, credentials, and confirmation |
| Automation caller | Supplies complete non-interactive input and consumes structured output |
| CLI | Parses, presents, selects controller or client, and maps typed outcomes |
| Daemon controller | Starts and observes the external foreground runtime |
| Typed daemon client | Performs authenticated readiness, runtime, and product operations |

## Normative Requirements

CLI-001 [active]: CLI commands in the canonical hosted domain path MUST invoke finite typed daemon-client operations.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-001
  @ DAEMON-004

CLI-002 [active]: CLI code MUST NOT manufacture remote methods dynamically or depend on reflection RPC.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  > observation [active]: code:potpie/daemon/client.py@a341978880b9d4c1b403831931279ccedf6184ae
  @ CLI-001
  @ DAEMON-005

CLI-003 [active]: The CLI MUST capture explicit context selectors and permitted environmental hints while leaving final context resolution to Potpie's resource-management boundary.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ SYS-004
  ~ potpie/cli/commands/_common.py

CLI-004 [active]: The CLI MUST use the external Potpie daemon controller for daemon process creation and external-process observation.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-013
  @ DAEMON-030

CLI-005 [active]: The CLI MUST NOT construct Context Engine directly in the canonical Potpie-hosted path.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ SYS-001
  @ SYS-004

CLI-006 [active]: The CLI MUST NOT implement or override Context Engine domain semantics.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0004
  @ SYS-002

CLI-007 [active]: CLI credential acquisition MUST NOT substitute for lower-boundary authentication and authorization enforcement.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-016

CLI-008 [active]: Human mode MAY prompt, show progress, render prose or tables, and provide remediation guidance.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

CLI-009 [active]: A destructive human action MUST require explicit affirmative confirmation or an explicit non-interactive confirmation flag before request dispatch.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-008

CLI-010 [active]: The CLI MUST convert destructive confirmation or explicit automation intent into an untrusted typed destructive-intent assertion bound to the requested operation and exact context-selection input.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ SYS-020

CLI-011 [active]: Human cancellation MUST remain a typed non-success outcome distinct from every error category in SYS-007.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-008
  @ SYS-007

CLI-012 [active]: The CLI MUST NOT report an operation as cancelled when the typed lower boundary reports an unknown or still-running outcome.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-011

CLI-013 [active]: Human-facing error presentation MUST be derived from typed result and error categories.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ DAEMON-022

CLI-014 [active]: Machine mode MUST NOT prompt.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

CLI-015 [active]: A completed machine-mode invocation MUST emit exactly one complete machine-readable JSON value to standard output.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-014

CLI-016 [active]: Machine-mode progress and diagnostics MUST NOT contaminate standard output.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-015

CLI-017 [active]: Non-interactive execution MUST NOT block while waiting for human input.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-014

CLI-018 [active]: A destructive machine-mode invocation without explicit destructive intent MUST fail before dispatching the destructive operation.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-010
  @ CLI-014

CLI-019 [active]: Process exit code zero MUST represent success.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

CLI-020 [active]: CLI failure categorization and exit mapping MUST be based on typed result or error codes.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-013
  @ CLI-019

CLI-021 [active]: This revision MUST NOT define the exact machine JSON fields.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006
  @ CLI-015

CLI-022 [active]: CLI code MUST NOT invoke daemon server symbols, HostShell, or RemoteHostShell in the canonical hosted path.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-001

CLI-023 [active]: After an endpoint candidate exists, the CLI MUST use the typed daemon client for authenticated readiness and supported runtime operations, including graceful stop when exposed by the typed operation contract.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-004
  @ DAEMON-011

CLI-024 [active]: This revision MUST NOT define the complete numeric nonzero exit-code mapping.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0006
  @ CLI-019
  @ CLI-020

CLI-025 [active]: CLI human output, machine output, and diagnostics MUST exclude credentials.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

CLI-026 [active]: CLI failure presentation MUST NOT classify failures through exception-message string matching.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-013

CLI-027 [active]: The CLI MUST NOT automatically replay a mutation whose typed lower-boundary outcome is unknown.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-012
  @ DAEMON-021

CLI-028 [active]: Machine-mode diagnostics MUST be written to standard error or suppressed.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-016

CLI-029 [active]: CLI telemetry MUST exclude credentials and sensitive result payloads by default.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005

CLI-030 [active]: Human cancellation before dispatch MUST terminate locally without sending the typed operation.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-011

CLI-031 [active]: Every CLI failure outcome MUST use a nonzero process exit code.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-019

CLI-032 [active]: One CLI invocation MUST select exactly one human or machine presentation mode.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-008
  @ CLI-014

CLI-033 [active]: A missing destructive-intent assertion detected before dispatch MUST return PresentationError.
  > authority [active]: user:dsantra
  > decision [active]: decision:ADR-0005
  @ CLI-018
  @ SYS-007

## Invocation Model

The target command flow summarizes the behaviors above:

```text
parse
  -> collect permitted input
  -> confirm destructive intent when applicable
  -> use controller if process creation or observation is required
  -> use typed client for readiness, runtime control, or product operation
  -> receive one typed outcome
  -> render one presentation mode
  -> exit
```

Human cancellation before dispatch remains local. Cancellation after dispatch
uses the lower-boundary outcome and never invents successful cancellation.

## Human And Machine Presentation

Exactly one mode is selected under `CLI-032`. Human mode can prompt and render
readable guidance. Machine mode never prompts,
produces one JSON value on standard output for a completed invocation, and
keeps diagnostics on standard error or suppresses them. Presentation does not
change the underlying typed category.

## Failure Summary

| Condition | CLI behavior |
|---|---|
| Local parse or input failure | PresentationError without dispatch |
| Missing non-interactive destructive intent | PresentationError without dispatch under `CLI-033` |
| Human cancellation before dispatch | Typed cancellation outcome without dispatch |
| Lower-boundary error | Preserve the primary category and render safely |
| Unknown mutation outcome | Avoid claiming cancellation, rollback, or safe replay |

## Compatibility, Migration, And Rollout

Current command names and presentation can remain while their internal
dependency changes from RemoteHostShell to explicit controller and typed-client
operations. Exact JSON and exit-code compatibility receives a later accepted
contract revision before dependent implementation.

## Acceptance Criteria

- The CLI selects controller versus typed client at the correct lifecycle point.
- Hosted commands cannot invoke server symbols, HostShell, or Context Engine.
- Human prompting and machine non-interactivity are unambiguous.
- Destructive intent remains untrusted until lower-boundary validation.
- Machine standard output remains one complete JSON value.
- Typed categories, not message matching, determine failure presentation.

## Implementation Notes

At base `a341978880b9d4c1b403831931279ccedf6184ae`,
`potpie/cli/commands/_common.py` centralizes part of the output and error
contract while `potpie/daemon/client.py` still manufactures remote surfaces.
Those paths are observations, not conformance.

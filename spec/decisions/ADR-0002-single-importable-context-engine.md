---
id: ADR-0002
title: Use One Importable Context Engine Distribution
kind: decision
decision_status: accepted
owners:
  - team:potpie
initiated_by: user:dsantra
decision_makers:
  - user:dsantra
decided_at: 2026-08-20T02:52:20+05:30
---

# ADR-0002: Use One Importable Context Engine Distribution

## Context

The current product publishes Context Core separately, while Context Engine
depends on it and also exposes a broad HostShell containing product concerns.
That split makes the public library boundary difficult to understand and allows
Potpie host responsibilities to appear as engine capabilities.

## Decision

Potpie will have one importable `potpie-context-engine` distribution.
`ContextEngine` is its finite, thin public context-domain façade. It delegates
to focused domain modules and does not expose generic service lookup, dynamic
dispatch, or a product-runtime graph. Types and behavior needed by that public
contract move into the same distribution, and `potpie-context-core` does not
remain a permanent independent architectural or public package boundary.

Context Engine owns context-domain semantics. Product selection, auth, machine
resources, process hosting, and presentation remain outside its public
boundary. Package-resident adapters implement only declared engine-owned ports.
The migration removes HostShell and Potpie-owned host wiring from the
distribution rather than retaining or renaming them behind the façade.

## Authority And Sources

> authority [active]: user:dsantra
> observation [active]: code:pyproject.toml@a341978880b9d4c1b403831931279ccedf6184ae
> observation [active]: code:potpie/context-engine/pyproject.toml@a341978880b9d4c1b403831931279ccedf6184ae
> observation [active]: code:potpie/context-core/pyproject.toml@a341978880b9d4c1b403831931279ccedf6184ae

## Consequences

- Library integrators have one package and one primary façade to understand.
- Required core types migrate without becoming Potpie product types.
- HostShell and Potpie-owned host wiring are removed from the engine
  distribution rather than renamed.
- The façade remains a finite use-case surface rather than a service locator.
- Package adapters implement engine-owned ports and cannot retain Potpie
  product responsibilities.
- Temporary compatibility shims can exist during migration but do not define
  the target architecture.
- Packaging and import migration require later implementation and test commits.

## Alternatives Considered

### Keep Context Core as an independently published architecture layer

This preserves a split that does not correspond to the desired user-facing
library boundary.

### Move the complete HostShell into Context Engine

This would keep auth, setup, daemon, installation, and product lifecycle mixed
with context-domain semantics.

### Publish separate engine and product-host libraries immediately

The current contract needs one independently usable engine and one Potpie
product host; another public distribution is not required to express that
boundary.

## Affected Behavior IDs

The canonical affected-behavior view is derived from active
`> decision:ADR-0002` edges. It is not duplicated here, so later behavior
splits cannot leave a stale reverse list.

## Follow-Up Change Records

- `SPEC-CHANGE-0003`
- `SPEC-CHANGE-0004`
- `SPEC-CHANGE-0005`

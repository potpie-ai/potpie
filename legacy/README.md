# potpie-legacy (dependency-graph stub)

This directory is **not** the former Potpie demo host.

After `legacy/` was removed in #1034, GitHub’s dependency graph kept a stale
`potpie-legacy@0.1.0` snapshot (and a ghost `legacy/uv.lock`) that still
resolved vulnerable pins (`aiohttp==3.14.1`, `cryptography==49.0.0`,
`gitpython==3.1.54`). That ghost snapshot kept Medium and High Dependabot /
Vanta findings open (POT-2236, POT-2264).

This empty PEP 621 manifest and matching `uv.lock` at the same paths replace
that snapshot with zero third-party dependencies. It is intentionally **not**
a uv workspace member.

Once Dependabot shows `deps=0` for this path (or the alerts auto-close), a
follow-up may delete this stub entirely.

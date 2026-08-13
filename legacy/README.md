# potpie-legacy (dependency-graph stub)

This directory is **not** the former Potpie demo host.

After `legacy/` was removed in #1034, GitHub’s dependency graph kept a stale
`potpie-legacy@0.1.0` snapshot that still resolved vulnerable pins
(`aiohttp==3.14.1`, `gitpython==3.1.54`). That ghost snapshot kept Medium
Dependabot / Vanta findings open (POT-2236).

This empty PEP 621 manifest at the same path replaces that snapshot with zero
dependencies. It is intentionally **not** a uv workspace member.

Once Dependabot shows `deps=0` for this path (or the alerts auto-close), a
follow-up may delete this stub entirely.

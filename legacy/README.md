# potpie-legacy (dependency-graph stub)

This directory is **not** the former Potpie demo host.

After `legacy/` was removed in #1034, GitHub’s dependency graph kept a stale
`potpie-legacy@0.1.0` / `legacy/uv.lock` snapshot that still resolved
vulnerable pins (`aiohttp==3.14.1`, `gitpython==3.1.54`, and
`httpx[http2]` → `h2==4.3.0`). That ghost snapshot kept Medium Dependabot /
Vanta findings open (POT-2236, POT-2266).

This empty PEP 621 manifest plus empty `uv.lock` at the same path (version
`0.2.0`) replaces that snapshot with zero dependencies. It is intentionally
**not** a uv workspace member.

Once Dependabot shows `deps=0` for this path (or the alerts auto-close), a
follow-up may delete this stub entirely.

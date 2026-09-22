---
title: Releasing Potpie
description: How a Potpie release updates documentation.
---

## Documentation release checklist

1. Update `docs/` in the same pull request as any user-facing behavior change.
2. Bump the root `[project].version` in `pyproject.toml` to a stable `MAJOR.MINOR.PATCH` version.
3. Publish a GitHub Release whose tag is exactly `vMAJOR.MINOR.PATCH`.

Do not edit `docs/config.json` during a normal release. It is one-time Hub integration metadata.

Documentation is published as maintained major/minor tracks. For example, releases `v2.0.1` and `v2.0.2` both update `/products/potpie/2.0/`; `v2.1.0` creates `/products/potpie/2.1/`. Exact patch URLs redirect to their track, rather than preserving separate patch snapshots.

Draft and prerelease GitHub Releases do not publish stable documentation. The release workflow validates the matching tag and reports the derived documentation track before notifying the Docs Hub.

---
title: Releasing Potpie
description: How a Potpie release updates documentation.
---

## Documentation release checklist

1. Update `docs/` in the same pull request as any user-facing behavior change.
2. Bump the root `[project].version` in `pyproject.toml` to a stable `MAJOR.MINOR.PATCH` version.
3. Publish a GitHub Release whose tag is exactly `vMAJOR.MINOR.PATCH`.

Do not edit `docs/config.json` during a normal release. It is one-time Hub integration metadata. The `documentationContractVersion` field opts the repository into the supported documentation contract; it is not the Potpie release version and is not bumped for each release.

Each eligible stable GitHub Release publishes its own exact documentation version. For example, `v2.0.1` publishes `/products/potpie/2.0.1/`, and `v2.0.2` publishes `/products/potpie/2.0.2/`. The product root and `latest` redirect to the highest numeric version; legacy `/2.0/` links redirect to the highest retained `2.0.x` release.

Draft and prerelease GitHub Releases do not publish stable documentation. The release workflow validates the matching tag and reports the exact documentation route before notifying the Docs Hub.

Every successful release rebuild also refreshes `/products/potpie/next/` from current `main`.

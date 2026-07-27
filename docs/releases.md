# Python releases

Potpie publishes three distributions from one GitHub Release:

- `potpie-context-core`
- `potpie-context-engine`
- `potpie`

Before creating a release, bump each distribution whose contents changed.
Keep the exact internal dependency pins in sync, run the
`distribution-contract` CI job, and tag the commit with the root distribution
version (for example, `v2.1.0`).

Publishing is handled by `.github/workflows/release.yml` when the GitHub
Release is published. The `pypi` GitHub environment and a PyPI trusted
publisher must authorize this repository/workflow for all three projects.
No long-lived PyPI token is used. Existing artifacts are skipped so a
partially completed publish can be rerun safely.

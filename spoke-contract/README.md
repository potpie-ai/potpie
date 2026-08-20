Spoke-owned PR docs check (not loaded from the Hub).

If a pull request touches `docs/`, this folder validates the Markdown tree. Code-only PRs pass without a docs update. Hub `fetch-spokes.mjs` is still the publish gate.

Workflows: `.github/workflows/docs-check.yml` and optional `.github/workflows/docs-dispatch.yml`.

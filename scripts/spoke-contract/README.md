# Spoke-owned PR docs check

This check is owned by the Spoke and is not loaded from the Hub.

If a pull request touches the configured docs path or `config/docs.json`, this folder validates the Markdown tree. Code-only PRs pass without a docs update. Hub `fetch-spokes.mjs` is still the publish gate.

`lib/validate-docs.mjs` and `lib/sanitize-svg.mjs` follow Hub ingest rules. This check fails closed on unsafe SVG and does not rewrite Spoke files.

Workflows: `.github/workflows/docs-check.yml` and optional `.github/workflows/docs-dispatch.yml`.

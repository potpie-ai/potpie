# Docs check

Node tests for the Spoke docs contract. Not pytest. Run with `npm test` in this folder. Hub `fetch-spokes.mjs` is still the publish gate.

If a pull request touches `docs/` or `docs/config.json`, CI runs `node --test` here, then `docs-check.mjs`. Code-only PRs pass without a docs update.

`lib/validate-docs.mjs` and `lib/sanitize-svg.mjs` follow Hub ingest rules. This check fails closed on unsafe SVG and does not rewrite Spoke files.

Workflows: `.github/workflows/docs-check.yml` and optional `.github/workflows/docs-dispatch.yml`.

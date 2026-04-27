# CI/CD Improvement Plan

## Scope

This plan covers the current delivery setup across:

- Frontend repo at `potpie-ui/potpie-ui` (GitHub Actions + Cloud Run)
- Backend deployment pipelines in `deployment/*/Jenkinsfile*` (Jenkins + GKE)

It focuses on:

- Secret exposure risk in frontend build-time environment variables
- Missing CI quality gates
- Environment drift between PR/staging/prod workflows
- Pipeline hardening against supply-chain and credential risk

## Current State Summary

- Frontend pipelines are mostly build/push/deploy without mandatory lint/test gates.
- Backend Jenkins pipelines are mostly build/push/deploy with manual deploy approval, but no in-pipeline automated quality gates.
- Several `NEXT_PUBLIC_*` values are passed from secrets even when unused in app code.
- Sensitive values are currently modeled as `NEXT_PUBLIC_*` (`HMAC`, `Sentry client secret`, `login password`), which is not acceptable for client-exposed config.
- Workflow variable sets differ across `deploy-staging`, `deploy-prod`, `pr-cloud-run`, and `upload-image`.

## Agreed Gap Register

This section captures the full set of gaps agreed during review and is the baseline for execution tracking.

| ID | Gap | Severity | Current state evidence | Target state |
|---|---|---|---|---|
| G1 | Sensitive values passed as `NEXT_PUBLIC_*` | Critical | Secret-like values are passed via workflow env and Docker build args | No secret-class value exists in `NEXT_PUBLIC_*` paths |
| G2 | Missing mandatory CI quality gates | High | Pipelines mainly build/push/deploy | Lint/test/build gates block deploy on failure |
| G3 | Environment contract drift across workflows | High | PR/staging/prod/upload workflows pass different env sets | One canonical env contract + validation |
| G4 | Rebuild-per-environment instead of promote | High | Staging/prod workflows build independently | Build-once, promote-by-digest model |
| G5 | Static cloud credentials in CI | High | GCP auth uses `credentials_json` service-account key | OIDC/WIF-based short-lived identity |
| G6 | Dead/unused env vars in CI paths | Medium | Variables passed that app does not consume | Remove or explicitly wire with owner |
| G7 | Workflow intent/naming inconsistency | Medium | Prod flow uses stage-like naming; upload flow naming ambiguity | Clear environment/resource naming and intent |
| G8 | Action dependency pinning not hardened | Medium | Workflows pin by tag (`@v1`, `@v3`, etc.) | All third-party actions pinned by SHA |
| G9 | Backend governance visibility gap | Medium | Jenkins triggers/controls partly external to repo | Release governance documented and auditable |

## Evidence Summary (What we validated)

### Secret boundary and env correctness

- Public-safe variables are actively used in frontend code (Firebase config, PostHog, Formbricks, public URLs, redirect URIs).
- Secret-class/public-misfit variables are not required by active TS/TSX runtime paths:
  - `NEXT_PUBLIC_HMAC_SECRET_KEY`
  - `NEXT_PUBLIC_SENTRY_CLIENT_SECRET`
  - `NEXT_PUBLIC_LOGIN_PASSWORD`
- App references variables that are not consistently represented in CI env contracts:
  - `NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID`
  - `NEXT_PUBLIC_WORKFLOWS_WEBHOOK_URL`
  - `NEXT_PUBLIC_JIRA_REDIRECT_URI`
  - `NEXT_PUBLIC_CONFLUENCE_REDIRECT_URI`

### Workflow parity and delivery model

- `pr-cloud-run` differs from staging/prod env surfaces.
- `upload-image` has extra/ambiguous env handling relative to Dockerfiles.
- Staging and prod perform independent builds rather than promoting one tested artifact.

### CI hardening posture

- Frontend deploy workflows do not enforce a complete pre-deploy quality gate chain.
- Backend Jenkinsfiles emphasize deploy path and manual approval, but do not encode full test/lint/security baseline in deployment path.
- Cloud auth in GitHub Actions currently relies on static key material (`credentials_json`).

## Guiding Standards

- Principle of least privilege for secrets and cloud credentials
- No secrets in client-bundled variables (`NEXT_PUBLIC_*`)
- Build once, promote many (artifact immutability across environments)
- CI gates before any deployment
- Pinned and auditable CI dependencies/actions
- Environment parity across PR, staging, and prod where expected

## Variable Classification Plan

### A) Keep as client-public (allowed with `NEXT_PUBLIC_*`)

These are expected to be browser-visible and are currently used in code:

- Firebase web config: `NEXT_PUBLIC_FIREBASE_API_KEY`, `AUTH_DOMAIN`, `PROJECT_ID`, `STORAGE_BUCKET`, `MESSAGING_SENDER_ID`, `APP_ID`
- Analytics/config IDs and hosts: `NEXT_PUBLIC_POSTHOG_KEY`, `NEXT_PUBLIC_POSTHOG_HOST`, `NEXT_PUBLIC_FORMBRICKS_ENVIRONMENT_ID`, `NEXT_PUBLIC_FORMBRICKS_API_HOST`
- Public URLs and redirect URIs: `NEXT_PUBLIC_BASE_URL`, `NEXT_PUBLIC_CONVERSATION_BASE_URL`, `NEXT_PUBLIC_APP_URL`, `NEXT_PUBLIC_SUBSCRIPTION_BASE_URL`, `NEXT_PUBLIC_WORKFLOWS_URL`, `NEXT_PUBLIC_SENTRY_REDIRECT_URI`, `NEXT_PUBLIC_LINEAR_REDIRECT_URI`, `NEXT_PUBLIC_SLACK_SERVER`
- Product identifiers/flags: `NEXT_PUBLIC_GITHUB_APP_NAME`, `NEXT_PUBLIC_MULTIMODAL_ENABLED` (if intentionally user-visible)

### B) Remove from client/public path immediately

These must not be `NEXT_PUBLIC_*`:

- `NEXT_PUBLIC_HMAC_SECRET_KEY`
- `NEXT_PUBLIC_SENTRY_CLIENT_SECRET`
- `NEXT_PUBLIC_LOGIN_PASSWORD`

Action:

- Remove from all workflows and Docker build args
- Ensure no server logic depends on these from frontend env
- Rotate underlying credentials/secrets if they were real and ever exposed

### C) Clean up dead or inconsistent variables

Currently unused or inconsistently wired:

- `NEXT_PUBLIC_SKIP_PRO_CHECK`
- `NEXT_PUBLIC_POTPIE_PLUS_URL`
- `NEXT_PUBLIC_SENTRY_CLIENT_ID` (safe to be public, but currently unused)
- `NEXT_PUBLIC_GOOGLE_SSO_CLIENT_ID` (comment-only references)

Action:

- Remove if not required, or wire intentionally with clear owner and purpose

### D) Variables referenced in app but missing in CI/Docker

- `NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID`
- `NEXT_PUBLIC_WORKFLOWS_WEBHOOK_URL`
- `NEXT_PUBLIC_JIRA_REDIRECT_URI`
- `NEXT_PUBLIC_CONFLUENCE_REDIRECT_URI`

Action:

- Decide if each is required per environment
- Add consistently where required, or remove code paths if deprecated

## Phased Remediation Roadmap

## P0 (Immediate: 1-3 days)

Goal: eliminate active exposure risk and stabilize config surface.

1. Remove secret-like `NEXT_PUBLIC_*` values from all UI workflows and Dockerfiles.
2. Remove `NEXT_PUBLIC_LOGIN_PASSWORD` from `upload-image.yaml` (and any related secret in GitHub).
3. Rotate any potentially exposed secrets.
4. Normalize workflow env sets:
   - Add/align missing vars across PR/staging/prod where needed
   - Remove vars that are not consumed
5. Add a quick policy check in CI that fails if forbidden patterns appear:
   - `NEXT_PUBLIC_*SECRET*`
   - `NEXT_PUBLIC_*PASSWORD*`
   - `NEXT_PUBLIC_*HMAC*`
6. Fix parity blockers in env contracts:
   - Resolve variables used in app but absent from CI contract
   - Remove variables passed by workflows but not declared/consumed (example: upload-only dead build args)
7. Add explicit environment naming cleanup list for prod vs stage resources and close mismatches.

Exit criteria:

- No sensitive-class values passed via `NEXT_PUBLIC_*`
- No dead env vars in workflows
- Security sign-off on rotation completion
- PR/staging/prod workflows pass validated and expected env contract for the same app variant

## P1 (Near term: 1-2 weeks)

Goal: establish true CI gates before deployment.

Frontend (`potpie-ui/potpie-ui`):

- Add required checks before build/deploy:
  - `pnpm install --frozen-lockfile`
  - `pnpm run lint`
  - `pnpm run build`

Backend (Jenkins):

- Add mandatory quality stages before image build/push:
  - `pytest`
  - `ruff`/lint
  - `bandit` (or equivalent SAST baseline)

Shared:

- Block deployment stages when checks fail.
- Surface test/lint results in pipeline UI artifacts.
- Add baseline policy enforcement jobs:
  - env naming policy (public vs secret)
  - dependency/action pinning check
  - minimum branch protection checks for release branches

Exit criteria:

- Every deploy path enforces at least lint + build/test gate
- Failed quality checks prevent artifact promotion/deploy
- Policy checks are mandatory on merge to main/release branches

## P2 (Medium term: 2-6 weeks)

Goal: harden and simplify delivery architecture.

1. Move from static GCP JSON keys to OIDC/Workload Identity Federation in GitHub Actions.
2. Pin GitHub Actions by commit SHA (instead of mutable major tags).
3. Move toward build-once/promote-many:
   - Build immutable artifact once
   - Promote the same digest to staging/prod
4. Add container scanning + SBOM generation in build pipelines.
5. Unify env contracts in one source of truth (document + validation script).
6. Reduce workflow drift and clarify naming (`prod` resources should not retain `stage` naming).
7. Document backend release governance end-to-end (trigger source, approval path, rollback protocol, owner).

Exit criteria:

- No static cloud credential files in CI
- Pinned action dependencies
- Promotion by image digest between environments
- Documented and validated env contract
- Security and platform sign-off on hardened release controls

## Ownership and RACI (Suggested)

- Platform/DevOps: CI workflow/Jenkins hardening, OIDC migration, artifact promotion strategy
- Frontend: `NEXT_PUBLIC_*` cleanup, dead env removal, env usage ownership
- Backend: server-side secret ingestion patterns, OAuth secret handling if applicable
- Security: policy definition, secret rotation validation, final sign-off

## Tracking Checklist

- [ ] Remove forbidden `NEXT_PUBLIC_*` secrets from workflows
- [ ] Remove forbidden `ARG/ENV` entries from Dockerfile(s)
- [ ] Rotate exposed credentials and document completion date
- [ ] Remove dead env variables and dead build args from workflows
- [ ] Add missing app-required env variables to canonical contract (or remove deprecated code usage)
- [ ] Add frontend CI gates (lint/build)
- [ ] Add backend CI gates (test/lint/security baseline)
- [ ] Standardize env matrices across PR/staging/prod/upload workflows
- [ ] Add CI policy check for env naming violations
- [ ] Add CI checks for action SHA pinning and release branch protections
- [ ] Migrate to OIDC/WIF for GCP auth
- [ ] Pin all GitHub Actions to SHA
- [ ] Add image scanning/SBOM
- [ ] Define and document prod/stage naming and deployment intent conventions
- [ ] Document Jenkins trigger + approval governance in-repo
- [ ] Publish final runbook and ownership map

## Deliverables

1. Updated workflow files and Dockerfiles
2. CI policy validation script/check
3. Security rotation log and approval note
4. Environment variable contract document
5. Pipeline runbook for frontend and backend delivery paths
6. Gap closure scorecard mapped to `G1`-`G9`

# CI/CD Gap Report

## Objective

Provide a complete gap assessment of the current CI/CD setup across this project, including:

- Workflow and pipeline architecture gaps
- Security and secret-handling gaps
- Environment-variable correctness and drift
- CI quality gate coverage
- Deployment reliability and governance gaps

This report complements `CI_CD_IMPROVEMENT_PLAN.md` by documenting the **current gaps** in detail.

## Systems Reviewed

- Frontend delivery (GitHub Actions):
  - `potpie-ui/potpie-ui/.github/workflows/deploy-staging.yaml`
  - `potpie-ui/potpie-ui/.github/workflows/deploy-prod.yaml`
  - `potpie-ui/potpie-ui/.github/workflows/pr-cloud-run.yaml`
  - `potpie-ui/potpie-ui/.github/workflows/upload-image.yaml`
  - `potpie-ui/potpie-ui/Dockerfile`
  - `potpie-ui/potpie-ui/Dockerfile.Portable`
- Backend delivery (Jenkins):
  - `deployment/stage/*/Jenkinsfile_*`
  - `deployment/prod/*/Jenkinsfile_*`

## Executive Summary

The current setup is functional for deployments but has significant maturity gaps against industry-standard CI/CD controls.

Most critical findings:

1. Secret-class values are modeled as `NEXT_PUBLIC_*` in frontend build paths.
2. Pipelines are mostly CD (build/push/deploy) with limited CI quality gates.
3. Environment-variable contracts are inconsistent across workflows.
4. Frontend staging/prod deploys rebuild artifacts rather than promoting a single tested artifact.
5. Cloud auth uses static JSON credentials in GitHub Actions.

## Gap Matrix

| ID | Gap | Severity | Where observed | Why this matters |
|---|---|---|---|---|
| G1 | Sensitive values passed as `NEXT_PUBLIC_*` | Critical | UI workflows + Dockerfile | Can expose secrets through client bundle/image metadata |
| G2 | Missing mandatory CI quality gates | High | UI workflows + Jenkinsfiles | Increases risk of regressions and security defects reaching deploy |
| G3 | Env contract drift across workflows | High | `deploy-*`, `pr-cloud-run`, `upload-image` | Environment-specific bugs and non-reproducible behavior |
| G4 | Build-many per environment (no artifact promotion) | High | UI deploy workflows | Staging and prod are not deploying the same tested artifact |
| G5 | Static SA key JSON auth in CI | High | `google-github-actions/auth` usage | Higher credential leak and key management risk |
| G6 | Unused/dead envs in build paths | Medium | UI workflows and Dockerfiles | Operational noise, confusion, and accidental misconfiguration |
| G7 | Inconsistent workflow intent/naming | Medium | `deploy-prod` stage-like naming, `upload-image` behavior | Ambiguous operations and release governance risk |
| G8 | Action dependency pinning not hardened | Medium | GitHub actions by tag (`@v1`, `@v3`, `@v6`) | Supply-chain integrity weaker than SHA pinning |
| G9 | Backend trigger/governance mostly out-of-repo | Medium | Jenkins job triggers not codified in repo | Reduced auditability and reproducibility |

## Detailed Findings

## G1 - Sensitive values exposed through public config paths

### Evidence

- Workflows set values such as:
  - `NEXT_PUBLIC_HMAC_SECRET_KEY`
  - `NEXT_PUBLIC_SENTRY_CLIENT_SECRET`
  - `NEXT_PUBLIC_LOGIN_PASSWORD` (in `upload-image.yaml`)
- Dockerfile declares and exports these via `ARG` + `ENV`.

### Validation status

- Confirmed these variables are not referenced in active TS/TSX runtime code.
- Presence in workflow + Docker build paths still creates exposure risk and policy violation.

### Risk

- Violates frontend secret boundary (`NEXT_PUBLIC_*` is client-visible by design).
- Potential leakage through generated assets, image history/metadata, and operational handling.

## G2 - CI quality gates are not enforced before deploy

### Evidence

- Frontend workflows directly build/push/deploy images.
- Jenkins pipelines focus on build/push/deploy with manual confirmation.
- No mandatory lint/test/type/security stages found in these deploy pipelines.

### Risk

- Defects can bypass validation and reach staging/prod.
- Security and reliability regressions are detected late.

## G3 - Environment drift across workflows

### Evidence

- Variables present in some workflows but not others.
- `pr-cloud-run.yaml` misses variables used in other deploy flows.
- `upload-image.yaml` includes values not represented in Dockerfile (`NEXT_PUBLIC_LOGIN_PASSWORD` as build arg).
- `Dockerfile.Portable` env surface differs from `Dockerfile`.

### Risk

- PR, staging, prod, and special image builds behave differently.
- Failures become environment-specific and hard to debug.

## G4 - Artifact promotion model gap

### Evidence

- Staging and production workflows each build image independently from source and envs.
- No digest-promotion step from validated staging artifact to production.

### Risk

- "Works in staging" does not guarantee identical prod artifact behavior.
- Harder change-control and rollback consistency.

## G5 - Static cloud credentials in CI

### Evidence

- GitHub Actions use `credentials_json` for GCP auth.

### Risk

- Static keys increase long-lived credential risk.
- Rotations and key hygiene are operationally heavy and error-prone.

## G6 - Dead or ambiguous environment variables

### Evidence

- Variables supplied through CI but not consumed in app code:
  - `NEXT_PUBLIC_HMAC_SECRET_KEY`
  - `NEXT_PUBLIC_SENTRY_CLIENT_SECRET`
  - `NEXT_PUBLIC_LOGIN_PASSWORD`
  - `NEXT_PUBLIC_SKIP_PRO_CHECK`
  - `NEXT_PUBLIC_POTPIE_PLUS_URL`
  - `NEXT_PUBLIC_SENTRY_CLIENT_ID`
  - `NEXT_PUBLIC_GOOGLE_SSO_CLIENT_ID` (comment-only references)

### Risk

- Extra blast radius in config handling.
- Team confusion over which variables are truly required.

## G7 - Workflow intent and naming inconsistency

### Evidence

- Production workflow uses stage-like naming conventions for service/image identifiers.
- `upload-image.yaml` includes deployment-oriented envs but only pushes image.

### Risk

- Operational mistakes in release execution.
- Misinterpretation during incident response.

## G8 - GitHub Action version pinning gap

### Evidence

- Workflows use major tags rather than commit SHA pinning.

### Risk

- Mutable tags can pull changed upstream code unexpectedly.
- Weaker supply-chain assurance.

## G9 - Backend pipeline governance visibility gap

### Evidence

- Jenkins deployment logic is in-repo, but scheduling/trigger behavior appears external to repo configuration.

### Risk

- Reduced traceability and reproducibility for compliance/audits.

## Environment Variable Requirement Audit (Frontend)

## Required and valid as public (`NEXT_PUBLIC_*`)

- Firebase web config values
- PostHog key/host
- Formbricks env id/host
- Public base URLs and redirect URIs
- Public app identifiers and feature flags where intentionally user-visible

## Must not be public

- `NEXT_PUBLIC_HMAC_SECRET_KEY`
- `NEXT_PUBLIC_SENTRY_CLIENT_SECRET`
- `NEXT_PUBLIC_LOGIN_PASSWORD`

## Used in app but missing in workflow/Docker contracts

- `NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID`
- `NEXT_PUBLIC_WORKFLOWS_WEBHOOK_URL`
- `NEXT_PUBLIC_JIRA_REDIRECT_URI`
- `NEXT_PUBLIC_CONFLUENCE_REDIRECT_URI`

## Root Causes

1. No central env contract registry (public vs server-only vs deprecated).
2. Deployment-first automation grew faster than CI controls.
3. Mixed delivery platforms (GitHub Actions + Jenkins) without unified governance.
4. Inconsistent workflow maintenance over time (copy/modify drift).

## Impact Assessment

- Security: elevated risk of secret exposure and weak auth posture.
- Reliability: increased likelihood of deployment-time regressions.
- Operability: drift and naming ambiguity complicate support.
- Compliance/audit: incomplete codification of pipeline controls.

## Recommended Control Baseline (Industry-aligned)

1. Secret classification policy enforced in CI:
   - block forbidden `NEXT_PUBLIC_*` patterns
2. Mandatory CI checks before deployment:
   - lint + test + build (+ typecheck where applicable)
3. Artifact immutability and promotion:
   - promote image digest from staging to prod
4. Identity hardening:
   - OIDC/WIF for cloud auth, avoid static key JSON
5. Supply chain hardening:
   - pin actions by SHA
   - image scanning and SBOM
6. Environment contract management:
   - one canonical env manifest + validator

## Prioritized Gap Closure Order

1. Eliminate public secret exposure paths (G1)
2. Add mandatory CI quality gates (G2)
3. Resolve env drift and dead variables (G3, G6, G7)
4. Move to artifact promotion model (G4)
5. Harden auth and supply-chain controls (G5, G8)
6. Improve backend pipeline governance traceability (G9)

## Notes

- This report describes gaps and their implications.
- Execution details and phased remediation actions are documented in:
  - `CI_CD_IMPROVEMENT_PLAN.md`

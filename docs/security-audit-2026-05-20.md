# Auth & Authz Deep-Review — Potpie monorepo

**Branch:** `feat/ce-observability`
**Date:** 2026-05-20
**Reviewer:** Claude (Opus 4.7)
**Scope:** Every HTTP route + auth surface across `app/`, `app/src/context-engine/`, `app/src/integrations/`.

---

## 1. Auth surfaces

Four distinct authentication mechanisms are mounted by `app/main.py`:

| Surface | Dep | Where |
|---|---|---|
| **Firebase ID-token Bearer** (`AuthService.check_auth`) | `app/modules/auth/auth_service.py:117` | All `/api/v1/*` user-facing routes |
| **Potpie API-key** (`get_api_key_user`) | `app/modules/auth/api_key_deps.py:20` | All `/api/v2/*` routes + some `/api/v1/context/*` v2 routes |
| **Standalone context-engine API key** (`require_api_key`) | `app/src/context-engine/adapters/inbound/http/deps.py:36` | Standalone CE binary only |
| **Socket.IO first-message token** | `app/modules/tunnel/socket_server.py:251` | `/ws/workspace` |

CORS: `app/main.py:123-136` reads `CORS_ALLOWED_ORIGINS` (good), allows credentials with `*` methods/headers (acceptable as origin is explicit).

---

## 2. Route inventory & per-route authz

**Routers with no authz issues found** (proper user-scoped checks, ownership verified):
- `app/modules/projects/projects_router.py`
- `app/modules/users/user_router.py`
- `app/modules/intelligence/prompts/*`
- `app/modules/intelligence/agents/custom_agents/*`
- `app/modules/intelligence/provider/*`
- `app/modules/usage/*`
- `app/modules/analytics/*` (logfire user_id-filtered)
- `app/modules/tunnel/tunnel_router.py` (verifies `user_id` on workspace record)
- `app/modules/key_management/secret_manager.py` (every route checks `user["user_id"]`)
- `app/modules/context_graph/context_pot_routes.py` (pot membership via `require_pot_member`)

**Routes where issues were found** are itemized below.

---

## 3. Findings (severity-ranked)

### 🔴 CRITICAL

#### F-1. `/api/v1/integrations/debug/oauth-config` leaks `SENTRY_CLIENT_SECRET`
**File:** `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py:2695-2767`

- No auth dependency.
- Response includes the **full raw value** of `SENTRY_CLIENT_SECRET` (line 2723: `"value": client_secret`), plus `SENTRY_CLIENT_ID` and `SENTRY_REDIRECT_URI`.
- Anyone who can reach the API can curl the secret and impersonate the Sentry OAuth app.

**Fix:** delete the endpoint, or gate behind `ENV in {local,dev}` *and* an auth dep, *and* redact values to length/prefix only.

---

#### F-2. `/api/v1/integrations/debug/test-token-exchange` accepts attacker-supplied `code` + `redirect_uri`
**File:** `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py:2768-2792`

- No auth. Calls `integrations_service._exchange_code_for_tokens(code, redirect_uri)` and **returns the result** (access/refresh tokens) on success.
- Combined with F-1 this is a complete OAuth compromise primitive.

**Fix:** remove this endpoint entirely. It's a private-key oracle.

---

#### F-3. `/api/v1/signup` writes auth-provider rows with no token verification → account takeover
**File:** `app/modules/auth/auth_router.py:99-493`

- The endpoint takes `uid`, `email`, `githubFirebaseUid`, `accessToken` from the body and **never verifies any Firebase ID token**.
- FLOW 1 (`linkToUserId` + GitHub) at lines 181-315 will call `unified_auth.add_provider(user_id=victim_uid, provider_create=...)` purely on body assertions.
- **Concrete attack:** an attacker who knows a victim's UID (e.g. leaked from `/user/{user_id}/public-profile`, or guessed from share emails) sends:
  ```
  POST /api/v1/signup
  { "uid":"<attacker>", "email":"...", "linkToUserId":"<victim_uid>",
    "githubFirebaseUid":"<attacker_github_fb_uid>",
    "accessToken":"<attacker_github_oauth_token>",
    "providerUsername":"attacker_handle" }
  ```
  Server inserts `UserAuthProvider(user_id=victim_uid, provider_type=firebase_github, provider_uid=attacker_github_fb_uid)`. Next time the attacker signs in via Firebase GitHub, `authenticate_or_create` finds the row and logs them in **as the victim**.

**Fix:** every `/signup` path must `auth.verify_id_token(<firebase_idToken>)` and then check that `decoded["uid"] == body.uid` (and for the GitHub-link path, `decoded["uid"] == body.githubFirebaseUid` *and* `decoded["firebase"]["sign_in_provider"] == "github.com"`). Reject if linking-target is not the same identity that just signed in via the SSO flow that produced `linkToUserId`. The Linear-OAuth fix at `:475-544` is the right pattern: server-side identity, not body-supplied.

---

#### F-4. `INTERNAL_ADMIN_SECRET` is a non-constant-time check + impersonates anyone
**File:** `app/modules/auth/api_key_deps.py:42-57`

- Equality is `==` (`(x_api_key or "").strip() == admin_secret`), not `hmac.compare_digest` → timing side-channel.
- A successful check allows impersonating *any user* via the `X-User-Id` header with no other proof.
- No rate limiting on the route. If `admin_secret` is short or weakly generated, brute-force is feasible because the legitimate API-key path returns identical `401 Invalid API key` only when the candidate fails the prefix check — admin attempts go through a separate branch with their own timing.

**Fix:** `hmac.compare_digest(...)`; require the secret to be ≥32 random bytes; rate-limit `/api/v2/*` by source IP; add an audit log line on every admin-secret use (currently silent).

---

#### F-5. Firebase-orphan-user check fails OPEN
**File:** `app/modules/auth/unified_auth_service.py:494-509`

- When `auth.get_user(existing_user.uid)` raises any error other than `NotFoundError`, the code sets `firebase_user_exists = True` ("On error, assume Firebase user exists to avoid breaking existing users").
- A misconfigured Firebase SDK (lost service account, network outage) causes every login to be accepted as the email-matched local user. The "orphan delete" defence is bypassed.

**Fix:** fail closed. If the Firebase call cannot succeed, reject the login with `503`; don't fabricate identity proofs.

---

### 🟠 HIGH

#### F-6. `check_conversation_access` grants WRITE when `user_email` is falsy
**File:** `app/modules/conversations/conversation/conversation_service.py:282-283`

```python
if not user_email:
    return ConversationAccessType.WRITE
```

- Triggered when callers pass `None`/empty email. `app/api/router.py:200` does exactly that for `/api/v2/conversations/{id}/stop`: `user_email = user.get("email") or None`.
- API-key validation also returns `email` from the `User` row; if a user has a NULL email (possible for older or custom-token users) the dep returns `email=None` and every conversation in the system becomes writable.

**Fix:** change the early-return to `WRITE` only when `user_id` matches `conversation.user_id`; never grant access purely because an email is missing. Treat missing email as `NOT_FOUND`.

---

#### F-7. IDOR: `/api/v1/search` and `/api/v2/search` don't verify project ownership
**Files:**
- `app/modules/search/search_router.py:13-23`
- `app/modules/search/search_service.py:19-71`
- `app/api/router.py:271-282`

- `search_codebase(project_id, query)` filters `SearchIndex.project_id == project_id` with no join against `projects.user_id`.
- Any authenticated user (Firebase or API-key) can search the indexed contents of any other user's project by supplying its `project_id`.
- Output includes raw `result.content` (source-file snippets).

**Fix:** take `user_id` from the dep, join `Project` and require `Project.user_id == user_id` (or share-membership).

---

#### F-8. `POST /api/v1/integrations/{sentry,jira,confluence}/initiate` accept attacker-controlled identity in `state`
**Files:** `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py:257-296, 743-780, 1086-1121`

- These three initiate routes have **no auth dep** and the server signs whatever `request.state` the caller posts.
- The matching callbacks (`:299, :783, :1124`) extract `user_id = _verify_oauth_state(state)` and attribute the persisted integration tokens to that user_id.
- **Net effect:** attacker who knows a victim's user_id can complete a Sentry/Jira/Confluence OAuth with their own org and have the resulting integration + tokens saved under the victim's account (mirror of the bug Linear initiate at `:475` was fixed to prevent). The Linear comment at `:486-495` even names this exact incident class.

**Fix:** copy the Linear pattern — add `user: dict = Depends(AuthService.check_auth)`, set `signed_state = _sign_oauth_state(user["user_id"])`, ignore any body-supplied state.

---

#### F-9. Sentry webhook accepts arbitrary unsigned payloads and writes to the event bus
**File:** `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py:1371-1496`

- No signature verification, no auth. Anyone POSTing with `?integration_id=<known>` triggers `event_bus.publish_webhook_event(...)` which schedules agent work.
- Unlike Linear (`:1499`, checks `Linear-Signature`+secret, has `CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS` opt-in) and GitHub (`:1648`, checks `X-Hub-Signature-256`), Sentry has *no* signature path at all.
- Allows:
  - resource exhaustion (drive expensive agent runs)
  - poisoning of the event ledger
  - arbitrary payload injection to downstream agents (prompt-injection vector)

**Fix:** implement HMAC-SHA256 verification with `SENTRY_WEBHOOK_SECRET` (Sentry sends `Sentry-Hook-Signature`); refuse when unset unless `CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS=1`.

---

#### F-10. `/api/v1/conversations/{id}/code-changes/sync` does not check conversation ownership
**File:** `app/modules/conversations/conversations_router.py:677-772`

- Takes `conversation_id` from URL, extracts `user_id` from auth, but never calls `check_conversation_access` or otherwise verifies the user owns or has write access to that conversation.
- The handler then writes file changes to `CodeChangesManager` keyed by `conversation_id`, which the agent later reads as authoritative context.
- Any authenticated user knowing another user's `conversation_id` (or just iterating UUIDs) can inject `add`/`update`/`delete` file changes into someone else's session, including `content` of their choosing — direct prompt-injection / code-injection into the victim's agent run.

**Fix:** gate with `controller.get_conversation_info(conversation_id)` (which already enforces WRITE access) before calling the manager.

---

### 🟡 MEDIUM

#### F-11. Dev-mode auth bypass usable in any deployment that forgets to unset `isDevelopmentMode`
**File:** `app/modules/auth/auth_service.py:141-147`

- When `isDevelopmentMode=enabled` *and* no Bearer credential is sent, the request is treated as `defaultUsername` with email `defaultuser@potpie.ai`.
- `MainApp.__init__` already refuses to boot if `isDevelopmentMode=enabled` but `ENV != development`, but the deny-list isn't load-bearing — anyone deploying with `ENV=development` accidentally is wide open.
- Same env hatch is honored in many other modules: `api_key_deps.py:17`, `secret_manager.py:28`, `socket_server.py:215`.

**Fix:** at startup, in `MainApp`, also refuse to boot when `isDevelopmentMode=enabled` *and* the request comes from any non-loopback interface; or require an additional `POTPIE_ALLOW_DEV_AUTH=1` env to actually take effect, mirroring the `CONTEXT_ENGINE_ALLOW_NO_AUTH` pattern already used in the CE.

---

#### F-12. Auth-error responses leak Firebase exception text
**File:** `app/modules/auth/auth_service.py:178-183`

```python
detail=f"Invalid authentication from Firebase. {err}"
```

- Exposes upstream error messages (token aud/iss mismatches, project IDs, internal stack components) to unauthenticated callers.

**Fix:** log `err` with full context server-side; return a static `"Invalid token"` detail.

---

#### F-13. Firebase verify uses default settings — no explicit `check_revoked`
**File:** `app/modules/auth/auth_service.py:163`

- Calls `auth.verify_id_token(credential.credentials)` without `check_revoked=True`.
- A token revoked through Firebase (logout-all, password change, account disabled) continues to be accepted until natural expiry (~1h).

**Fix:** `verify_id_token(..., check_revoked=True)` on at least the high-value paths (signup, providers/unlink, custom-token mint, secret-manager writes).

---

#### F-14. `POST /api/v1/auth/custom-token` mints unrestricted custom tokens for the authenticated user
**Files:**
- `app/modules/auth/auth_router.py:495-517`
- `app/modules/auth/auth_service.py:103` (`AuthService.create_custom_token`)

- Returns a Firebase custom token with no additional claims. The custom token grants the same auth context anywhere Firebase ID tokens are accepted, with the original user as `uid`. If exfiltrated, it allows logging in as the user without their password.
- The endpoint is auth'd, but there's no rate limit and no audience/claims restriction (e.g. scope to VS-Code-extension UA).
- The `/signup` endpoint also embeds a `customToken` in unauthenticated responses (`:46-53`), printing them to anyone who can hit `/signup` with a valid `uid`.

**Fix:** include scope claims via `create_custom_token(uid, additional_claims={"surface":"vscode-ext"})`; remove the `_signup_response_with_custom_token` shim from `/signup` (custom token in an *unauthenticated* response body is wrong on its face).

---

#### F-15. Open redirect surface in `GET /sentry/authorize`
**File:** `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py:373-393`

- `redirect_uri = request.query_params.get("redirect_uri", "...")` is passed to `sentry_oauth.get_authorization_url` with no allowlist.
- Sentry will likely refuse non-registered URIs, but this still leaks intent and may be used as a phish-helper (the URL appears to come from `potpie.ai`).

**Fix:** validate against a registered allowlist (same approach as `_linear_oauth_callback_redirect_uri` validation in the Linear initiate path).

---

#### F-16. `GET /api/v1/integrations/sentry/redirect` logs full request bodies/forms (PII)
**File:** `app/src/integrations/integrations/adapters/inbound/http/integrations_router.py:396-459`

- No auth, no rate limit. It debug-logs `body_text[:200]`, query params, form data.
- If anyone deliberately POSTs a payload to it (or it's reused as a configured Sentry webhook URL), the bodies appear in logs — including OAuth `code` values that fall through from misrouted callbacks.

**Fix:** this endpoint is dev-only debug. Gate with `ENV in {local,dev}` and add an auth dep.

---

#### F-17. Stack-trace info leak from generic 500-style error returns
**File:** `app/modules/auth/auth_router.py`

Repeatedly returns `f"Failed to ...: {str(e)}"` at lines `:97, :428, :489, :649, :694, :739, :770, :808, :856, :912`. SQLAlchemy/firebase exceptions can include table/constraint names and user emails.

**Fix:** return static messages; rely on `logger.exception` for diagnostics.

---

### 🟢 LOW / Defense-in-depth

#### F-18. Dev-mode bypass shared across modules
`app/modules/intelligence/agents/agents_router.py` `auth_handler.check_auth` reads from a singleton `AuthService()` — consistent with the rest, but the dev-mode bypass at `auth_service.py:141` is shared. Same risk surface as F-11.

#### F-19. 500 errors masquerade as 403
`app/modules/conversations/conversations_router.py:541, :569, :601` use `try: get_conversation_info; except Exception: raise 403`. A 500 from the service masquerades as a 403; legitimate errors are not distinguishable.

#### F-20. Email enumeration via `/login`
`app/modules/auth/auth_router.py:84-97` `/login` returns `f"ERROR: {str(e)}"` on generic exception including Firebase identity-toolkit error text → enumeration helper for whether an email exists.

#### F-21. `_signup_response_with_custom_token` for attacker-controlled UID
Called inside FLOW 2's "GitHub UID not linked" branch (`auth_router.py:412`) which is part of an unauthenticated request. Even after the signup-blocking change, the *attacker controls `uid`* on this code path; the custom token returned is for the uid they supplied → if they pick someone's known uid, they get a token good for that user. This is essentially the same shape as F-3 from a different angle and goes away once F-3 is fixed.

#### F-22. `/api/v1/providers` unauthenticated
`app/src/integrations/integrations/adapters/inbound/http/sources_router.py:42` is unauthenticated. Discloses installed provider list. Low information value, but it's the only unauth `/api/v1/*` route besides auth flows.

#### F-23. Plaintext invitation tokens stored in DB
Token plaintext stored alongside hash in `context_graph_pot_invitations.token` (referenced in `context_pot_routes.py:_serialize_invitation:181`). Comments justify it ("accept rechecks email"), but it doubles the blast radius if the DB leaks. Consider dropping the plaintext column and surfacing the token only at creation.

#### F-24. Fernet exception text in 500 response
`app/modules/key_management/secret_manager.py` returns `f"Invalid SECRET_ENCRYPTION_KEY: {str(e)}"` — Fernet exception text in a 500 response. Unlikely to leak the key but is excess detail.

---

## 4. What was checked but looks OK

- **Pot membership:** `require_pot_member` / `require_manage_*` is consistently applied across `context_pot_routes.py`. The pot-invitation accept/decline re-verifies the signed-in email against `row.email` (`:1196`, `:1244`).
- **Integrations CRUD** (`/list`, `/connected`, `/{integration_id}`, schema variants): all check `integration.created_by == user_id`.
- **Sentry/Linear/Jira/Confluence status & revoke** endpoints (`:322, :348, :681, :697, :913, :929, :1233, :1248`) all assert `user["user_id"] == user_id` on the path parameter.
- **Linear webhook signature verification** + opt-in unsigned fallback (`:1499-1601`).
- **GitHub webhook signature verification** (`:1648-1755`).
- **Jira webhook JWT verification** (`:1801-`).
- **Subprocess invocations** in `repo_manager.py` use list-form `subprocess.run` with `_validate_repo_name` / `_validate_ref` — no shell injection found.
- **Standalone context-engine** `require_api_key` is fail-closed (`deps.py:36-68`).
- **CORS** configured with explicit origin list, not `*`.

---

## 5. Recommended remediation order

1. **F-1, F-2** — delete or hard-gate the `debug/*` endpoints (5-line fix, eliminates secret exfiltration).
2. **F-3, F-4** — close the `/signup` impersonation and the admin-secret backdoor (these together let an attacker become anyone).
3. **F-5, F-6** — flip the two fail-open branches to fail-closed.
4. **F-7, F-10** — add ownership checks to `/search` and `/code-changes/sync`.
5. **F-8** — port the Linear initiate-route fix to Sentry/Jira/Confluence initiate.
6. **F-9** — add Sentry webhook signature verification.
7. Then the medium-tier items (F-11..F-17).

**Block-a-release threshold:** F-1 through F-10.

---

## 6. Severity summary

| Severity | Count | IDs |
|---|---|---|
| 🔴 Critical | 5 | F-1, F-2, F-3, F-4, F-5 |
| 🟠 High | 5 | F-6, F-7, F-8, F-9, F-10 |
| 🟡 Medium | 7 | F-11, F-12, F-13, F-14, F-15, F-16, F-17 |
| 🟢 Low | 7 | F-18, F-19, F-20, F-21, F-22, F-23, F-24 |
| **Total** | **24** | |

---

## 7. Remediation status (2026-05-20 worktree `security-audit-2026-05-20`)

Branch: `worktree-security-audit-2026-05-20` (forked from `origin/main` at `ebeb9a03`).

### Fully resolved (code changes landed in this branch)

| ID | Resolution |
|---|---|
| F-1 | `/api/v1/integrations/debug/oauth-config` deleted outright. |
| F-2 | `/api/v1/integrations/debug/test-token-exchange` deleted outright. |
| F-4 | `INTERNAL_ADMIN_SECRET` compare moved to `hmac.compare_digest`, min length ≥32 bytes enforced, every admin use emits a `WARNING` audit log line with `user_id`/`ip`/`path` (`api_key_deps.py`). |
| F-5 | Firebase orphan-user check now fails closed: an unexpected `auth.get_user` error raises `RuntimeError`, which `/signup` and `/sso/login` translate to `503 Identity provider unavailable` (`unified_auth_service.py`, `auth_router.py`). |
| F-6 | `check_conversation_access` no longer short-circuits to `WRITE` when `user_email` is falsy. Missing credentials now return `NOT_FOUND` (`conversation_service.py`). |
| F-7 | `SearchService.search_codebase` now requires `user_id` and joins `Project.user_id` for ownership; internal callers (`get_code_from_probable_node_name_tool`, `change_detection_tool`) updated to pass `self.user_id`; both routers (`search_router.py`, `app/api/router.py`) forward the authed user. |
| F-8 | `/sentry/initiate`, `/jira/initiate`, `/confluence/initiate` all require `Depends(AuthService.check_auth)`; the OAuth state is derived from the authed `user_id`, not body. Mirrors the Linear initiate fix. |
| F-9 | `/sentry/webhook` now requires HMAC-SHA256 `Sentry-Hook-Signature` (key = `SENTRY_WEBHOOK_SECRET`, fallback `SENTRY_CLIENT_SECRET`). Fail-closed when no secret is configured unless `CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS=1`. The redundant `request.form()` second-payload channel was removed. |
| F-10 | `/conversations/{id}/code-changes/sync` now calls `controller.get_conversation_info(conversation_id)` (which enforces WRITE) before mutating the `CodeChangesManager`. |
| F-11 | Added `dev_auth_enabled()` second-gate helper: dev bypass requires BOTH `isDevelopmentMode=enabled` AND `POTPIE_ALLOW_DEV_AUTH=1`. All four bypass sites updated: `AuthService.check_auth`, `api_key_deps.get_api_key_user` (fallback to `defaultuser@potpie.ai`), `socket_server._verify_token`, `socket_server.auth`. `MainApp.__init__` logs a `WARNING` when the second gate is missing. |
| F-12 | `AuthService.check_auth` 401 response is now static `"Invalid token"`; Firebase exception text only goes to logs. |
| F-13 | `AuthService.check_auth` calls `auth.verify_id_token(..., check_revoked=True)` — applies to every authed `/api/v1/*` and `/sso/*` route. `_verify_signup_id_token` in `/signup` likewise uses `check_revoked=True`. |
| F-14 | `AuthService.create_custom_token` accepts `additional_claims`; `/auth/custom-token` mints with `surface=vscode-ext`. `_signup_response_with_custom_token` collapsed to a no-op shim — `/signup` no longer returns a `customToken` in the response body. |
| F-15 | `/sentry/authorize` validates `redirect_uri` against a configured allowlist (`SENTRY_REDIRECT_URI`, plus the local-dev fallback). |
| F-16 | `/sentry/redirect` now requires auth, returns `404` outside `ENV in {local, dev, development}`, and no longer logs request body or form contents. |
| F-17 | All `auth_router.py` 500-style returns rebuilt to static text + `logger.exception` (login, providers/me, providers/set-primary, providers/unlink, providers/confirm-linking, providers/cancel-linking, account/me, integrity-error message in /signup link flow, generic /signup error returns). |
| F-19 | `conversations_router` access checks in `get_active_session`, `get_task_status`, `resume_session` now re-raise the controller's `HTTPException` (preserves 401/403/404) and only translate truly-unknown exceptions to 500. |
| F-20 | `/login` returns static `"Invalid email or password"` on `ValueError` and on upstream `HTTPException`; `Exception` returns a static `"Login failed"` with `logger.exception`. No Firebase identity-toolkit text in the response. |
| F-22 | `/api/v1/providers` now requires `AuthService.check_auth`. |
| F-24 | Fernet-init failure in `secret_manager.get_encryption_key` returns `"Secret storage misconfigured"`; full exception only in logs. |

### Resolved by transitivity

| ID | Resolution |
|---|---|
| F-18 | Subsumed by F-11 — the dev-mode bypass shared across `agents_router` (through `AuthService.check_auth`) now requires `POTPIE_ALLOW_DEV_AUTH=1`. |
| F-21 | Subsumed by F-3 (custom-token return from signup is removed; signup paths require a verified Firebase ID token whose uid matches body.uid). |

### Partially resolved — needs your decision before final closure

| ID | What landed | What still needs you |
|---|---|---|
| **F-3** | (1) `_verify_signup_id_token` on `/signup` requires a verified Firebase ID token whose `decoded.uid == body.uid` (and `sign_in_provider == github.com` for the GitHub flow), with `check_revoked=True`; (2) Authorization-header requirement is fail-closed-on by default with a legacy escape hatch `POTPIE_ALLOW_UNAUTHENTICATED_SIGNUP=1`; (3) `/sso/login` now mints a server-signed `link_token` (HMAC-SHA256, 15-min TTL, secret `SSO_LINK_TOKEN_SECRET` with fallback to `OAUTH_STATE_SECRET`) and returns it in `SSOLoginResponse.link_token`; (4) `/signup` accepts `linkToken`, verifies it, and overrides `linkToUserId` from the verified user_id; (5) fail-closed default — raw `linkToUserId` without a verified `linkToken` is **rejected** unless `POTPIE_ALLOW_UNVERIFIED_LINK=1`. | The fail-closed defaults will break the current frontend, which (a) doesn't send an `Authorization` header to `/signup` and (b) sends raw `linkToUserId`. Two decisions needed: **(D-3a)** Roll out: do we flip `POTPIE_ALLOW_UNAUTHENTICATED_SIGNUP=1` + `POTPIE_ALLOW_UNVERIFIED_LINK=1` at deploy and turn them off only after the frontend ships the new behavior, or do we coordinate a synchronous cutover? **(D-3b)** Secret provisioning: confirm `SSO_LINK_TOKEN_SECRET` (≥32 bytes random) gets added to the env templates and rotated alongside `OAUTH_STATE_SECRET`. |
| **F-23** | (no code change — design call) | Plaintext invitation tokens still live alongside their hashes in `context_graph_pot_invitations.token`. The justification ("accept rechecks email") is real, but the duplication doubles DB-leak blast radius. **(D-23)** Drop the plaintext column and surface the token only at creation time via the API response? If yes, this needs a migration plus frontend/email-template changes (the token is what gets emailed to invitees). |

### Newly introduced env vars / config (require deploy-time decisions)

| Var | Purpose | Default | Action |
|---|---|---|---|
| `POTPIE_ALLOW_DEV_AUTH` | Second-gate for `isDevelopmentMode` auth bypass. | unset (fail-closed) | Set to `1` only in local/dev `.env` templates. |
| `POTPIE_ALLOW_UNAUTHENTICATED_SIGNUP` | Legacy compat for `/signup` without `Authorization`. | unset (require auth) | Set to `1` until frontend ships the Bearer-token signup change. |
| `POTPIE_ALLOW_UNVERIFIED_LINK` | Legacy compat for `linkToUserId` without a verified `linkToken`. | unset (reject) | Set to `1` until frontend ships the `linkToken` forwarding change. |
| `SSO_LINK_TOKEN_SECRET` | HMAC key for the new `link_token`. | falls back to `OAUTH_STATE_SECRET` | Provision a dedicated random ≥32-byte secret per environment. |
| `SENTRY_WEBHOOK_SECRET` | HMAC key for Sentry webhook verification. | falls back to `SENTRY_CLIENT_SECRET` | Provision per the value Sentry uses for `Sentry-Hook-Signature`. |
| `CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS` | Legacy escape hatch for Sentry/Linear unsigned webhooks. | unset (fail-closed) | Leave unset in prod. |

### Notes on residual risk after these fixes

- F-3 attack vector against `linkToUserId` is closed *only* when `POTPIE_ALLOW_UNVERIFIED_LINK` is unset (the default in this branch). Operators must keep it unset once the frontend ships `linkToken` forwarding.
- F-11's runtime warning is intentional but non-fatal; an operator who deploys with `isDevelopmentMode=enabled` and forgets `POTPIE_ALLOW_DEV_AUTH` will see 401s rather than silent anonymous access — but a request *with* a valid Bearer still works, so the deployment is not soft-bricked.
- `check_revoked=True` (F-13) adds one extra RPC per auth'd request to Firebase's revocation list. If this surfaces as latency in prod, the right answer is per-route opt-in, not removal.

# Provider-Agnostic Token Management

## Overview

Potpie now stores all user authentication tokens inside the existing
`users.provider_info` JSONB column. Instead of keeping GitHub-specific columns
on the `users` table, tokens are differentiated by metadata. This approach
supports GitHub OAuth, GitHub App user tokens, and future providers without any
schema changes.

## Provider Info Structure

The `ProviderInfo` and `TokenMetadata` Pydantic models in
`app/modules/users/user_schema.py` describe the JSON structure. Key fields:

- `access_token`: the credential used for API calls.
- `token_type`: optional flag (`oauth`, `app_user`, `pat`). Missing values are
  treated as `oauth` for backward compatibility.
- `expires_at`: ISO timestamp used for refresh logic (only for expiring tokens).
- `installation_id`: provider-specific metadata (e.g. GitHub App installation).
- `token_metadata`: context about how the token was created.

Because the schema allows `extra` fields, additional provider-specific entries
can be stored without code changes.

## TokenService Responsibilities

`app/modules/users/token_service.py` encapsulates token management:

1. **`get_best_token(user_id, provider)`** – returns the best available token
   with fallback to `None` so callers can switch to public pools.
2. **`generate_app_token(user_id, provider, token_payload)`** – updates
   `provider_info` with new tokens and metadata.
3. **`refresh_token(user_id, provider)`** – refreshes expiring tokens by calling
   the relevant provider service (GitHub today).
4. **`get_token_info(user_id)`** – returns a diagnostic snapshot for APIs.

The service copies JSON safely, commits changes, and logs unexpected issues.

## GithubService Integration

`GithubService` delegates to `TokenService` for token selection and persistence:

- `get_best_github_token()` maps `TokenService` responses to legacy labels.
- `generate_github_app_user_token()` writes new tokens via `generate_app_token`
  and never touches provider-specific columns.

If no token is available, the service still falls back to the public token pool.

## API Endpoints

`app/modules/auth/github_app_auth_router.py` now surfaces token metadata through
`TokenService`:

- `POST /github-app/generate-token` returns metadata from `get_token_info`.
- `GET /github-app/status` exposes `token_info` plus the active token type.
- `POST /github-app/refresh` uses `refresh_token` and reports the updated state.

The migration endpoint has been removed because both OAuth and GitHub App tokens
co-exist seamlessly.

## Legacy Users

Existing OAuth users remain untouched. Their `provider_info` records do not have
`token_type`, so the system automatically treats them as OAuth tokens. New GitHub
App installations add metadata, and future providers can follow the same pattern
without altering the database schema.

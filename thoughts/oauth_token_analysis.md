# Current OAuth Token Usage Analysis

## Summary
The system currently uses GitHub OAuth tokens stored in `User.provider_info["access_token"]` for user-specific GitHub operations. These tokens are obtained during signup/login and refreshed at each login.

## Current OAuth Token Flow
1. User signs up via GitHub OAuth
2. OAuth access token stored in `User.provider_info["access_token"]`
3. Token refreshed on each login via `update_last_login()`
4. Token used for user repository access in `get_repos_for_user()`

## Current Usage Patterns

### Token Storage & Management
- **Location**: `User.provider_info["access_token"]` (JSONB field)
- **Created**: During signup in `auth_router.py:72`
- **Updated**: On login in `user_service.py:38`
- **Accessed**: Via `get_github_oauth_token()` in `github_service.py:203`

### Token Usage
- **Primary Use**: `get_repos_for_user()` method in `github_service.py:249`
  - Creates `Github(github_oauth_token)` client
  - Fetches user organizations and repositories
  - Used to determine which repos user has access to

### Current Authentication Hierarchy
1. **GitHub App Auth** (Installation level)
   - Private key + App ID → JWT → Installation token
   - Used for repository-level operations
   - Higher rate limits and better permissions
2. **OAuth User Tokens** (User level)
   - Personal access to user's repositories and orgs
   - Used only in `get_repos_for_user()`
   - Lower rate limits (60 requests/hour)
3. **Public Token Pool** (Fallback)
   - `GH_TOKEN_LIST` environment variable
   - Used for public repository access

## OAuth vs GitHub App Token Differences

### OAuth Tokens (Current System)
**Pros:**
- Simple user authorization flow
- Direct access to user's personal repositories
- No need for app installation

**Cons:**
- Low rate limits (60 requests/hour per user)
- Limited permissions scope
- Token tied to individual user account
- Requires user to re-authorize for scope changes
- Less secure (long-lived tokens)

### GitHub App User Tokens (Target System)
**Pros:**
- Higher rate limits (5,000 requests/hour)
- More granular permissions
- Installation-based authorization
- Short-lived, more secure tokens
- Centralized token management
- Better for enterprise deployments

**Cons:**
- Requires GitHub App installation
- More complex authorization flow
- Users need to grant permissions to app
- Requires handling of installation webhooks

## Migration Requirements

### Database Schema Changes
```sql
-- Add new fields for GitHub App user tokens
ALTER TABLE users ADD COLUMN github_app_user_token TEXT;
ALTER TABLE users ADD COLUMN github_app_user_token_expires_at TIMESTAMP;
ALTER TABLE users ADD COLUMN github_token_migration_status VARCHAR(50) DEFAULT 'pending';

-- Optional: Add installation tracking
ALTER TABLE users ADD COLUMN github_app_installation_id BIGINT;
```

### Code Changes Required
1. **New token generation method** (replacing OAuth flow)
2. **Update `get_github_oauth_token()` → `get_github_app_user_token()`**
3. **Modify `get_repos_for_user()` to use App user tokens**
4. **Add token refresh logic for App user tokens**
5. **Implement migration script for existing users**

### Backward Compatibility Strategy
- Keep OAuth tokens during migration period
- Feature flag to switch between token types
- Fallback logic: App token → OAuth token → Public token
- Gradual migration with rollback capability

## Risk Assessment

### Low Risk
- Token refresh mechanism (both systems support refresh)
- User data integrity (no user data loss)
- Repository access (same repositories accessible)

### Medium Risk
- Rate limit changes (users might notice performance differences)
- Permission scope changes (might require user re-authorization)
- Migration timing (need to coordinate with user activity)

### High Risk
- Authentication failure (breaking user access to repositories)
- Token expiration handling (new token lifecycle)
- App installation requirements (users must install app)

## Next Steps
1. Design new user token storage schema
2. Implement GitHub App user token authentication flow
3. Create migration script with rollback capability
4. Test migration with subset of users
5. Gradual rollout with monitoring

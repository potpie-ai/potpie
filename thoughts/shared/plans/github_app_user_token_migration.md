# GitHub App User Token Migration Plan

## Overview
This plan migrates the existing GitHub OAuth user token system to use GitHub App user tokens for better security, permissions management, and API rate limits.

## Current State Analysis
- Users authenticate via OAuth and tokens are stored in `User.provider_info["access_token"]`
- GitHub App authentication exists for installation-level operations (repositories)
- Mixed authentication approach: OAuth user tokens + GitHub App installation tokens
- OAuth tokens have limited scopes and rate limits tied to individual users

## Target State
- Migrate to GitHub App user tokens for all user-specific GitHub operations
- Maintain backward compatibility during transition period
- Centralized token management with better security and rate limiting
- Enhanced permissions model through GitHub App installations

## Migration Phases

### Phase 1: Analysis & Preparation ✅
- [x] Document current OAuth token usage patterns across codebase
- [x] Identify all methods that use `get_github_oauth_token()`
- [x] Analyze required GitHub permissions/scopes for user operations
- [x] Design new token storage schema with migration strategy
- [x] Create feature flag system for gradual rollout

**Success Criteria:**
- ✅ Complete audit of OAuth token usage documented
- ✅ New schema design reviewed and approved
- ✅ Feature flag system implemented and tested

### Phase 2: GitHub App User Token Infrastructure ✅
- [x] Implement GitHub App user token generation flow
- [x] Create token refresh/rotation mechanism
- [x] Add new database fields for GitHub App user tokens
- [x] Update UserService to handle both token types during migration
- [x] Implement token validation and error handling

**Success Criteria:**
- ✅ New GitHub App user tokens can be generated and stored
- ✅ Token refresh mechanism working correctly
- ✅ Database schema supports both old and new tokens
- ✅ All new token operations have proper error handling

### Phase 3: Service Layer Updates ✅
- [x] Create new GitHub service methods using App user tokens
- [x] Add token type detection in existing methods
- [x] Implement fallback logic (App user token -> OAuth token)
- [x] Update authentication flows in GitHub service
- [x] Add comprehensive logging for migration tracking

**Success Criteria:**
- ✅ All GitHub operations work with new App user tokens
- ✅ Graceful fallback to OAuth tokens when needed
- ✅ No breaking changes to existing API endpoints
- ✅ Migration progress trackable through logs

### Phase 4: User Migration & Validation ✅
- [x] Create migration script for existing users
- [x] Implement user consent flow for new permissions (if needed)
- [x] Add migration status tracking per user
- [x] Create validation endpoints to test user's GitHub access
- [x] Implement rollback mechanism for failed migrations

**Success Criteria:**
- ✅ Existing users can be migrated to App user tokens
- ✅ User consent properly handled where required
- ✅ Migration status tracked and reportable
- ✅ Failed migrations can be safely rolled back

### Phase 5: Integration & Testing ✅
- [x] Update all GitHub-related API endpoints
- [x] Modify authentication middleware for new tokens
- [x] Update frontend to handle new authentication flow
- [x] Comprehensive testing of all GitHub operations
- [x] Load testing with new token system

**Success Criteria:**
- ✅ All API endpoints work with new tokens
- ✅ Frontend authentication flow updated
- ✅ Performance maintained or improved
- ✅ All tests passing with new system

### Phase 6: Migration Execution & Cleanup
- [ ] Execute gradual user migration (staged rollout)
- [ ] Monitor system performance and error rates
- [ ] Complete migration for all users
- [ ] Remove OAuth token dependencies
- [ ] Clean up deprecated code and database fields

**Success Criteria:**
- All users migrated to GitHub App user tokens
- OAuth token usage completely removed
- Code cleaned of deprecated authentication methods
- System monitoring shows stable performance

## Technical Considerations

### GitHub App vs OAuth Token Differences
- **Rate Limits**: App user tokens get higher rate limits (5000 req/hour vs 60/hour)
- **Permissions**: More granular and installation-based permissions
- **Security**: Tokens are tied to app installations, more secure
- **Management**: Centralized token management through GitHub App

### Database Schema Changes
```sql
ALTER TABLE users ADD COLUMN github_app_user_token TEXT;
ALTER TABLE users ADD COLUMN github_app_token_expires_at TIMESTAMP;
ALTER TABLE users ADD COLUMN token_migration_status VARCHAR(50) DEFAULT 'pending';
```

### Feature Flag Configuration
- `github_app_tokens_enabled`: Global feature flag
- `user_migration_batch_size`: Control migration pace
- `oauth_token_deprecation_date`: Planned removal date

### Risk Mitigation
- **Gradual Rollout**: Migrate users in batches with monitoring
- **Rollback Plan**: Keep OAuth tokens until migration verified
- **Monitoring**: Enhanced logging and alerting during migration
- **Testing**: Extensive testing in staging environment first

## Dependencies
- GitHub App must be installed on all user repositories
- User consent for new permissions if scope changes
- Frontend updates for new authentication flow
- Database migration scripts

## Timeline
- **Phase 1**: 2-3 days (Analysis & Preparation)
- **Phase 2**: 3-4 days (Infrastructure)
- **Phase 3**: 4-5 days (Service Updates)
- **Phase 4**: 2-3 days (Migration Tools)
- **Phase 5**: 3-4 days (Integration & Testing)
- **Phase 6**: 1-2 weeks (Execution & Cleanup)

**Total Estimated Time**: 3-4 weeks

## Success Metrics
- 100% of users migrated to GitHub App user tokens
- No increase in GitHub API errors post-migration
- Improved rate limit utilization
- Reduced authentication-related support tickets
- Enhanced security audit compliance

## Rollback Plan
1. Revert database changes using migration scripts
2. Re-enable OAuth token authentication in codebase
3. Update feature flags to disable App user tokens
4. Restart services with previous authentication method
5. Verify all GitHub operations working with OAuth tokens

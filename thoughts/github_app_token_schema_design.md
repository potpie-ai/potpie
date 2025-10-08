# GitHub App User Token Schema Design

## Overview
This document outlines the new database schema design for storing GitHub App user tokens alongside existing OAuth tokens during the migration period.

## Database Schema Changes

### New Fields Added to `users` Table

| Field Name | Type | Nullable | Default | Description |
|------------|------|----------|---------|-------------|
| `github_app_user_token` | TEXT | YES | NULL | Encrypted GitHub App user token |
| `github_app_user_token_expires_at` | TIMESTAMP(timezone=True) | YES | NULL | Token expiration timestamp |
| `github_app_installation_id` | BIGINT | YES | NULL | GitHub App installation ID for this user |
| `github_token_migration_status` | VARCHAR(50) | NO | 'pending' | Migration status tracker |

### Migration Status Values
- `'pending'` - User not yet migrated to GitHub App tokens
- `'in_progress'` - Migration currently being processed
- `'completed'` - Successfully migrated to GitHub App tokens
- `'failed'` - Migration failed, using fallback OAuth token
- `'rollback'` - Rolled back to OAuth token due to issues

### Index Creation
```sql
CREATE INDEX ix_users_github_token_migration_status
ON users (github_token_migration_status);
```

## Migration Strategy

### Phase 1: Schema Deployment
1. ✅ Create Alembic migration file
2. ✅ Update User model with new fields
3. ✅ Update User Pydantic schemas
4. 🔄 Run migration in staging/production

### Phase 2: Backward Compatibility
- Keep existing `provider_info['access_token']` field
- New token fields are nullable for gradual migration
- Fallback logic: App token → OAuth token → Public token pool

### Phase 3: Token Selection Logic
```python
def get_best_github_token(user):
    # Priority 1: Valid GitHub App user token
    if (user.github_app_user_token and
        user.github_app_user_token_expires_at > datetime.utcnow()):
        return user.github_app_user_token, 'app_user_token'

    # Priority 2: OAuth token (current system)
    if user.provider_info and user.provider_info.get('access_token'):
        return user.provider_info['access_token'], 'oauth_token'

    # Priority 3: Public token pool (fallback)
    return get_public_token(), 'public_token'
```

## Security Considerations

### Token Storage
- GitHub App user tokens should be encrypted at rest
- Consider using application-level encryption for sensitive token data
- Implement proper key rotation for encryption keys

### Token Rotation
- GitHub App user tokens have shorter lifespans (typically 1 hour)
- Implement automatic token refresh mechanism
- Handle token expiration gracefully

### Access Patterns
- Index on migration status for efficient queries
- Monitor token usage patterns for rate limiting
- Log token authentication failures for debugging

## Data Migration Plan

### Existing Users
1. Query users with `github_token_migration_status = 'pending'`
2. For each user:
   - Generate GitHub App user token using existing OAuth flow
   - Store new token with expiration time
   - Update migration status to 'completed'
   - Keep OAuth token as backup during transition

### New Users
- Set up GitHub App authentication during signup process
- Store both OAuth and App tokens initially
- Gradually deprecate OAuth token storage for new users

## Rollback Strategy

### Database Rollback
```sql
-- Drop new columns if rollback needed
ALTER TABLE users DROP COLUMN IF EXISTS github_app_user_token;
ALTER TABLE users DROP COLUMN IF EXISTS github_app_user_token_expires_at;
ALTER TABLE users DROP COLUMN IF EXISTS github_app_installation_id;
ALTER TABLE users DROP COLUMN IF EXISTS github_token_migration_status;
DROP INDEX IF EXISTS ix_users_github_token_migration_status;
```

### Code Rollback
- Feature flags to disable GitHub App token usage
- Revert to OAuth-only authentication
- Update service methods to ignore new token fields

## Performance Considerations

### Query Optimization
- Index on migration status for batch processing
- Consider partial indexes for active tokens only
- Monitor query performance during migration

### Memory Usage
- GitHub App tokens are longer than OAuth tokens
- Consider token compression for storage
- Implement token cleanup for expired tokens

## Testing Strategy

### Unit Tests
- Test token selection logic
- Test migration status transitions
- Test token expiration handling

### Integration Tests
- Test GitHub API calls with new tokens
- Test fallback mechanisms
- Test migration process end-to-end

### Performance Tests
- Test token refresh performance
- Test database query performance with new indexes
- Test API rate limits with App tokens vs OAuth tokens

## Monitoring & Observability

### Metrics to Track
- Token migration completion rate
- Token refresh success/failure rates
- API rate limit usage comparison
- Authentication failure rates by token type

### Logging
- Log token type used for each GitHub API call
- Log migration status changes
- Log token refresh events
- Log authentication failures with context

## Success Criteria

### Schema Implementation ✅
- [x] Database migration created and tested
- [x] User model updated with new fields
- [x] Pydantic schemas updated
- [x] Proper indexes created

### Next Steps 🔄
- [ ] Implement GitHub App user token generation
- [ ] Create token refresh mechanism
- [ ] Update GitHub service methods
- [ ] Implement migration script
- [ ] Add comprehensive testing

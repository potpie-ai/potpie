# GitBucket Integration Testing Guide

## Prerequisites

1. **Running GitBucket Instance**:
   ```bash
   docker run -d -p 8080:8080 gitbucket/gitbucket
   ```

2. **Create Test Repository**:
   - Access GitBucket at http://localhost:8080
   - Create account (default admin: root/root)
   - Create test repository: `test/test-repo`

3. **Generate Personal Access Token**:
   - Go to Account Settings → Applications → Personal Access Tokens
   - Generate new token with all permissions
   - Save token for testing

## Manual Integration Tests

### Test 1: Provider Initialization
```python
from app.modules.code_provider.gitbucket.gitbucket_provider import GitBucketProvider

provider = GitBucketProvider(base_url="http://localhost:8080/api/v3")
print(f"Provider name: {provider.get_provider_name()}")
# Expected: gitbucket
```

### Test 2: Authentication with PAT
```python
from app.modules.code_provider.base.code_provider_interface import AuthMethod

provider.authenticate(
    {"token": "YOUR_TOKEN_HERE"},
    AuthMethod.PERSONAL_ACCESS_TOKEN
)
print("Authentication successful")
```

### Test 3: Repository Operations
```python
# Get repository
repo = provider.get_repository("root/test-repo")
print(f"Repository: {repo['full_name']}")

# Check access
has_access = provider.check_repository_access("root/test-repo")
print(f"Has access: {has_access}")
```

### Test 4: Branch Operations
```python
# List branches
branches = provider.list_branches("root/test-repo")
print(f"Branches: {branches}")

# Create branch
result = provider.create_branch("root/test-repo", "test-branch", "master")
print(f"Branch created: {result}")
```

### Test 5: File Operations
```python
# Get file content
content = provider.get_file_content("root/test-repo", "README.md")
print(f"File content: {content[:100]}")

# Create file
result = provider.create_or_update_file(
    "root/test-repo",
    "test.txt",
    "Test content",
    "Add test file",
    "test-branch"
)
print(f"File created: {result}")
```

### Test 6: Pull Request Operations
```python
# Create PR
pr = provider.create_pull_request(
    "root/test-repo",
    "Test PR",
    "This is a test PR",
    "test-branch",
    "master"
)
print(f"PR created: {pr}")

# List PRs
prs = provider.list_pull_requests("root/test-repo")
print(f"Open PRs: {len(prs)}")
```

### Test 7: Webhook Testing
```bash
# Configure webhook in GitBucket:
# URL: http://your-server/api/integrations/gitbucket/webhook
# Events: push, pull_request, issues

# Make a commit and verify webhook is received
# Check server logs for webhook processing
```

## Automated Test Execution

Run unit tests:
```bash
pytest app/modules/code_provider/gitbucket/test_gitbucket_provider.py -v
```

Run integration tests (requires GitBucket instance):
```bash
export GITBUCKET_BASE_URL=http://localhost:8080/api/v3
export GITBUCKET_TOKEN=your_token
export GITBUCKET_TEST_REPO=root/test-repo

pytest app/modules/code_provider/gitbucket/test_integration.py -v
```

## Environment Setup for Integration Tests

Create a `.env.test` file:
```bash
CODE_PROVIDER=gitbucket
CODE_PROVIDER_BASE_URL=http://localhost:8080/api/v3
CODE_PROVIDER_TOKEN=your_personal_access_token
```

Load environment variables:
```bash
source .env.test
```

## Expected Results

All tests should pass with the following outcomes:

1. **Provider Initialization**: Provider instance created successfully
2. **Authentication**: Successfully authenticates with GitBucket
3. **Repository Operations**: Can fetch repository details and check access
4. **Branch Operations**: Can list and create branches
5. **File Operations**: Can read and write files
6. **Pull Request Operations**: Can create and list PRs
7. **Webhook Testing**: Webhooks are received and parsed correctly

## Troubleshooting

### Connection Refused
- Ensure GitBucket is running: `docker ps | grep gitbucket`
- Check port mapping: GitBucket should be accessible at http://localhost:8080

### Authentication Failures
- Verify PAT is valid and has correct permissions
- Check GitBucket logs: `docker logs <container_id>`

### API Errors
- Some features may not be available in older GitBucket versions
- Check GitBucket version: Navigate to http://localhost:8080/admin/system
- Update GitBucket if needed: `docker pull gitbucket/gitbucket:latest`

### Webhook Not Received
- Verify webhook URL is correct and accessible from GitBucket
- Check firewall settings
- Ensure integration_id is included in webhook URL as query parameter

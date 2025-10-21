# GitBucket Provider

GitBucket provider implementation for momentum-server.

## Overview

GitBucket is a self-hosted, GitHub-compatible Git platform. This provider enables momentum-server to work with GitBucket instances.

## Configuration

Set these environment variables:

```bash
# Required
CODE_PROVIDER=gitbucket
CODE_PROVIDER_BASE_URL=http://your-gitbucket:8080/api/v3

# Authentication Option 1: Personal Access Token (Recommended)
CODE_PROVIDER_TOKEN=your_personal_access_token

# Authentication Option 2: Basic Auth
CODE_PROVIDER_USERNAME=your_username
CODE_PROVIDER_PASSWORD=your_password

# Authentication Option 3: OAuth Token
CODE_PROVIDER_TOKEN=your_oauth_token
```

## Supported Features

- ✅ Repository operations (get, check access)
- ✅ File operations (read, write, update)
- ✅ Branch operations (list, get, create)
- ✅ Pull request operations (list, get, create, comment)
- ✅ Issue operations (list, get, create)
- ✅ Webhooks (push, PR, issues)
- ❌ GitHub App authentication (not supported by GitBucket)

## Limitations

GitBucket implements a subset of GitHub's API. Some features may not work:

1. **No GitHub App Support**: Use Personal Access Token or Basic Auth
2. **Partial API Coverage**: Some advanced GitHub features may not be available
3. **Rate Limiting**: May differ from GitHub's rate limits

## Usage Example

```python
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.code_provider.base.code_provider_interface import AuthMethod

# Create provider
provider = CodeProviderFactory.create_provider(
    provider_type="gitbucket",
    base_url="http://localhost:8080/api/v3"
)

# Authenticate
provider.authenticate(
    {"token": "your_pat"},
    AuthMethod.PERSONAL_ACCESS_TOKEN
)

# Use provider
repo_info = provider.get_repository("owner/repo")
```

## Webhook Setup

In your GitBucket repository settings:

1. Go to Settings → Webhooks
2. Add webhook URL: `https://your-server/api/integrations/gitbucket/webhook`
3. Select events: Push, Pull Request, Issues
4. Save webhook

## Troubleshooting

### Authentication Fails
- Verify `CODE_PROVIDER_BASE_URL` is correct (should end with `/api/v3`)
- Check PAT has required permissions in GitBucket
- For Basic Auth, verify username/password are correct

### API Errors
- Check GitBucket version (some features require v4.3+)
- Verify GitBucket instance is accessible from server
- Check GitBucket logs for detailed error messages

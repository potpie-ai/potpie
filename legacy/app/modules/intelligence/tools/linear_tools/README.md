# Linear Integration Tools

This directory contains tools for integrating with the Linear project management system.

## Overview

The Linear integration provides tools to interact with Linear issues:

- `get_linear_issue` - Fetch details of a Linear issue by ID
- `update_linear_issue` - Update a Linear issue with new details

## API Key Configuration

The Linear integration supports two methods of API key configuration:

### 1. Environment Variable (Global)

Set a global Linear API key via environment variable:

```bash
export LINEAR_API_KEY=your_linear_api_key
```

This will be used as a fallback when no user-specific key is available.

### 2. User-Specific API Keys (Recommended)

For better security and multi-user support, configure user-specific Linear API keys using the secret management system:

```python
from app.modules.key_management.secret_manager import SecretManager
from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService

@router.post("/configure-linear")
async def configure_linear_api_key(
    api_key: str,
    user=Depends(AuthService.check_auth),
    db: Session = Depends(get_db),
):
    # Create an integration key request
    from app.modules.key_management.secrets_schema import (
        CreateIntegrationKeyRequest,
        IntegrationKey,
    )

    request = CreateIntegrationKeyRequest(
        integration_keys=[
            IntegrationKey(
                service="linear",
                api_key=api_key
            )
        ]
    )

    # Store the key in the secret manager
    SecretManager.create_integration_keys(
        request=request,
        user=user,
        db=db
    )

    return {"message": "Linear API key configured successfully"}
```

## Using the Linear Tools

The Linear tools automatically use the current user's API key. User identity is required when initializing the tools:

### In ToolService

The tools are initialized with the user's ID and database session:

```python
# In ToolService._initialize_tools
tools = {
    # Other tools...
    "get_linear_issue": get_linear_issue_tool(self.db, self.user_id),
    "update_linear_issue": update_linear_issue_tool(self.db, self.user_id),
}
```

### Direct Usage

If you need to use the tools directly:

```python
from app.modules.intelligence.tools.linear_tools import get_linear_issue_tool
from sqlalchemy.orm import Session
from app.core.database import get_db

# Get database session
db = next(get_db())

# Create a tool with user context (both parameters are required)
tool = get_linear_issue_tool(db, user_id)

# Use the tool (user context is already applied)
issue_data = await tool.func(issue_id="ISSUE-123")
```

## Testing

You can test if a user's Linear API key is configured correctly:

```bash
python -m app.modules.intelligence.tools.linear_tools.test_client --user-id=user-uuid-here
```

## Implementation Details

- Each tool is a self-contained class that encapsulates user context
- The Linear client first tries to find the user-specific API key from the secret manager
- If no user-specific key is found, it falls back to the environment variable
- If neither is available, it will raise an error
- Unlike traditional tools, both `db` and `user_id` are required parameters
- User context is injected when the tool is initialized, not when it's called

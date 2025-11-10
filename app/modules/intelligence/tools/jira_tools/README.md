# Jira Integration Tools & Agent

This module provides AI agent tools for interacting with Jira using OAuth2 authentication, plus a specialized Jira Integration Agent.

## Overview

The Jira tools enable agents to perform common Jira operations on behalf of users, including:

- Fetching issue details
- Searching for issues with JQL
- Creating new issues (with proper issue types: Task, Bug, Story, etc.)
- Updating existing issues
- Adding comments
- Transitioning issues between statuses (To Do, In Progress, Done, etc.)

### Understanding Jira Terminology

**Issue Type** vs **Status** - Two distinct concepts:

- **Issue Type**: The KIND/CATEGORY of work (e.g., Task, Bug, Story, Epic)
  - Set at creation time
  - Defines what type of work item it is
  - Examples: Bug, Task, Story, Epic, Subtask
  
- **Status**: The current STATE in the workflow (e.g., To Do, In Progress, Done)
  - Changes throughout the issue's lifecycle
  - Represents where the issue is in its workflow
  - Changed via workflow transitions

## Jira Integration Agent

The `JiraIntegrationAgent` is a specialized system agent that orchestrates all Jira operations through natural language:

- **Location**: `app/modules/intelligence/agents/chat_agents/system_agents/jira_integration_agent.py`
- **Agent ID**: `jira_integration_agent`
- **Purpose**: Translates natural language requests into Jira operations
- **Tools Used**: All 6 Jira tools (get, search, create, update, comment, transition)

### Using the Agent

Users can interact with the agent using natural language:

- "Show me all high priority bugs in PROJ"
- "Create a bug about the login page crashing"
- "Move PROJ-123 to In Progress"
- "What's the status of PROJ-456?"

The agent handles JQL construction, error handling, and provides helpful responses.

## Architecture

### JiraClient (`jira_client.py`)

Core wrapper around the official Jira Python library. Handles:

- OAuth2 authentication with token refresh
- Connection to user's Jira instance
- Common Jira operations with proper error handling
- Conversion of Jira objects to dictionaries

### Helper Function

`get_jira_client_for_user(user_id, db)` - Retrieves user's Jira integration from database and creates an authenticated client

### Tools

Each tool is a LangChain StructuredTool that wraps a specific Jira operation:

- `get_jira_issue_tool.py` - Fetch issue details by key
- `search_jira_issues_tool.py` - Search issues using JQL
- `create_jira_issue_tool.py` - Create new issues
- `update_jira_issue_tool.py` - Update issue fields
- `add_jira_comment_tool.py` - Add comments to issues
- `transition_jira_issue_tool.py` - Transition issues between statuses

## Usage

### In Agent Code

```python
from app.modules.intelligence.tools.jira_tools.get_jira_issue_tool import get_jira_issue_tool

# Get the tool
jira_tool = get_jira_issue_tool(db, user_id)

# Use in agent
result = jira_tool.run(issue_key="PROJ-123")
```

### Example Response

```python
{
    "success": True,
    "issue": {
        "key": "PROJ-123",
        "summary": "Fix login bug",
        "description": "Users unable to login...",
        "status": "In Progress",
        "priority": "High",
        "assignee": "John Doe",
        "reporter": "Jane Smith",
        "created": "2025-10-31T10:30:00+00:00",
        "updated": "2025-11-01T14:15:00+00:00",
        "url": "https://yoursite.atlassian.net/browse/PROJ-123"
    }
}
```

## Authentication

Users must connect their Jira account via the Integrations page in the UI. The OAuth2 flow:

1. User initiates OAuth from frontend
2. Backend generates authorization URL
3. User authenticates with Atlassian
4. Backend receives callback with code
5. Backend exchanges code for tokens
6. Tokens stored encrypted in database
7. Tools use tokens with automatic refresh

## Token Refresh

The client automatically refreshes expired access tokens (which expire after 1 hour) using the refresh token. This happens transparently in `get_jira_client_for_user()`.

## Error Handling

All tools return structured responses with success flags:

- On success: `{"success": True, "issue": {...}}`
- On error: `{"success": False, "error": "message"}`

Common errors:

- No Jira integration found (user needs to connect)
- Token expired and refresh failed (re-authentication needed)
- Invalid issue key
- Permission denied
- Network/API errors

## Dependencies

- `jira==3.8.0` - Official Jira Python library
- Integration with potpie's integrations service
- Database access for retrieving user integrations

# Confluence Tools

A collection of LangChain tools for interacting with Confluence Cloud via the REST API v2.

## Overview

These tools enable AI agents to interact with Confluence documentation, including searching, reading, creating, updating pages, and adding comments. They use OAuth 2.0 (3LO) authentication and require users to connect their Confluence account first.

## Available Tools

### 1. Get Confluence Spaces
**Tool Name:** `get_confluence_spaces`

Lists available Confluence spaces for the user.

**Parameters:**
- `limit` (int, default=25): Maximum number of spaces to return (1-250)
- `space_type` (str, default="all"): Filter by type - "global", "personal", or "all"

**Returns:**
- List of spaces with ID, key, name, type, status, homepage ID, and description

**Example Use:**
```python
result = get_confluence_spaces_tool.run(limit=10, space_type="global")
```

---

### 2. Get Confluence Page
**Tool Name:** `get_confluence_page`

Retrieves content and metadata of a specific Confluence page by ID.

**Parameters:**
- `page_id` (str, required): The page ID
- `body_format` (str, default="storage"): Format - "storage" (HTML), "atlas_doc_format", or "view"
- `get_draft` (bool, default=False): Get draft version instead of published

**Returns:**
- Page data including title, content, author, version info, labels, space ID, parent ID

**Example Use:**
```python
result = get_confluence_page_tool.run(page_id="123456")
```

---

### 3. Search Confluence Pages
**Tool Name:** `search_confluence_pages`

Searches for Confluence pages using CQL (Confluence Query Language).

**Parameters:**
- `cql` (str, required): CQL query string
- `limit` (int, default=25): Maximum results to return (1-250)
- `include_archived` (bool, default=False): Include archived pages

**CQL Examples:**
- `type=page AND title~"API"` - Search pages with "API" in title
- `space=DEMO AND text~"documentation"` - Search text in DEMO space
- `type=page AND creator=currentUser() ORDER BY created DESC` - Your recent pages

**Returns:**
- List of matching pages with title, ID, space, author, excerpt

**Example Use:**
```python
result = search_confluence_pages_tool.run(
    cql='type=page AND space=DOCS AND text~"authentication"',
    limit=10
)
```

---

### 4. Get Confluence Space Pages
**Tool Name:** `get_confluence_space_pages`

Lists all pages within a specific Confluence space.

**Parameters:**
- `space_id` (str, required): The space ID
- `limit` (int, default=25): Maximum pages to return (1-250)
- `status` (str, default="current"): Filter - "current", "draft", "archived", or "any"

**Returns:**
- List of pages with title, ID, status, parent relationships, metadata

**Example Use:**
```python
result = get_confluence_space_pages_tool.run(space_id="123456", limit=50)
```

---

### 5. Create Confluence Page
**Tool Name:** `create_confluence_page`

Creates a new Confluence page in a space.

**Parameters:**
- `space_id` (str, required): The space ID where page will be created
- `title` (str, required): Page title
- `body` (str, required): Page content (HTML or plain text)
- `parent_id` (str, optional): Parent page ID to create as child
- `status` (str, default="current"): "current" (published) or "draft"

**Returns:**
- Created page data including ID, version, links

**Example Use:**
```python
result = create_confluence_page_tool.run(
    space_id="123456",
    title="API Documentation",
    body="<p>This is the API documentation page.</p>",
    status="current"
)
```

---

### 6. Update Confluence Page
**Tool Name:** `update_confluence_page`

Updates an existing Confluence page's title or content.

**Parameters:**
- `page_id` (str, required): The page ID to update
- `version_number` (int, required): Current version number (get from get_confluence_page)
- `title` (str, optional): New title
- `body` (str, optional): New content (HTML or plain text)
- `status` (str, default="current"): "current" or "draft"

**Important:** Must provide the current version number to prevent conflicts. Use `get_confluence_page` first to retrieve it.

**Returns:**
- Updated page data including new version number

**Example Use:**
```python
# First get current version
page = get_confluence_page_tool.run(page_id="123456")
version = page["page"]["version"]["number"]

# Then update
result = update_confluence_page_tool.run(
    page_id="123456",
    version_number=version,
    title="Updated API Documentation",
    body="<p>Updated content here.</p>"
)
```

---

### 7. Add Confluence Comment
**Tool Name:** `add_confluence_comment`

Adds a comment to a Confluence page or replies to an existing comment.

**Parameters:**
- `page_id` (str, required): The page ID to comment on
- `comment` (str, required): Comment text (plain text or HTML)
- `parent_comment_id` (str, optional): Parent comment ID for replies
- `status` (str, default="current"): "current" (published) or "draft"

**Returns:**
- Created comment data including ID, author, timestamp

**Example Use:**
```python
# Top-level comment
result = add_confluence_comment_tool.run(
    page_id="123456",
    comment="Great documentation!"
)

# Reply to comment
result = add_confluence_comment_tool.run(
    page_id="123456",
    comment="Thanks for the feedback!",
    parent_comment_id="789"
)
```

---

## Setup & Authentication

### Prerequisites
1. Confluence Cloud account
2. OAuth 2.0 app configured in Atlassian Developer Console
3. Environment variables set:
   - `CONFLUENCE_CLIENT_ID`
   - `CONFLUENCE_CLIENT_SECRET`
   - `CONFLUENCE_REDIRECT_URI`

### Required Scopes
The following OAuth scopes are required:
- `read:confluence-space.summary`
- `read:confluence-content.all`
- `read:confluence-content.summary`
- `search:confluence`
- `write:confluence-content`
- `readonly:content.attachment:confluence`
- `read:confluence-props`
- `read:confluence-content.permission`
- `read:confluence-user`
- `read:confluence-groups`
- `write:confluence-props`
- `write:confluence-file`
- `read:audit-log:confluence`
- `read:analytics.content:confluence`

### Integration Flow
1. User initiates OAuth flow via `/integrations/confluence`
2. User authorizes app on Atlassian
3. OAuth callback exchanges code for tokens
4. Integration saved to database with encrypted tokens
5. Tools use stored tokens for API requests

## Architecture

### Client Layer
**File:** `confluence_client.py`

The `ConfluenceClient` class provides a low-level HTTP client for Confluence REST API v2:
- OAuth 2.0 Bearer token authentication
- Automatic token refresh
- Context manager support
- Storage format conversion (HTML ↔ text)
- Error handling and logging

### Tool Layer
Each tool is implemented as:
1. **Input Schema** (Pydantic BaseModel) - Validates parameters
2. **Tool Class** - Contains business logic in `run()` method
3. **Factory Function** - Creates LangChain `StructuredTool` instance

### Integration Check
All tools verify Confluence integration exists before executing:
```python
has_integration = await check_confluence_integration_exists(user_id, db)
```

## Usage in Agents

### Getting All Tools
```python
from app.modules.intelligence.tools.confluence_tools import get_all_confluence_tools

# Get all 7 Confluence tools for a user
tools = get_all_confluence_tools(db, user_id)
```

### Individual Tools
```python
from app.modules.intelligence.tools.confluence_tools import (
    search_confluence_pages_tool,
    get_confluence_page_tool,
    create_confluence_page_tool,
)

search_tool = search_confluence_pages_tool(db, user_id)
get_tool = get_confluence_page_tool(db, user_id)
create_tool = create_confluence_page_tool(db, user_id)
```

### Integration with LangChain Agents
```python
from langchain.agents import initialize_agent, AgentType

tools = get_all_confluence_tools(db, user_id)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Search for API documentation in Confluence")
```

## API Reference

### Base URL
```
https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2/
```

### Endpoints Used
- `GET /spaces` - List spaces
- `GET /spaces/{id}` - Get space details
- `GET /spaces/{id}/pages` - Get pages in space
- `GET /pages/{id}` - Get page
- `GET /pages` - Search pages (with CQL)
- `POST /pages` - Create page
- `PUT /pages/{id}` - Update page
- `POST /pages/{id}/footer-comments` - Add comment

### Storage Format
Confluence uses **HTML storage format** for page content, unlike Jira's Atlassian Document Format (ADF). The API accepts and returns HTML directly in the storage format.

## Error Handling

All tools return standardized response format:
```python
{
    "success": bool,
    "data": {...},  # Tool-specific data
    "error": str    # Present only if success=False
}
```

Common errors:
- **No integration found** - User hasn't connected Confluence
- **Invalid page ID** - Page doesn't exist or no access
- **Version conflict** - Page was updated since version retrieved
- **Permission denied** - User lacks permission for operation
- **Invalid CQL** - Malformed query syntax

## Differences from Jira Tools

1. **No Webhooks** - Confluence OAuth 2.0 doesn't support webhooks (unlike Jira)
2. **Storage Format** - Uses HTML storage format instead of ADF
3. **Spaces vs Projects** - Confluence has spaces, Jira has projects
4. **CQL vs JQL** - Different query languages (similar syntax)
5. **Comments** - Footer comments (not inline like Jira)
6. **No Issues** - Pages are the primary content type

## Best Practices

1. **Always get version before updating** - Prevents conflicts
2. **Use CQL for complex searches** - More powerful than basic filters
3. **Check space ID before creating** - Verify space exists
4. **Validate HTML content** - Ensure well-formed HTML for storage format
5. **Handle pagination** - Use `has_more` flag for large result sets
6. **Cache space lists** - Avoid repeated calls to get_spaces

## Limitations

- Maximum 250 results per request (use pagination for more)
- Version number required for updates (prevents concurrent edits)
- OAuth 2.0 limitations (no webhooks, no impersonation)
- HTML storage format only (no Markdown support in API v2)
- Comments are footer comments (not inline)

## Future Enhancements

Potential additions:
- Get page attachments
- Upload attachments
- Manage page labels
- Get page history/versions
- Archive/restore pages
- Manage page permissions
- Get page analytics

## Support

For issues or questions:
1. Check Confluence REST API v2 documentation: https://developer.atlassian.com/cloud/confluence/rest/v2/intro/
2. Verify OAuth scopes are configured correctly
3. Check logs for detailed error messages
4. Ensure user has proper permissions in Confluence

## License

Same as parent project.

"""
Confluence OAuth integration using Atlassian 3LO

Confluence uses the same OAuth infrastructure as Jira (Atlassian 3LO),
with product-specific scopes and API endpoints.

NOTE: Unlike Jira, Confluence does NOT support programmatic webhook registration
via OAuth 2.0 apps. Webhooks are only available for Connect apps registered in
the app descriptor. See: https://developer.atlassian.com/cloud/confluence/modules/webhook/
"""

from starlette.config import Config
from .atlassian_oauth_base import AtlassianOAuthBase


class ConfluenceOAuth(AtlassianOAuthBase):
    """Confluence OAuth integration handler using Atlassian 3LO"""

    @property
    def product_name(self) -> str:
        """Product identifier for Confluence"""
        return "confluence"

    @property
    def default_scope(self) -> str:
        """
        Default OAuth scopes for Confluence integration

        Scopes configured by user:
        - write:confluence-content: Create/update pages, blogs, comments
        - read:confluence-space.summary: Read space information
        - write:confluence-space: Create/manage spaces
        - write:confluence-file: Upload/manage attachments
        - read:confluence-props: Read content properties
        - write:confluence-props: Write content properties
        - read:confluence-content.all: Read all content
        - read:confluence-content.summary: Read content summaries
        - search:confluence: Search functionality (CQL)
        - read:confluence-content.permission: Read content permissions
        - read:confluence-user: Read user information
        - read:confluence-groups: Read group information
        - readonly:content.attachment:confluence: Read attachments
        - offline_access: Refresh token support
        """
        return self.config(
            "CONFLUENCE_OAUTH_SCOPE",
            default=(
                "write:confluence-content "
                "read:confluence-space.summary "
                "write:confluence-space "
                "write:confluence-file "
                "read:confluence-props "
                "write:confluence-props "
                "read:confluence-content.all "
                "read:confluence-content.summary "
                "search:confluence "
                "read:confluence-content.permission "
                "read:confluence-user "
                "read:confluence-groups "
                "readonly:content.attachment:confluence "
                "offline_access"
            ),
        )

    def __init__(self, config: Config) -> None:
        """Initialize Confluence OAuth with configuration"""
        super().__init__(config)

    # Note: Unlike JiraOAuth, we don't include webhook methods here
    # because Confluence OAuth 2.0 apps cannot register webhooks via API.
    # Webhooks in Confluence are only available for Connect apps.
    #
    # If webhook functionality is needed in the future, users would need to:
    # 1. Create a Connect app descriptor
    # 2. Register webhooks in the descriptor
    # 3. Deploy as a Connect app instead of pure OAuth 2.0 integration

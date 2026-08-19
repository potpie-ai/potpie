"""GitLab source connector package.

Public surface mirrors ``connectors/github``: the connector class, the
connector-internal read port and its REST implementation, and the agent
tool builder.
"""

from potpie_context_engine.adapters.outbound.connectors.gitlab.api_client import (
    GitLabApiError,
    GitLabReadPort,
    GitLabRestSourceControl,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.agent_tools import (
    build_gitlab_tools,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.connector import (
    GitLabConnector,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.graphql_client import (
    GitLabGraphQLClient,
    graphql_enabled,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.resolver import (
    GitLabMergeRequestResolver,
)

__all__ = [
    "GitLabApiError",
    "GitLabConnector",
    "GitLabGraphQLClient",
    "GitLabMergeRequestResolver",
    "GitLabReadPort",
    "GitLabRestSourceControl",
    "build_gitlab_tools",
    "graphql_enabled",
]

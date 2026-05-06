"""Source resolver adapters for ``source_policy`` modes."""

from adapters.outbound.source_resolvers.composite import CompositeSourceResolver
from adapters.outbound.source_resolvers.documentation import DocumentationUriResolver
from adapters.outbound.source_resolvers.github_pull_request import (
    GitHubPullRequestResolver,
)
from adapters.outbound.source_resolvers.null import NullSourceResolver

__all__ = [
    "CompositeSourceResolver",
    "DocumentationUriResolver",
    "GitHubPullRequestResolver",
    "NullSourceResolver",
]

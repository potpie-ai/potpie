"""Source resolver adapters for ``source_policy`` modes."""

from potpie.context_engine.adapters.outbound.connectors.github.resolver import GitHubPullRequestResolver
from potpie.context_engine.adapters.outbound.source_resolvers.composite import CompositeSourceResolver
from potpie.context_engine.adapters.outbound.source_resolvers.documentation import DocumentationUriResolver
from potpie.context_engine.adapters.outbound.source_resolvers.null import NullSourceResolver

__all__ = [
    "CompositeSourceResolver",
    "DocumentationUriResolver",
    "GitHubPullRequestResolver",
    "NullSourceResolver",
]

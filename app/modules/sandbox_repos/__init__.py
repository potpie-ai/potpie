"""Adapters layered on top of the standalone sandbox module.

The sandbox library defines auth/identity/PR concerns as ports; these
adapters bridge them to potpie's existing auth chain so the GitHub App
installation token / user OAuth / env fallback is shared end-to-end:

* :class:`GitHubGitPlatformProvider` — PR creation through
  ``app.modules.code_provider.github.GitHubProvider``.
* :class:`PotpieBotIdentityProvider` — stamps ``potpie-ai[bot]`` on
  every commit the sandbox issues.
* :class:`PotpieRemoteAuthProvider` — re-injects auth on push/fetch
  per call (the bare clone scrubs the token, so push needs its own
  resolution path).
"""

from app.modules.sandbox_repos.git_platform import GitHubGitPlatformProvider
from app.modules.sandbox_repos.identity import PotpieBotIdentityProvider
from app.modules.sandbox_repos.remote_auth import PotpieRemoteAuthProvider

__all__ = [
    "GitHubGitPlatformProvider",
    "PotpieBotIdentityProvider",
    "PotpieRemoteAuthProvider",
]

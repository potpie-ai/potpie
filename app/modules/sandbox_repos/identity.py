"""Bot identity adapter for the sandbox.

Implements :class:`sandbox.BotIdentityProvider` so every commit the
sandbox issues — whether through ``SandboxClient.commit`` or raw
``sandbox_shell git commit`` — is attributed to one canonical Potpie
bot identity rather than whatever ``git config user.email`` happened to
be set on the host.

Identity resolution (in priority order):

1. ``POTPIE_BOT_NAME`` / ``POTPIE_BOT_EMAIL`` env vars — explicit
   override for ops who want a custom display name.
2. The configured GitHub App slug — when ``GITHUB_APP_ID`` is set we
   derive ``<slug>[bot]`` + ``<numeric_id>+<slug>[bot]@users.noreply.github.com``.
   The numeric id is looked up once per process via
   ``GET /users/<slug>[bot]`` and cached.
3. Hardcoded fallback ``("potpie-ai[bot]", "potpie-ai-bot@users.noreply.github.com")``
   — keeps dev environments working without any env config.

We deliberately do NOT fall back to the calling user's identity. The
agent is the bot; commits should be attributable to the bot regardless
of who triggered the run. Callers that need user attribution can pass
``author=...`` to :meth:`SandboxClient.commit` explicitly.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sandbox import Author, RepoIdentity


logger = logging.getLogger(__name__)


_DEFAULT_BOT_NAME = "potpie-ai[bot]"
_DEFAULT_BOT_EMAIL = "potpie-ai-bot@users.noreply.github.com"


class PotpieBotIdentityProvider:
    """Resolve the Potpie-bot author for sandbox commits."""

    def __init__(
        self,
        *,
        name: str | None = None,
        email: str | None = None,
        github_app_slug: str | None = None,
    ) -> None:
        self._explicit_name = name or os.getenv("POTPIE_BOT_NAME")
        self._explicit_email = email or os.getenv("POTPIE_BOT_EMAIL")
        # The App slug is what GitHub uses in the bot username. Ops can
        # override it explicitly; otherwise we derive from the App ID
        # via a lazy lookup.
        self._github_app_slug = github_app_slug or os.getenv("GITHUB_APP_SLUG")
        self._cached_author: Optional[Author] = None

    async def identity_for_repo(
        self, *, repo: RepoIdentity, user_id: str | None = None
    ) -> Author:
        del repo, user_id  # bot identity is repo-agnostic by design
        if self._cached_author is not None:
            return self._cached_author

        name = self._explicit_name or self._derive_app_bot_name() or _DEFAULT_BOT_NAME
        email = (
            self._explicit_email
            or self._derive_app_bot_email(name)
            or _DEFAULT_BOT_EMAIL
        )
        self._cached_author = Author(name=name, email=email)
        return self._cached_author

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _derive_app_bot_name(self) -> str | None:
        if self._github_app_slug:
            return f"{self._github_app_slug}[bot]"
        return None

    def _derive_app_bot_email(self, bot_name: str) -> str | None:
        """Build the noreply email if we know the App's numeric user id.

        GitHub Apps expose a numeric id at ``GET /users/<slug>[bot]``;
        the bot's noreply email is ``<id>+<slug>[bot]@users.noreply.github.com``.
        We only fetch this when the slug is known and ``GITHUB_APP_ID``
        is configured (the lookup needs no auth, but skipping it when
        no app is set keeps dev runs offline-clean).
        """
        if not self._github_app_slug:
            return None
        if not os.getenv("GITHUB_APP_ID"):
            return None
        try:
            import requests

            response = requests.get(
                f"https://api.github.com/users/{bot_name}",
                headers={"Accept": "application/vnd.github+json"},
                timeout=5,
            )
            if response.status_code != 200:
                logger.debug(
                    "PotpieBotIdentityProvider: lookup of %s returned %d, "
                    "falling back to default email",
                    bot_name,
                    response.status_code,
                )
                return None
            user_id = response.json().get("id")
            if not user_id:
                return None
            return f"{user_id}+{bot_name}@users.noreply.github.com"
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "PotpieBotIdentityProvider: failed to look up bot email: %r", exc
            )
            return None

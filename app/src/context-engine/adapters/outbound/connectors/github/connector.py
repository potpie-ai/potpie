"""GitHub source connector — single point of contact for everything GitHub.

Bundles four GitHub-shaped surfaces behind one class:

- Read access to PRs/issues → :attr:`api_client`
- Reference resolution → :meth:`fetch`
- Webhook signature + parsing (called from
  ``adapters/inbound/http/webhooks/integrations/github.py``) →
  :meth:`normalize_webhook`
- Agent tool bindings → :func:`build_github_agent_tools`

The application layer never imports anything in this module.

Rebuild plan P0: removed ``propose_plan`` (deterministic PR-merge plan
compilation). Webhooks now produce raw :class:`ContextEvent`\\s; the P5
deterministic activity layer + LLM reconciliation agent turn them into
:RELATES_TO claims.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from uuid import uuid4

from adapters.outbound.connectors.github.api_client import (
    GitHubReadPort,
    PyGithubSourceControl,
)
from adapters.outbound.connectors.github.resolver import GitHubPullRequestResolver
from domain.context_events import ContextEvent
from domain.ports.source_connector import SourceConnectorPort
from domain.source_connector import ConnectorScope, SourceCapability
from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    ResolverAuthContext,
    ResolverBudget,
    SourceResolutionResult,
)

logger = logging.getLogger(__name__)


SourceControlFactory = Callable[[str], GitHubReadPort]


class GitHubConnector(SourceConnectorPort):
    """The unified GitHub connector.

    The connector is constructed with a ``source_for_repo`` factory and a
    ``repo_resolver`` that maps ``pot_id → repo_name`` so the engine never
    needs to import GitHub-specific identifiers anywhere else.
    """

    KIND = "github"

    def __init__(
        self,
        *,
        source_for_repo: SourceControlFactory,
        repo_resolver: Callable[[str, SourceReferenceRecord], str | None] | None = None,
        webhook_secret: str | None = None,
        allow_unsigned: bool = False,
    ) -> None:
        self._source_for_repo = source_for_repo
        self._repo_resolver = repo_resolver or _default_repo_resolver
        self._webhook_secret = (webhook_secret or "").strip() or None
        self._allow_unsigned = allow_unsigned
        self._unsigned_warned = False
        self._resolver = GitHubPullRequestResolver(
            source_for_repo=source_for_repo,
            repo_resolver=repo_resolver,
        )

    # ------------------------------------------------------------------
    # SourceConnectorPort
    # ------------------------------------------------------------------
    def kind(self) -> str:
        return self.KIND

    def capabilities(self) -> Sequence[SourceCapability]:
        # Mirror the resolver's capability matrix.
        resolver_caps = self._resolver.capabilities()
        out: list[SourceCapability] = []
        for cap in resolver_caps:
            out.append(
                SourceCapability(
                    provider=cap.provider,
                    source_kind=cap.source_kind,
                    policies=cap.policies,
                    fetch_capable=True,
                    list_capable=cap.source_kind in {"pull_request", "issue"},
                    webhook_capable=True,
                    sync_capable=cap.source_kind == "pull_request",
                )
            )
        if not out:
            # Connector available but no fetch policies (PyGithub disabled).
            out.append(
                SourceCapability(
                    provider="github",
                    source_kind="pull_request",
                    policies=frozenset(),
                    fetch_capable=False,
                    list_capable=False,
                    webhook_capable=bool(self._webhook_secret),
                    sync_capable=True,
                    notes="github_token unavailable",
                )
            )
        return out

    def list_artifacts(
        self,
        scope: ConnectorScope,
    ) -> Iterable[SourceReferenceRecord]:
        repo_name = (scope.scope.get("repo_name") or "").strip()
        if not repo_name:
            return ()
        try:
            client = self._source_for_repo(repo_name)
        except Exception as exc:
            logger.warning("github list_artifacts: source_for_repo failed: %s", exc)
            return ()
        out: list[SourceReferenceRecord] = []
        for pr in client.iter_closed_pulls(repo_name):
            if not getattr(pr, "merged_at", None):
                continue
            number = getattr(pr, "number", None)
            if number is None:
                continue
            out.append(
                SourceReferenceRecord(
                    ref=f"github:pr:{repo_name}:{number}",
                    source_type="pull_request",
                    source_system="github",
                    external_id=str(number),
                    fetchable=True,
                    access="allowed",
                )
            )
        return out

    def normalize_webhook(
        self,
        payload: bytes,
        headers: Mapping[str, str],
    ) -> ContextEvent | None:
        signature = headers.get("X-Hub-Signature-256") or headers.get(
            "x-hub-signature-256"
        )
        if self._webhook_secret is None:
            # Fail closed: an unsigned webhook is an unauthenticated graph
            # write + a free trigger for expensive agent work. Only a loud,
            # explicit dev opt-in may bypass the signature requirement.
            if not self._allow_unsigned:
                raise PermissionError(
                    "github webhook signature required: GITHUB_WEBHOOK_SECRET "
                    "is not configured (set it, or set "
                    "CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS=1 for local dev "
                    "only)"
                )
            if not self._unsigned_warned:
                logger.warning(
                    "SECURITY: GITHUB_WEBHOOK_SECRET is unset and "
                    "CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS is enabled — "
                    "github webhooks are being accepted UNAUTHENTICATED. "
                    "Never use this in a network-reachable deployment."
                )
                self._unsigned_warned = True
        elif not _verify_signature(payload, signature, self._webhook_secret):
            raise PermissionError("github webhook signature mismatch")

        event_name = headers.get("X-GitHub-Event") or headers.get("x-github-event") or ""
        if event_name != "pull_request":
            return None
        try:
            body = json.loads(payload.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return None
        action = body.get("action")
        pr = body.get("pull_request") or {}
        if action != "closed" or not pr.get("merged"):
            return None

        repository = body.get("repository") or {}
        repo = repository.get("full_name")
        pr_number = pr.get("number")
        if not repo or pr_number is None:
            return None

        provider_host = _provider_host_from_repo(repository) or "github.com"

        delivery_id = (
            headers.get("X-GitHub-Delivery") or headers.get("x-github-delivery") or ""
        )
        sender_login = ((pr.get("user") or {}).get("login") or "").strip() or None
        return ContextEvent(
            event_id=str(uuid4()),
            source_system="github",
            event_type="pull_request",
            action="merged",
            pot_id="",  # filled in by the inbound dispatcher per pot mapping
            provider="github",
            provider_host=provider_host,
            repo_name=repo,
            source_id=f"pr_{int(pr_number)}_merged",
            source_event_id=str(delivery_id) or None,
            payload={
                "pr_number": int(pr_number),
                "repo_name": repo,
                "sender_login": sender_login,
                "is_live_bridge": True,
            },
        )

    async def fetch(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        return await self._resolver.resolve(
            pot_id=pot_id,
            refs=refs,
            source_policy=source_policy,
            budget=budget,
            auth=auth,
        )

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _provider_host_from_repo(repository: Mapping[str, Any]) -> str | None:
    """Derive provider host from a webhook ``repository`` object.

    Webhooks from GitHub Enterprise Server carry a host other than
    ``github.com``; ``html_url`` is the only consistently-present field
    that surfaces it.
    """
    for key in ("html_url", "url", "clone_url", "git_url"):
        raw = repository.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        host = urlparse(raw).hostname
        if host:
            return host.lower()
    return None


def _verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, f"sha256={expected}")


def _default_repo_resolver(_pot_id: str, ref: SourceReferenceRecord) -> str | None:
    hint = ref.resolver_hint or {}
    if isinstance(hint, dict) and hint.get("repo_name"):
        return str(hint["repo_name"])
    return None


__all__ = [
    "GitHubConnector",
    "GitHubReadPort",
    "PyGithubSourceControl",
    "SourceControlFactory",
]

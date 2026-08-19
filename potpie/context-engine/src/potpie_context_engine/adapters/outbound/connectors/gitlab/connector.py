"""GitLab source connector — single point of contact for everything GitLab.

Bundles the same four GitLab-shaped surfaces the GitHub connector bundles:

- Read access to MRs/issues → :class:`GitLabReadPort`
- Reference resolution → :meth:`fetch`
- Webhook token check + parsing (called from
  ``adapters/inbound/http/webhooks/integrations/gitlab.py``) →
  :meth:`normalize_webhook`
- Agent tool bindings → :func:`build_gitlab_tools`

The application layer never imports anything in this module.

Verified against GitLab CE 19.2. Two things differ structurally from
GitHub and are handled here rather than leaking outward:

1. GitLab authenticates webhooks with a **shared secret echoed verbatim**
   in ``X-Gitlab-Token``, not an HMAC over the body. The comparison is
   still constant-time, and the fail-closed posture is identical.
2. A project path may contain nested groups (``group/sub/project``), so a
   repo name is split from the right, never at the first slash.
"""

from __future__ import annotations

import hmac
import json
import logging
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from uuid import uuid4

from potpie_context_engine.adapters.outbound.connectors.gitlab.api_client import (
    GitLabReadPort,
    GitLabRestSourceControl,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.resolver import (
    GitLabMergeRequestResolver,
)
from potpie_context_core.context_events import ContextEvent
from potpie_context_core.source_references import SourceReferenceRecord
from potpie_context_engine.domain.ports.source_connector import SourceConnectorPort
from potpie_context_engine.domain.source_connector import (
    ConnectorScope,
    SourceCapability,
)
from potpie_context_engine.domain.source_resolution import (
    ResolverAuthContext,
    ResolverBudget,
    SourceResolutionResult,
)

logger = logging.getLogger(__name__)


SourceControlFactory = Callable[[str], GitLabReadPort]

# GitLab names its webhook kinds in the X-Gitlab-Event header.
_MERGE_REQUEST_EVENT = "merge request hook"
_ISSUE_EVENT = "issue hook"


class GitLabConnector(SourceConnectorPort):
    """The unified GitLab connector.

    Constructed with a ``source_for_project`` factory and a
    ``project_resolver`` that maps ``pot_id → project path`` so the engine
    never needs to import GitLab-specific identifiers anywhere else.
    """

    KIND = "gitlab"

    def __init__(
        self,
        *,
        source_for_project: SourceControlFactory,
        project_resolver: Callable[[str, SourceReferenceRecord], str | None]
        | None = None,
        webhook_secret: str | None = None,
        allow_unsigned: bool = False,
        instance_host: str = "gitlab.com",
    ) -> None:
        self._source_for_project = source_for_project
        self._project_resolver = project_resolver or _default_project_resolver
        self._webhook_secret = (webhook_secret or "").strip() or None
        self._allow_unsigned = allow_unsigned
        self._unsigned_warned = False
        self._instance_host = (instance_host or "gitlab.com").strip().lower()
        self._resolver = GitLabMergeRequestResolver(
            source_for_project=source_for_project,
            project_resolver=self._project_resolver,
        )

    # ------------------------------------------------------------------
    # SourceConnectorPort
    # ------------------------------------------------------------------
    def kind(self) -> str:
        return self.KIND

    def capabilities(self) -> Sequence[SourceCapability]:
        resolver_caps = self._resolver.capabilities()
        out: list[SourceCapability] = []
        for cap in resolver_caps:
            out.append(
                SourceCapability(
                    provider=cap.provider,
                    source_kind=cap.source_kind,
                    policies=cap.policies,
                    fetch_capable=True,
                    list_capable=True,
                    webhook_capable=True,
                    sync_capable=True,
                )
            )
        for source_kind in ("merge_request", "issue"):
            out.append(
                SourceCapability(
                    provider="gitlab",
                    source_kind=source_kind,
                    policies=frozenset({"summary", "verify", "snippets"}),
                    fetch_capable=True,
                    list_capable=True,
                    webhook_capable=True,
                    sync_capable=source_kind == "merge_request",
                )
            )
        return out

    def list_artifacts(
        self,
        scope: ConnectorScope,
    ) -> Iterable[SourceReferenceRecord]:
        project = _project_from_scope(scope)
        if not project:
            return ()
        try:
            client = self._source_for_project(project)
        except Exception as exc:
            logger.warning("gitlab list_artifacts: source_for_project failed: %s", exc)
            return ()
        out: list[SourceReferenceRecord] = []
        try:
            for mr in client.iter_merged_merge_requests(project):
                iid = mr.get("iid") or mr.get("number")
                if iid is None:
                    continue
                out.append(
                    SourceReferenceRecord(
                        ref=f"gitlab:mr:{project}:{iid}",
                        source_type="merge_request",
                        source_system="gitlab",
                        external_id=str(iid),
                        fetchable=True,
                        access="allowed",
                    )
                )
        except Exception as exc:
            logger.warning("gitlab list_artifacts: enumeration failed: %s", exc)
        return out

    def normalize_webhook(
        self,
        payload: bytes,
        headers: Mapping[str, str],
    ) -> ContextEvent | None:
        self._authorize_webhook(headers)

        event_name = (
            (headers.get("X-Gitlab-Event") or headers.get("x-gitlab-event") or "")
            .strip()
            .lower()
        )
        if event_name not in (_MERGE_REQUEST_EVENT, _ISSUE_EVENT):
            return None
        try:
            body = json.loads(payload.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(body, dict):
            return None

        project = body.get("project") or {}
        repo_name = (project.get("path_with_namespace") or "").strip()
        if not repo_name:
            return None
        attrs = body.get("object_attributes") or {}
        iid = attrs.get("iid")
        if iid is None:
            return None

        if event_name == _MERGE_REQUEST_EVENT:
            # "merge" is the action GitLab reports when an MR is merged;
            # some versions only flip state, so accept either signal.
            if attrs.get("action") != "merge" and attrs.get("state") != "merged":
                return None
            event_type, action = "merge_request", "merged"
            source_id = f"mr_{int(iid)}_merged"
        else:
            if attrs.get("action") != "open":
                return None
            event_type, action = "issue", "opened"
            source_id = f"issue_{int(iid)}_opened"

        provider_host = _provider_host_from_project(project) or self._instance_host
        delivery_id = (
            headers.get("X-Gitlab-Event-UUID")
            or headers.get("x-gitlab-event-uuid")
            or ""
        )
        sender_login = ((body.get("user") or {}).get("username") or "").strip() or None
        return ContextEvent(
            event_id=str(uuid4()),
            source_system="gitlab",
            event_type=event_type,
            action=action,
            pot_id="",  # filled in by the inbound dispatcher per pot mapping
            provider="gitlab",
            provider_host=provider_host,
            repo_name=repo_name,
            source_id=source_id,
            source_event_id=str(delivery_id) or None,
            payload={
                "iid": int(iid),
                "project_path": repo_name,
                "repo_name": repo_name,
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _authorize_webhook(self, headers: Mapping[str, str]) -> None:
        """Fail closed unless the delivery carries the configured secret token.

        An unsigned webhook is an unauthenticated graph write plus a free
        trigger for expensive agent work; only a loud, explicit dev opt-in
        may bypass the check.
        """
        token = headers.get("X-Gitlab-Token") or headers.get("x-gitlab-token")
        if self._webhook_secret is None:
            if not self._allow_unsigned:
                raise PermissionError(
                    "gitlab webhook secret token required: GITLAB_WEBHOOK_SECRET "
                    "is not configured (set it, or set "
                    "CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS=1 for local dev "
                    "only)"
                )
            if not self._unsigned_warned:
                logger.warning(
                    "SECURITY: GITLAB_WEBHOOK_SECRET is unset and "
                    "CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS is enabled — "
                    "gitlab webhooks are being accepted UNAUTHENTICATED. "
                    "Never use this in a network-reachable deployment."
                )
                self._unsigned_warned = True
            return
        # Compare as bytes: ``compare_digest`` rejects str arguments that are
        # not ASCII-only, and a secret token is operator-chosen text.
        if not token or not hmac.compare_digest(
            token.encode("utf-8"), self._webhook_secret.encode("utf-8")
        ):
            raise PermissionError("gitlab webhook secret token mismatch")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _project_from_scope(scope: ConnectorScope) -> str:
    """Read the project path from a connector scope.

    ``repo_name`` is accepted alongside ``project_path`` so callers that
    speak the shared repo vocabulary (the pot resolution port, the
    backfill job) do not need a GitLab-specific branch.
    """
    raw = scope.scope.get("project_path") or scope.scope.get("repo_name") or ""
    return str(raw).strip().strip("/")


def _provider_host_from_project(project: Mapping[str, Any]) -> str | None:
    """Derive the instance host from a webhook ``project`` object.

    Self-managed CE lives on an arbitrary host, and the payload's URLs are
    the only place it appears.
    """
    for key in ("web_url", "homepage", "http_url", "git_http_url", "url"):
        raw = project.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        host = urlparse(raw).hostname
        if host:
            return host.lower()
    return None


def _default_project_resolver(_pot_id: str, ref: SourceReferenceRecord) -> str | None:
    hint = ref.resolver_hint or {}
    if isinstance(hint, dict):
        for key in ("project_path", "repo_name"):
            if hint.get(key):
                return str(hint[key])
    return None


__all__ = [
    "GitLabConnector",
    "GitLabReadPort",
    "GitLabRestSourceControl",
    "SourceControlFactory",
]

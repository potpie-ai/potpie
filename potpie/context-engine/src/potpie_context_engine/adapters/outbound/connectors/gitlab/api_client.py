"""GitLab API client used internally by :class:`GitLabConnector`.

This module owns the connector's read access to GitLab. The
``GitLabReadPort`` Protocol below is the narrow internal contract tests
substitute via fakes; it is not exported as a domain port (it lives
behind the connector boundary).

Transport is plain REST v4 over the shared :class:`AuthHttpClient` — no
GitLab SDK dependency. Everything the GitHub connector reaches through
PyGithub has a REST v4 counterpart here, and the returned dict keys are
deliberately aligned with :class:`GitHubReadPort` (``title``, ``body``,
``state``, ``merged_at``, ``head_branch``, ``base_branch``, ``url``,
``author``, ``labels``, …) so playbooks and graph mutations share one
vocabulary across forges. GitLab-native fields that have no GitHub
equivalent (approvals, resolvable discussion threads, weight, time
tracking) are added on top rather than dropped.

Verified against GitLab CE 19.2 (REST API v4).
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Protocol
from urllib.parse import quote, urlparse

from potpie_context_engine.adapters.outbound.cli_auth.gitlab_client import (
    gitlab_auth_headers,
    normalize_instance_url,
)
from potpie_context_engine.adapters.outbound.cli_auth.http import (
    AuthHttpClient,
    AuthHttpError,
    HttpClient,
)
from potpie_context_engine.adapters.outbound.cli_auth.provider_config import (
    GITLAB_DEFAULT_INSTANCE,
    gitlab_api_base_url,
)
from potpie_context_engine.domain.backfill_window import (
    backfill_window_since,
    clamp_backfill_limit,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0
# GitLab caps per_page at 100 regardless of what the caller asks for.
_MAX_PER_PAGE = 100
# Hard ceiling on pages walked by one list call, so a misconfigured window
# can never turn a single tool call into an unbounded crawl.
_MAX_PAGES = 20


class GitLabApiError(Exception):
    """A GitLab REST call failed in a way the caller should surface."""


class GitLabReadPort(Protocol):
    """Connector-internal read surface over GitLab.

    Mirrors ``GitHubReadPort`` verb for verb; ``project`` is a
    ``group/subgroup/project`` path (or a numeric project id) and ``iid``
    is the project-scoped internal id GitLab shows in the UI (``!42`` /
    ``#42``), not the global id.
    """

    def get_merge_request(
        self,
        project: str,
        iid: int,
        include_diff: bool = False,
    ) -> dict[str, Any]: ...

    def get_merge_request_commits(
        self, project: str, iid: int
    ) -> list[dict[str, Any]]: ...

    def get_merge_request_discussions(
        self,
        project: str,
        iid: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    def get_merge_request_notes(
        self,
        project: str,
        iid: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def get_merge_request_approvals(self, project: str, iid: int) -> dict[str, Any]: ...

    def get_merge_request_state_events(
        self, project: str, iid: int
    ) -> list[dict[str, Any]]: ...

    def get_merge_request_closes_issues(
        self, project: str, iid: int
    ) -> list[dict[str, Any]]: ...

    def get_issue(self, project: str, iid: int) -> dict[str, Any]: ...

    def get_issue_notes(
        self,
        project: str,
        iid: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]: ...

    def get_issue_links(self, project: str, iid: int) -> dict[str, Any]: ...

    def iter_merged_merge_requests(self, project: str) -> Iterator[dict[str, Any]]: ...

    def list_merge_requests(
        self,
        project: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def list_issues(
        self,
        project: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...


class GitLabRestSourceControl(GitLabReadPort):
    """REST v4 implementation of :class:`GitLabReadPort`.

    Args:
        instance_url: Base URL of the GitLab instance. Self-managed CE
            installs mounted under a subpath are handled by
            :func:`normalize_instance_url`.
        token: Personal access token with at least ``read_api``.
        http: Injected transport; tests pass a fake implementing
            :class:`HttpClient`. When omitted the client owns an
            :class:`AuthHttpClient` and never closes it implicitly — call
            :meth:`close`.
        graphql: Optional review-history fast path. When present,
            :meth:`get_merge_request_review_history` tries it first and
            falls back to the REST composition on any failure.
    """

    def __init__(
        self,
        instance_url: str,
        token: str,
        *,
        http: HttpClient | None = None,
        graphql: Any | None = None,
    ) -> None:
        self._instance_url = (
            normalize_instance_url(instance_url) or GITLAB_DEFAULT_INSTANCE
        )
        self._api_base = gitlab_api_base_url(self._instance_url)
        self._headers = gitlab_auth_headers(token)
        self._http = http or AuthHttpClient(timeout=_HTTP_TIMEOUT)
        self._owns_http = http is None
        self._graphql = graphql

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    @property
    def instance_url(self) -> str:
        return self._instance_url

    @property
    def instance_host(self) -> str:
        return (urlparse(self._instance_url).hostname or "").lower()

    def close(self) -> None:
        if self._owns_http:
            close = getattr(self._http, "close", None)
            if callable(close):
                close()

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allow_missing: bool = False,
    ) -> Any:
        """GET ``{api_base}{path}``.

        ``allow_missing`` turns 403/404 into ``None`` instead of raising —
        used for endpoints a CE instance may legitimately not expose
        (approvals) or that a token may not be scoped for.
        """
        url = f"{self._api_base}{path}"
        try:
            response = self._http.get(url, headers=self._headers, params=params)
        except AuthHttpError as exc:
            raise GitLabApiError(f"GitLab API request failed: {exc}") from exc
        status = response.status_code
        if status == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise GitLabApiError("GitLab API returned a non-JSON response") from exc
        if status in (403, 404) and allow_missing:
            return None
        if status == 401:
            raise PermissionError(
                "GitLab rejected the token (401). It may be expired or revoked."
            )
        if status == 403:
            raise PermissionError(
                "GitLab token lacks the required scope (403); read_api is needed."
            )
        if status == 404:
            raise GitLabApiError(f"GitLab resource not found: {path}")
        if status == 429:
            raise GitLabApiError("GitLab rate limit exceeded (429)")
        raise GitLabApiError(f"GitLab API returned HTTP {status} for {path}")

    def _get_list(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allow_missing: bool = False,
    ) -> list[dict[str, Any]]:
        data = self._get(path, params=params, allow_missing=allow_missing)
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _paginate(
        self,
        path: str,
        *,
        params: dict[str, Any],
        cap: int,
    ) -> Iterator[dict[str, Any]]:
        """Walk offset-paginated results up to ``cap`` items.

        ``per_page`` is fixed for the whole walk: GitLab's ``page`` is an
        offset *in units of per_page*, so shrinking it on a later page
        would re-serve rows already yielded and skip the tail. Stops on
        the first short page, so a single-page result costs one request.
        ``_MAX_PAGES`` bounds the walk even if the server keeps returning
        full pages.
        """
        per_page = min(_MAX_PER_PAGE, max(1, cap))
        emitted = 0
        for page in range(1, _MAX_PAGES + 1):
            page_params = dict(params)
            page_params["per_page"] = str(per_page)
            page_params["page"] = str(page)
            batch = self._get_list(path, params=page_params)
            if not batch:
                return
            for item in batch:
                yield item
                emitted += 1
                if emitted >= cap:
                    return
            if len(batch) < per_page:
                return

    # ------------------------------------------------------------------
    # Merge requests
    # ------------------------------------------------------------------
    def get_merge_request(
        self,
        project: str,
        iid: int,
        include_diff: bool = False,
    ) -> dict[str, Any]:
        base = f"/projects/{_encode_project(project)}/merge_requests/{int(iid)}"
        mr = self._get(base)
        if not isinstance(mr, dict):
            raise GitLabApiError(
                f"Unexpected merge request payload for {project}!{iid}"
            )
        result = _map_merge_request(mr)
        if include_diff:
            diffs = self._get_list(f"{base}/diffs", params={"per_page": "100"})
            result["files"] = [_map_diff_file(d) for d in diffs]
        return result

    def get_merge_request_commits(self, project: str, iid: int) -> list[dict[str, Any]]:
        commits = self._get_list(
            f"/projects/{_encode_project(project)}/merge_requests/{int(iid)}/commits",
            params={"per_page": "100"},
        )
        return [
            {
                "sha": c.get("id"),
                "short_sha": c.get("short_id"),
                "message": c.get("message") or c.get("title"),
                "author": c.get("author_name"),
                "author_email": c.get("author_email"),
                "committed_at": c.get("committed_date") or c.get("authored_date"),
                "url": c.get("web_url"),
            }
            for c in commits
        ]

    def get_merge_request_discussions(
        self,
        project: str,
        iid: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Inline (diff) review comments — GitLab's answer to PR review comments.

        GitLab models review feedback as *discussions*, each holding one or
        more notes. Notes of type ``DiffNote`` carry a ``position`` anchored
        to a file and line; those are the review comments. Non-diff notes in
        a discussion thread are returned too (they are replies within the
        same review thread), flagged by ``type``. Purely system notes are
        excluded here — :meth:`get_merge_request_notes` surfaces those.
        """
        raw = self._get_list(
            f"/projects/{_encode_project(project)}/merge_requests/{int(iid)}"
            "/discussions",
            params={"per_page": "100"},
        )
        out: list[dict[str, Any]] = []
        for discussion in raw:
            discussion_id = discussion.get("id")
            notes = discussion.get("notes")
            if not isinstance(notes, list):
                continue
            for note in notes:
                if not isinstance(note, dict) or note.get("system"):
                    continue
                out.append(_map_diff_note(note, discussion_id))
                if len(out) >= limit:
                    return out
        return out

    def get_merge_request_notes(
        self,
        project: str,
        iid: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Conversation-thread notes on an MR, oldest first.

        ``system`` notes are kept on purpose: on GitLab CE they are the
        durable record of review actions ("approved this merge request",
        "requested review from @x", "marked as draft"), which is exactly
        the review history the GitHub side gets from review objects.
        """
        notes = self._get_list(
            f"/projects/{_encode_project(project)}/merge_requests/{int(iid)}/notes",
            params={"per_page": "100", "sort": "asc", "order_by": "created_at"},
        )
        return [_map_note(n) for n in notes[:limit]]

    def get_merge_request_approvals(self, project: str, iid: int) -> dict[str, Any]:
        """Approval state for an MR.

        The ``/approvals`` endpoint is not exposed by every CE
        configuration, so a 403/404 is reported as
        ``{"available": False}`` rather than raised — the caller falls back
        to the "approved this merge request" system notes.
        """
        data = self._get(
            f"/projects/{_encode_project(project)}/merge_requests/{int(iid)}/approvals",
            allow_missing=True,
        )
        if not isinstance(data, dict):
            return {
                "available": False,
                "reason": "approvals endpoint not available on this instance",
                "approved_by": [],
            }
        approvers = []
        for entry in data.get("approved_by") or []:
            user = entry.get("user") if isinstance(entry, dict) else None
            if isinstance(user, dict):
                approvers.append(_username(user))
        return {
            "available": True,
            "approved": bool(data.get("approved")),
            "approvals_required": data.get("approvals_required"),
            "approvals_left": data.get("approvals_left"),
            "approved_by": [a for a in approvers if a],
        }

    def get_merge_request_state_events(
        self, project: str, iid: int
    ) -> list[dict[str, Any]]:
        """State transitions (opened / closed / merged / reopened) with actors."""
        events = self._get_list(
            f"/projects/{_encode_project(project)}/merge_requests/{int(iid)}"
            "/resource_state_events",
            params={"per_page": "100"},
            allow_missing=True,
        )
        return [
            {
                "id": e.get("id"),
                "state": e.get("state"),
                "user": _username(e.get("user")),
                "created_at": e.get("created_at"),
            }
            for e in events
        ]

    def get_merge_request_closes_issues(
        self, project: str, iid: int
    ) -> list[dict[str, Any]]:
        """Issues this MR closes on merge — the MR→issue task-tracking link."""
        issues = self._get_list(
            f"/projects/{_encode_project(project)}/merge_requests/{int(iid)}"
            "/closes_issues",
            params={"per_page": "100"},
            allow_missing=True,
        )
        return [
            {
                "iid": i.get("iid"),
                "title": i.get("title"),
                "state": i.get("state"),
                "url": i.get("web_url"),
            }
            for i in issues
        ]

    def get_merge_request_review_history(
        self, project: str, iid: int
    ) -> dict[str, Any]:
        """Approvals + threaded review discussions in one payload.

        Tries the batched GraphQL query first (one round-trip instead of
        three) and falls back to the REST composition on any failure, so a
        GitLab version whose GraphQL schema differs never breaks ingestion.
        """
        if self._graphql is not None:
            try:
                result = self._graphql.merge_request_review_history(project, int(iid))
                if result is not None:
                    result["transport"] = "graphql"
                    return result
            except Exception:
                logger.info(
                    "gitlab graphql review history failed for %s!%s; "
                    "falling back to REST",
                    project,
                    iid,
                    exc_info=True,
                )
        return {
            "transport": "rest",
            "approvals": self.get_merge_request_approvals(project, iid),
            "discussions": self.get_merge_request_discussions(project, iid),
            "notes": self.get_merge_request_notes(project, iid),
        }

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------
    def get_issue(self, project: str, iid: int) -> dict[str, Any]:
        issue = self._get(
            f"/projects/{_encode_project(project)}/issues/{int(iid)}",
        )
        if not isinstance(issue, dict):
            raise GitLabApiError(f"Unexpected issue payload for {project}#{iid}")
        return _map_issue(issue)

    def get_issue_notes(
        self,
        project: str,
        iid: int,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        notes = self._get_list(
            f"/projects/{_encode_project(project)}/issues/{int(iid)}/notes",
            params={"per_page": "100", "sort": "asc", "order_by": "created_at"},
        )
        return [_map_note(n) for n in notes[:limit]]

    def get_issue_links(self, project: str, iid: int) -> dict[str, Any]:
        """Work linked to an issue: MRs that reference it and MRs that close it."""
        encoded = _encode_project(project)
        related = self._get_list(
            f"/projects/{encoded}/issues/{int(iid)}/related_merge_requests",
            params={"per_page": "100"},
            allow_missing=True,
        )
        closed_by = self._get_list(
            f"/projects/{encoded}/issues/{int(iid)}/closed_by",
            params={"per_page": "100"},
            allow_missing=True,
        )
        return {
            "related_merge_requests": [_map_mr_ref(m) for m in related],
            "closed_by_merge_requests": [_map_mr_ref(m) for m in closed_by],
        }

    # ------------------------------------------------------------------
    # Enumeration (backfill)
    # ------------------------------------------------------------------
    def list_merge_requests(
        self,
        project: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact MR refs, newest-first, bounded by the backfill window/cap.

        Cheap enumeration for the agent's backfill todo list: identity +
        title + state only. The agent hydrates each via
        ``gitlab_get_merge_request`` when it needs the description/diff/
        discussions. The window is pushed server-side via ``updated_after``
        so the walk stops paging on its own.
        """
        params: dict[str, Any] = {
            "state": _mr_state_param(state),
            "order_by": "updated_at",
            "sort": "desc",
        }
        since = backfill_window_since()
        if since is not None:
            params["updated_after"] = since.isoformat()
        cap = clamp_backfill_limit(limit)
        path = f"/projects/{_encode_project(project)}/merge_requests"
        return [_map_mr_ref(mr) for mr in self._paginate(path, params=params, cap=cap)]

    def list_issues(
        self,
        project: str,
        *,
        state: str = "all",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compact issue refs, newest-first, bounded by the backfill window/cap.

        Unlike GitHub, GitLab keeps issues and merge requests in separate
        collections, so nothing has to be filtered back out here.
        """
        params: dict[str, Any] = {
            "state": _issue_state_param(state),
            "order_by": "updated_at",
            "sort": "desc",
        }
        since = backfill_window_since()
        if since is not None:
            params["updated_after"] = since.isoformat()
        cap = clamp_backfill_limit(limit)
        path = f"/projects/{_encode_project(project)}/issues"
        return [_map_issue_ref(i) for i in self._paginate(path, params=params, cap=cap)]

    def iter_merged_merge_requests(self, project: str) -> Iterator[dict[str, Any]]:
        """Merged MRs, newest-first, for connector artifact listing."""
        params: dict[str, Any] = {
            "state": "merged",
            "order_by": "updated_at",
            "sort": "desc",
        }
        since = backfill_window_since()
        if since is not None:
            params["updated_after"] = since.isoformat()
        path = f"/projects/{_encode_project(project)}/merge_requests"
        cap = clamp_backfill_limit(None)
        for mr in self._paginate(path, params=params, cap=cap):
            yield _map_mr_ref(mr)


# ----------------------------------------------------------------------
# Mapping helpers — GitLab payload → GitHub-aligned vocabulary
# ----------------------------------------------------------------------
def _encode_project(project: str | int) -> str:
    """URL-encode a project reference.

    GitLab accepts either a numeric id or a URL-encoded
    ``group/subgroup/project`` path; the slashes in a nested group path
    must be percent-encoded or they are read as extra URL segments.
    """
    if isinstance(project, int):
        return str(project)
    value = str(project).strip().strip("/")
    if value.isdigit():
        return value
    return quote(value, safe="")


def _mr_state_param(state: str) -> str:
    """Map the shared ``open|closed|merged|all`` vocabulary onto GitLab's."""
    normalized = (state or "all").strip().lower()
    if normalized in ("open", "opened"):
        return "opened"
    if normalized in ("closed", "merged", "locked", "all"):
        return normalized
    return "all"


def _issue_state_param(state: str) -> str:
    normalized = (state or "all").strip().lower()
    if normalized in ("open", "opened"):
        return "opened"
    if normalized == "closed":
        return "closed"
    return "all"


def _username(user: Any) -> str | None:
    if not isinstance(user, dict):
        return None
    name = user.get("username") or user.get("name")
    return str(name) if name else None


def _labels(raw: Any) -> list[dict[str, Any]]:
    """GitLab returns labels as bare strings; GitHub as objects with ``name``."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for label in raw:
        if isinstance(label, str):
            out.append({"name": label})
        elif isinstance(label, dict) and label.get("name"):
            out.append({"name": str(label["name"])})
    return out


def _milestone(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    title = raw.get("title")
    return {"title": str(title)} if title else None


def _map_merge_request(mr: dict[str, Any]) -> dict[str, Any]:
    return {
        # ``number`` mirrors the GitHub port; ``iid`` is GitLab's own name
        # for the same value and is kept so tools can echo it back.
        "number": mr.get("iid"),
        "iid": mr.get("iid"),
        "title": mr.get("title"),
        "body": mr.get("description"),
        "state": mr.get("state"),
        "merged": bool(mr.get("merged_at")),
        "created_at": mr.get("created_at"),
        "updated_at": mr.get("updated_at"),
        "merged_at": mr.get("merged_at"),
        "closed_at": mr.get("closed_at"),
        "head_branch": mr.get("source_branch"),
        "base_branch": mr.get("target_branch"),
        "url": mr.get("web_url"),
        "author": _username(mr.get("author")) or "unknown",
        "labels": _labels(mr.get("labels")),
        "milestone": _milestone(mr.get("milestone")),
        # GitLab-native review/task signals with no GitHub counterpart.
        "draft": bool(mr.get("draft") or mr.get("work_in_progress")),
        "merge_status": mr.get("detailed_merge_status") or mr.get("merge_status"),
        "sha": mr.get("sha"),
        "merge_commit_sha": mr.get("merge_commit_sha"),
        "reviewers": [
            u for u in (_username(r) for r in mr.get("reviewers") or []) if u
        ],
        "assignees": [
            u for u in (_username(a) for a in mr.get("assignees") or []) if u
        ],
        "merged_by": _username(mr.get("merged_by") or mr.get("merge_user")),
        "user_notes_count": mr.get("user_notes_count"),
    }


def _map_mr_ref(mr: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": mr.get("iid"),
        "iid": mr.get("iid"),
        "title": mr.get("title"),
        "state": mr.get("state"),
        "merged": bool(mr.get("merged_at")) or mr.get("state") == "merged",
        "draft": bool(mr.get("draft") or mr.get("work_in_progress")),
        "created_at": mr.get("created_at"),
        "updated_at": mr.get("updated_at"),
        "url": mr.get("web_url"),
        "author": _username(mr.get("author")),
    }


def _map_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue.get("iid"),
        "iid": issue.get("iid"),
        "title": issue.get("title"),
        "body": issue.get("description"),
        "state": issue.get("state"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "url": issue.get("web_url"),
        "author": _username(issue.get("author")) or "unknown",
        "labels": _labels(issue.get("labels")),
        "milestone": _milestone(issue.get("milestone")),
        # Task-tracking fields GitLab carries natively.
        "assignees": [
            u for u in (_username(a) for a in issue.get("assignees") or []) if u
        ],
        "closed_by": _username(issue.get("closed_by")),
        "due_date": issue.get("due_date"),
        "weight": issue.get("weight"),
        "issue_type": issue.get("issue_type"),
        "time_stats": issue.get("time_stats"),
        "comments": issue.get("user_notes_count"),
    }


def _map_issue_ref(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue.get("iid"),
        "iid": issue.get("iid"),
        "title": issue.get("title"),
        "state": issue.get("state"),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "url": issue.get("web_url"),
        "author": _username(issue.get("author")),
        "labels": _labels(issue.get("labels")),
        "milestone": _milestone(issue.get("milestone")),
        "comments": issue.get("user_notes_count"),
    }


def _map_note(note: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": note.get("id"),
        "body": note.get("body"),
        "user": {"login": _username(note.get("author"))}
        if note.get("author")
        else None,
        "system": bool(note.get("system")),
        "created_at": note.get("created_at"),
        "updated_at": note.get("updated_at"),
        "resolvable": note.get("resolvable"),
        "resolved": note.get("resolved"),
    }


def _map_diff_note(note: dict[str, Any], discussion_id: Any) -> dict[str, Any]:
    position = note.get("position") if isinstance(note.get("position"), dict) else {}
    return {
        "id": note.get("id"),
        "discussion_id": discussion_id,
        "type": note.get("type"),
        "body": note.get("body"),
        "user": {"login": _username(note.get("author"))}
        if note.get("author")
        else None,
        "path": position.get("new_path") or position.get("old_path"),
        "line": position.get("new_line") or position.get("old_line"),
        "resolvable": note.get("resolvable"),
        "resolved": note.get("resolved"),
        "resolved_by": _username(note.get("resolved_by")),
        "created_at": note.get("created_at"),
    }


def _map_diff_file(diff: dict[str, Any]) -> dict[str, Any]:
    """Map a GitLab diff entry onto the GitHub ``files[]`` shape."""
    if diff.get("new_file"):
        status = "added"
    elif diff.get("deleted_file"):
        status = "removed"
    elif diff.get("renamed_file"):
        status = "renamed"
    else:
        status = "modified"
    return {
        "filename": diff.get("new_path") or diff.get("old_path"),
        "previous_filename": diff.get("old_path"),
        "status": status,
        "patch": diff.get("diff"),
    }


__all__ = [
    "GitLabApiError",
    "GitLabReadPort",
    "GitLabRestSourceControl",
]

"""Deterministic GitHub merged-PR planner using generic graph mutations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from domain.context_events import EventRef
from domain.episode_formatters import build_pr_episode
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.reconciliation import (
    EpisodeDraft,
    EvidenceRef,
    ReconciliationPlan,
    ReconciliationRequest,
)


def build_github_pr_merged_plan(
    *,
    event_ref: EventRef,
    repo_name: str,
    pr_data: dict[str, Any],
    commits: list[dict[str, Any]],
    review_threads: list[dict[str, Any]],
    linked_issues: list[dict[str, Any]],
    issue_comments: list[dict[str, Any]] | None = None,
) -> ReconciliationPlan:
    """Build a generic reconciliation plan for a merged GitHub PR."""
    ep = build_pr_episode(
        pr_data=pr_data,
        commits=commits,
        review_threads=review_threads,
        linked_issues=linked_issues,
        issue_comments=issue_comments,
    )
    draft = EpisodeDraft(
        name=ep["name"],
        episode_body=ep["episode_body"],
        source_description=ep["source_description"],
        reference_time=ep["reference_time"],
    )

    pr_number = int(pr_data.get("number") or 0)
    pr_key = f"github:pr:{repo_name}:{pr_number}"
    repo_key = f"github:repo:{repo_name}"
    source_key = f"source-ref:github:pull_request:{repo_name}:{pr_number}"
    observed_at = _iso(pr_data.get("merged_at")) or _iso(draft.reference_time)
    title = str(pr_data.get("title") or f"PR #{pr_number}")
    author = str(pr_data.get("author") or "unknown")
    url = str(pr_data.get("url") or "")

    entities: list[EntityUpsert] = [
        EntityUpsert(
            entity_key=repo_key,
            labels=("Entity", "Repository"),
            properties={
                "name": repo_name,
                "provider": "github",
                "provider_host": "github.com",
                "repo_name": repo_name,
                "source_ref": source_key,
            },
        ),
        EntityUpsert(
            entity_key=source_key,
            labels=("Entity", "SourceReference"),
            properties={
                "name": f"GitHub PR {repo_name}#{pr_number}",
                "source_system": "github",
                "source_kind": "pull_request",
                "source_type": "pull_request",
                "ref_type": "pull_request",
                "external_id": f"{repo_name}#{pr_number}",
                "retrieval_uri": url,
                "uri": url,
                "source_ref": source_key,
                "observed_at": observed_at,
                "verification_state": "unverified",
                "sync_status": "needs_resync",
            },
        ),
        EntityUpsert(
            entity_key=pr_key,
            labels=("Entity", "Change", "PullRequest"),
            properties={
                "name": title,
                "title": title,
                "summary": str(pr_data.get("body") or ""),
                "description": str(pr_data.get("body") or ""),
                "change_type": "pull_request",
                "pr_number": pr_number,
                "number": pr_number,
                "repo_name": repo_name,
                "repository_key": repo_key,
                "author": author,
                "merged_at": observed_at,
                "head_branch": str(pr_data.get("head_branch") or ""),
                "base_branch": str(pr_data.get("base_branch") or ""),
                "url": url,
                "source_ref": source_key,
                "observed_at": observed_at,
            },
        ),
    ]
    edges: list[EdgeUpsert] = [
        EdgeUpsert("EVIDENCED_BY", pr_key, source_key, {"source_ref": source_key}),
    ]

    author_key = _person_key(author)
    if author_key:
        entities.append(_person_entity(author, source_key, observed_at))
        edges.append(
            EdgeUpsert(
                "REVIEWS",
                author_key,
                pr_key,
                {
                    "role": "author",
                    "source_ref": source_key,
                    "valid_from": observed_at,
                },
            )
        )

    for commit in commits or []:
        sha = str(commit.get("sha") or "").strip()
        if not sha:
            continue
        message = str(commit.get("message") or "").strip()
        first_line = message.splitlines()[0] if message else f"Commit {sha[:12]}"
        commit_key = f"github:commit:{repo_name}:{sha}"
        commit_author = str(commit.get("author") or "unknown")
        entities.append(
            EntityUpsert(
                entity_key=commit_key,
                labels=("Entity", "Commit"),
                properties={
                    "name": first_line[:300],
                    "title": first_line[:300],
                    "summary": message[:16000],
                    "sha": sha,
                    "author": commit_author,
                    "repo_name": repo_name,
                    "source_ref": source_key,
                    "observed_at": observed_at,
                },
            )
        )
        edges.append(
            EdgeUpsert(
                "HAS_COMMIT",
                pr_key,
                commit_key,
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )
        edges.append(
            EdgeUpsert(
                "PART_OF",
                commit_key,
                pr_key,
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )
        commit_author_key = _person_key(commit_author)
        if commit_author_key:
            entities.append(_person_entity(commit_author, source_key, observed_at))
            edges.append(
                EdgeUpsert(
                    "REVIEWS",
                    commit_author_key,
                    pr_key,
                    {
                        "role": "committer",
                        "source_ref": source_key,
                        "valid_from": observed_at,
                    },
                )
            )

    for issue in linked_issues or []:
        issue_key, issue_props = _issue_identity(repo_name, issue, source_key, observed_at)
        if not issue_key:
            continue
        entities.append(
            EntityUpsert(
                entity_key=issue_key,
                labels=("Entity", "Issue"),
                properties=issue_props,
            )
        )
        edges.append(
            EdgeUpsert(
                "ADDRESSES",
                pr_key,
                issue_key,
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )

    for file_info in pr_data.get("files") or []:
        path = str(file_info.get("filename") or file_info.get("path") or "").strip()
        if not path:
            continue
        asset_key = f"code:file:{repo_name}:{path}"
        entities.append(
            EntityUpsert(
                entity_key=asset_key,
                labels=("Entity", "CodeAsset"),
                properties={
                    "name": path,
                    "asset_type": "file",
                    "file_path": path,
                    "repo_name": repo_name,
                    "repository_key": repo_key,
                    "source_ref": source_key,
                    "observed_at": observed_at,
                },
            )
        )
        edges.append(
            EdgeUpsert(
                "MODIFIED",
                pr_key,
                asset_key,
                {
                    "source_ref": source_key,
                    "valid_from": observed_at,
                    "status": str(file_info.get("status") or ""),
                    "additions": int(file_info.get("additions") or 0),
                    "deletions": int(file_info.get("deletions") or 0),
                    "patch_excerpt": str(file_info.get("patch") or "")[:4000],
                },
            )
        )

    for thread in review_threads or []:
        decision = _decision_from_review_thread(
            repo_name, pr_number, thread, source_key, observed_at
        )
        if decision is None:
            continue
        decision_entity, decision_edges = decision
        entities.append(decision_entity)
        edges.extend(decision_edges)

    conversation = _decision_from_issue_comments(
        repo_name, pr_number, issue_comments or [], source_key, observed_at
    )
    if conversation is not None:
        decision_entity, decision_edges = conversation
        entities.append(decision_entity)
        edges.extend(decision_edges)

    return ReconciliationPlan(
        event_ref=event_ref,
        summary=f"merged GitHub PR #{pr_number} ({repo_name})",
        episodes=[draft],
        entity_upserts=_dedupe_entities(entities),
        edge_upserts=_dedupe_edges(edges),
        evidence=[
            EvidenceRef(
                kind="source_ref",
                ref=source_key,
                metadata={"provider": "github", "repo_name": repo_name, "pr_number": pr_number},
            )
        ],
        confidence=0.9,
    )


class GitHubPrMergedPlannerAgent:
    """Deterministic reconciliation agent for merged GitHub PR payloads."""

    def __init__(self, repo_name: str) -> None:
        self._repo_name = repo_name

    def run_reconciliation(self, request: ReconciliationRequest) -> ReconciliationPlan:
        p = request.event.payload
        ref = EventRef(
            event_id=request.event.event_id,
            source_system=request.event.source_system,
            pot_id=request.pot_id,
        )
        return build_github_pr_merged_plan(
            event_ref=ref,
            repo_name=self._repo_name,
            pr_data=p["pr_data"],
            commits=list(p.get("commits") or []),
            review_threads=list(p.get("review_threads") or []),
            linked_issues=list(p.get("linked_issues") or []),
            issue_comments=list(p.get("issue_comments") or []) or None,
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {"agent": "github_pr_merged_planner", "version": "2"}


def _decision_from_review_thread(
    repo_name: str,
    pr_number: int,
    thread: dict[str, Any],
    source_key: str,
    observed_at: str,
) -> tuple[EntityUpsert, list[EdgeUpsert]] | None:
    tid = thread.get("thread_id")
    if tid is None:
        return None
    comments = list(thread.get("comments") or [])
    lines = _comment_lines(comments)
    full_text = "\n\n".join(lines).strip() or "(empty thread)"
    if len(full_text) > 16000:
        full_text = full_text[:16000] + "\n..."
    first_excerpt = _first_comment_excerpt(comments) or full_text[:240]
    decision_key = f"github:decision:{repo_name}:{pr_number}:{tid!s}"
    pr_key = f"github:pr:{repo_name}:{pr_number}"
    path = thread.get("path")
    line = thread.get("line")
    entity = EntityUpsert(
        entity_key=decision_key,
        labels=("Entity", "Decision"),
        properties={
            "name": f"PR #{pr_number} review thread {tid!s}",
            "title": f"PR #{pr_number} review thread {tid!s}",
            "summary": full_text,
            "decision_made": first_excerpt,
            "status": "accepted",
            "thread_id": str(tid),
            "review_path": path,
            "review_line": line,
            "source_ref": source_key,
            "observed_at": observed_at,
        },
    )
    edges = [
        EdgeUpsert(
            "HAS_REVIEW_DECISION",
            pr_key,
            decision_key,
            {"thread_id": str(tid), "source_ref": source_key, "valid_from": observed_at},
        ),
        EdgeUpsert(
            "MADE_IN",
            decision_key,
            pr_key,
            {"source_ref": source_key, "valid_from": observed_at},
        ),
    ]
    if path:
        asset_key = f"code:file:{repo_name}:{path}"
        edges.append(
            EdgeUpsert(
                "AFFECTS",
                decision_key,
                asset_key,
                {"source_ref": source_key, "valid_from": observed_at},
            )
        )
    return entity, edges


def _decision_from_issue_comments(
    repo_name: str,
    pr_number: int,
    issue_comments: list[dict[str, Any]],
    source_key: str,
    observed_at: str,
) -> tuple[EntityUpsert, list[EdgeUpsert]] | None:
    lines = _comment_lines(issue_comments)
    if not lines:
        return None
    full_text = "\n\n".join(lines).strip()
    if len(full_text) > 16000:
        full_text = full_text[:16000] + "\n..."
    decision_key = f"github:decision:{repo_name}:{pr_number}:pr_conversation"
    pr_key = f"github:pr:{repo_name}:{pr_number}"
    entity = EntityUpsert(
        entity_key=decision_key,
        labels=("Entity", "Decision"),
        properties={
            "name": f"PR #{pr_number} conversation",
            "title": f"PR #{pr_number} conversation",
            "summary": full_text,
            "decision_made": lines[0][:500],
            "status": "accepted",
            "thread_id": "pr_conversation",
            "review_path": None,
            "review_line": None,
            "source_ref": source_key,
            "observed_at": observed_at,
        },
    )
    return entity, [
        EdgeUpsert(
            "HAS_REVIEW_DECISION",
            pr_key,
            decision_key,
            {
                "thread_id": "pr_conversation",
                "source_ref": source_key,
                "valid_from": observed_at,
            },
        ),
        EdgeUpsert(
            "MADE_IN",
            decision_key,
            pr_key,
            {"source_ref": source_key, "valid_from": observed_at},
        ),
    ]


def _issue_identity(
    repo_name: str,
    issue: dict[str, Any],
    source_key: str,
    observed_at: str,
) -> tuple[str | None, dict[str, Any]]:
    raw_number = issue.get("number") or issue.get("issue_number")
    raw_key = issue.get("key") or issue.get("id") or raw_number
    if raw_key is None:
        return None, {}
    issue_key = f"github:issue:{repo_name}:{raw_key}"
    title = str(issue.get("title") or f"Issue {raw_key}")
    status = str(issue.get("status") or issue.get("state") or "unknown")
    if status not in {"open", "closed", "triaged", "blocked", "unknown"}:
        status = "unknown"
    return issue_key, {
        "name": title,
        "title": title,
        "summary": str(issue.get("body") or issue.get("summary") or ""),
        "status": status,
        "issue_number": raw_number,
        "repo_name": repo_name,
        "source_ref": source_key,
        "observed_at": observed_at,
    }


def _person_entity(login: str, source_key: str, observed_at: str) -> EntityUpsert:
    name = login.strip() or "unknown"
    return EntityUpsert(
        entity_key=_person_key(name) or "github:user:unknown",
        labels=("Entity", "Person"),
        properties={
            "name": name,
            "github_login": name,
            "source_ref": source_key,
            "observed_at": observed_at,
        },
    )


def _person_key(login: str) -> str | None:
    value = (login or "").strip().lower()
    if not value or value == "unknown":
        return None
    return f"github:user:{value}"


def _comment_lines(comments: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for c in comments:
        raw_user = c.get("user")
        if isinstance(raw_user, dict):
            who = raw_user.get("login") or c.get("author") or "unknown"
        else:
            who = c.get("author") or raw_user or "unknown"
        body = str(c.get("body") or "").strip()
        if body:
            lines.append(f"{who}: {body}")
    return lines


def _first_comment_excerpt(comments: list[dict[str, Any]]) -> str:
    if not comments:
        return ""
    return str(comments[0].get("body") or "").strip()[:500]


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return datetime.now(timezone.utc).isoformat()
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _dedupe_entities(items: list[EntityUpsert]) -> list[EntityUpsert]:
    out: dict[str, EntityUpsert] = {}
    for item in items:
        if item.entity_key in out:
            prior = out[item.entity_key]
            labels = tuple(dict.fromkeys([*prior.labels, *item.labels]))
            props = {**prior.properties, **item.properties}
            out[item.entity_key] = EntityUpsert(item.entity_key, labels, props)
        else:
            out[item.entity_key] = item
    return list(out.values())


def _dedupe_edges(items: list[EdgeUpsert]) -> list[EdgeUpsert]:
    out: dict[tuple[str, str, str], EdgeUpsert] = {}
    for item in items:
        out[(item.edge_type, item.from_entity_key, item.to_entity_key)] = item
    return list(out.values())

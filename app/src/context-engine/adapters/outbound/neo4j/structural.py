"""Neo4j structural graph: bridges, entity stamping, agent queries."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from neo4j import Driver, GraphDatabase

from domain.deterministic_extractors import parse_diff_hunks
from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
from domain.ingestion import BridgeResult
from domain.ontology import ENTITY_TYPES, is_canonical_entity_label
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)

_PROJECT_MAP_LABELS_BY_INCLUDE: dict[str, tuple[str, ...]] = {
    "purpose": ("Pot", "System"),
    "repo_map": ("Repository",),
    "service_map": (
        "Service",
        "Component",
        "Interface",
        "DataStore",
        "Integration",
        "Dependency",
    ),
    "feature_map": (
        "Capability",
        "Feature",
        "Functionality",
        "Requirement",
        "RoadmapItem",
    ),
    "docs": ("Document",),
    "deployments": (
        "Deployment",
        "DeploymentTarget",
        "DeploymentStrategy",
        "Environment",
    ),
    "runbooks": ("Runbook",),
    "local_workflows": ("LocalWorkflow",),
    "scripts": ("Script",),
    "config": ("ConfigVariable",),
    "preferences": ("Preference",),
    "agent_instructions": ("AgentInstruction",),
    "operations": (
        "Deployment",
        "DeploymentTarget",
        "DeploymentStrategy",
        "Environment",
        "Runbook",
        "Script",
        "ConfigVariable",
        "LocalWorkflow",
    ),
    "owners": ("Person", "Team", "Role"),
}

_DEBUGGING_MEMORY_LABELS_BY_INCLUDE: dict[str, tuple[str, ...]] = {
    "prior_fixes": ("Fix", "BugPattern", "Investigation"),
    "diagnostic_signals": ("DiagnosticSignal",),
    "incidents": ("Incident",),
    "alerts": ("Alert",),
}


def _merge_decision_result_rows(
    code_rows: list[dict[str, Any]],
    pr_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in code_rows + pr_rows:
        dedupe = row.pop("_dedupe", None)
        key = (
            str(dedupe)
            if dedupe is not None
            else f"{row.get('decision_made')}|{row.get('pr_number')}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _driver_from_settings(settings: ContextEngineSettingsPort) -> Optional[Driver]:
    uri = settings.neo4j_uri()
    user = settings.neo4j_user()
    password = settings.neo4j_password()
    if not uri or user is None or password is None:
        return None
    return GraphDatabase.driver(uri, auth=(user, password))


class Neo4jStructuralAdapter(StructuralGraphPort):
    def __init__(self, settings: ContextEngineSettingsPort) -> None:
        self._settings = settings

    def _open(self) -> Optional[Driver]:
        if not self._settings.is_enabled():
            return None
        return _driver_from_settings(self._settings)

    @staticmethod
    def _ensure_indexes(session) -> None:
        session.run(
            "CREATE INDEX node_file_repo_idx IF NOT EXISTS "
            "FOR (n:NODE) ON (n.file_path, n.repoId)"
        )
        session.run(
            "CREATE INDEX entity_key_group_idx IF NOT EXISTS "
            "FOR (e:Entity) ON (e.entity_key, e.group_id)"
        )
        session.run(
            "CREATE INDEX entity_valid_to_idx IF NOT EXISTS "
            "FOR (e:Entity) ON (e.valid_to)"
        )

    def write_bridges(
        self,
        pot_id: str,
        pr_entity_key: str,
        pr_number: int,
        repo_name: str,
        files_with_patches: list[dict[str, Any]],
        review_threads: list[dict[str, Any]],
        merged_at: str | None,
        is_live: bool,
    ) -> BridgeResult:
        result = BridgeResult()
        drv = self._open()
        if drv is None:
            return result
        try:
            with drv.session() as session:
                self._ensure_indexes(session)

                for f in files_with_patches or []:
                    file_path = f.get("filename")
                    if not file_path:
                        continue
                    patch = f.get("patch") or ""
                    patch_excerpt = patch[:3000]
                    status = f.get("status")
                    additions = f.get("additions")
                    deletions = f.get("deletions")

                    res = session.run(
                        """
                        MATCH (file:FILE {repoId: $pot_id})
                        WHERE file.file_path = $file_path
                           OR file.file_path ENDS WITH ('/' + $file_path)
                        MATCH (pr:Entity {group_id: $pot_id, entity_key: $pr_key})
                        MERGE (file)-[r:TOUCHED_BY {pr_number: $pr_number}]->(pr)
                        SET r.updated_at = timestamp(),
                            r.status = $status,
                            r.additions = $additions,
                            r.deletions = $deletions,
                            r.patch_excerpt = $patch_excerpt,
                            r.valid_from = $merged_at
                        RETURN count(r) AS cnt
                        """,
                        pot_id=pot_id,
                        file_path=file_path,
                        pr_key=pr_entity_key,
                        pr_number=pr_number,
                        status=status,
                        additions=additions,
                        deletions=deletions,
                        patch_excerpt=patch_excerpt,
                        merged_at=merged_at,
                    )
                    result.touched_by += _count(res)

                    for start_line, end_line in parse_diff_hunks(f.get("patch")):
                        res = session.run(
                            """
                            MATCH (n:NODE {repoId: $pot_id})
                            WHERE (n.file_path = $file_path
                                   OR n.file_path ENDS WITH ('/' + $file_path))
                              AND any(lbl IN labels(n) WHERE lbl IN ['FUNCTION', 'CLASS'])
                              AND n.start_line <= $end_line
                              AND n.end_line >= $start_line
                            MATCH (pr:Entity {group_id: $pot_id, entity_key: $pr_key})
                            MERGE (n)-[r:MODIFIED_IN {pr_number: $pr_number}]->(pr)
                            SET r.merged_at = $merged_at,
                                r.valid_from = $merged_at,
                                r.updated_at = timestamp(),
                                r.is_approximate = $is_approximate
                            RETURN count(r) AS cnt
                            """,
                            pot_id=pot_id,
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            pr_key=pr_entity_key,
                            pr_number=pr_number,
                            merged_at=merged_at,
                            is_approximate=not is_live,
                        )
                        result.modified_in += _count(res)

                for thread in review_threads or []:
                    thread_id = thread.get("thread_id")
                    path = thread.get("path")
                    line = thread.get("line")
                    if not path or line is None or thread_id is None:
                        continue

                    decision_key = (
                        f"github:decision:{repo_name}:{pr_number}:{thread_id!s}"
                    )

                    res = session.run(
                        """
                        MATCH (n:NODE {repoId: $pot_id})
                        WHERE (n.file_path = $file_path
                               OR n.file_path ENDS WITH ('/' + $file_path))
                          AND any(lbl IN labels(n) WHERE lbl IN ['FUNCTION', 'CLASS'])
                          AND n.start_line <= $line
                          AND n.end_line >= $line
                        WITH n
                        MATCH (d:Entity {group_id: $pot_id, entity_key: $decision_key})
                        MERGE (n)-[r:HAS_DECISION]->(d)
                        SET r.updated_at = timestamp()
                        RETURN count(r) AS cnt
                        """,
                        pot_id=pot_id,
                        file_path=path,
                        line=line,
                        decision_key=decision_key,
                    )
                    result.has_decision += _count(res)

        except Exception:
            logger.exception("write_bridges failed project=%s pr=%s", pot_id, pr_number)
            raise
        finally:
            drv.close()

        logger.info(
            "Bridge write complete project=%s pr=%s edges=%s",
            pot_id,
            pr_number,
            result.as_dict(),
        )
        return result

    def stamp_pr_entities(
        self,
        pot_id: str,
        episode_uuid: str,
        repo_name: str,
        pr_number: int,
        commits: list[dict[str, Any]],
        review_threads: list[dict[str, Any]],
        pr_data: dict[str, Any] | None = None,
        author: str | None = None,
        pr_title: str | None = None,
        issue_comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        pr_key = f"github:pr:{repo_name}:{pr_number}"
        counts: dict[str, int] = {
            "stamped_pr": 0,
            "stamped_commits": 0,
            "stamped_developers": 0,
            "stamped_decisions": 0,
            "review_threads_linked": 0,
            "pr_conversation_linked": 0,
        }

        drv = self._open()
        if drv is None:
            return counts

        try:
            with drv.session() as session:
                session.run(
                    "CREATE INDEX entity_key_group_idx IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.entity_key, e.group_id)"
                )

                if episode_uuid:
                    result = session.run(
                        """
                        MATCH (ep:Episodic {uuid: $ep_uuid})-[:MENTIONS]->(e:Entity {group_id: $gid})
                        WHERE 'PullRequest' IN labels(e)
                        SET e.entity_key = $key
                        RETURN count(e) AS cnt
                        """,
                        ep_uuid=episode_uuid,
                        gid=pot_id,
                        key=pr_key,
                    )
                    counts["stamped_pr"] = _int(result)

                if counts["stamped_pr"] == 0:
                    result = session.run(
                        """
                        MATCH (e:Entity {group_id: $gid})
                        WHERE 'PullRequest' IN labels(e)
                          AND (e.entity_key IS NULL OR e.entity_key = '')
                          AND (toLower(coalesce(e.name,'')) CONTAINS $needle
                               OR toLower(coalesce(e.summary,'')) CONTAINS $needle)
                        SET e.entity_key = $key
                        RETURN count(e) AS cnt
                        """,
                        gid=pot_id,
                        needle=str(pr_number),
                        key=pr_key,
                    )
                    counts["stamped_pr"] = _int(result)

                if counts["stamped_pr"] == 0:
                    # Graphiti extraction can omit PullRequest entities and/or the ep->MENTIONS link.
                    # If we don't ensure a canonical PR entity keyed by github:pr:repo:N, bridges will
                    # write 0 edges even when FILE nodes exist.
                    session.run(
                        """
                        MERGE (pr:Entity:PullRequest {group_id: $gid, entity_key: $key})
                        ON CREATE SET
                          pr.uuid = randomUUID(),
                          pr.pr_number = $pr_num,
                          pr.name = CASE
                            WHEN $title IS NULL OR trim($title) = '' THEN 'PR #' + toString($pr_num)
                            ELSE $title
                          END,
                          pr.summary = '',
                          pr.created_at = timestamp()
                        SET pr.summary = coalesce(pr.summary, '')
                        """,
                        gid=pot_id,
                        key=pr_key,
                        pr_num=pr_number,
                        title=pr_title or "",
                    )
                    if episode_uuid:
                        session.run(
                            """
                            MATCH (ep:Episodic {uuid: $ep_uuid})
                            MATCH (pr:Entity:PullRequest {group_id: $gid, entity_key: $key})
                            MERGE (ep)-[:MENTIONS]->(pr)
                            """,
                            ep_uuid=episode_uuid,
                            gid=pot_id,
                            key=pr_key,
                        )
                    counts["stamped_pr"] = 1

                canonical_author = author or (pr_data or {}).get("author") or "unknown"
                canonical_title = pr_title or (pr_data or {}).get("title") or ""
                canonical_body = (pr_data or {}).get("body") or ""
                canonical_merged_at = (pr_data or {}).get("merged_at")
                canonical_head = (pr_data or {}).get("head_branch") or ""
                canonical_base = (pr_data or {}).get("base_branch") or ""
                canonical_url = (pr_data or {}).get("url") or ""
                session.run(
                    """
                    MATCH (pr:Entity:PullRequest {group_id: $gid, entity_key: $key})
                    SET pr.pr_number = $pr_num,
                        pr.name = CASE
                          WHEN $title IS NULL OR trim($title) = '' THEN 'PR #' + toString($pr_num)
                          ELSE $title
                        END,
                        pr.title = CASE
                          WHEN $title IS NULL OR trim($title) = '' THEN pr.title
                          ELSE $title
                        END,
                        pr.author = $author,
                        pr.description = $body,
                        pr.merged_at = $merged_at,
                        pr.head_branch = $head_branch,
                        pr.base_branch = $base_branch,
                        pr.url = $url,
                        pr.summary = coalesce(pr.summary, '')
                    """,
                    gid=pot_id,
                    key=pr_key,
                    pr_num=pr_number,
                    title=canonical_title,
                    author=canonical_author,
                    body=canonical_body,
                    merged_at=canonical_merged_at,
                    head_branch=canonical_head,
                    base_branch=canonical_base,
                    url=canonical_url,
                )

                # Graphiti's EntityNode model expects summary to be a string.
                # Normalize historical/null values to avoid pydantic validation errors on search.
                session.run(
                    """
                    MATCH (e:Entity {group_id: $gid})
                    WHERE 'PullRequest' IN labels(e)
                      AND (e.summary IS NULL)
                    SET e.summary = ''
                    """,
                    gid=pot_id,
                )

                for commit in commits or []:
                    sha = commit.get("sha")
                    if not sha:
                        continue
                    commit_key = f"github:commit:{repo_name}:{sha}"
                    result = session.run(
                        """
                        MATCH (e:Entity {group_id: $gid})
                        WHERE 'Commit' IN labels(e)
                          AND (toLower(coalesce(e.name,'')) CONTAINS toLower($sha)
                               OR toLower(coalesce(e.summary,'')) CONTAINS toLower($sha))
                        SET e.entity_key = $key
                        RETURN count(e) AS cnt
                        """,
                        gid=pot_id,
                        sha=sha[:12],
                        key=commit_key,
                    )
                    counts["stamped_commits"] += _int(result)
                    message = (commit.get("message") or "").strip()
                    first_line = (
                        message.splitlines()[0] if message else f"Commit {sha[:12]}"
                    )
                    commit_author = commit.get("author") or "unknown"
                    session.run(
                        """
                        MERGE (c:Entity:Commit {group_id: $gid, entity_key: $key})
                        ON CREATE SET
                          c.uuid = randomUUID(),
                          c.created_at = timestamp()
                        SET c.sha = $sha,
                            c.name = $name,
                            c.summary = $summary,
                            c.author = $author
                        WITH c
                        MATCH (pr:Entity:PullRequest {group_id: $gid, entity_key: $pkey})
                        MERGE (pr)-[:HAS_COMMIT]->(c)
                        """,
                        gid=pot_id,
                        key=commit_key,
                        sha=sha,
                        name=first_line[:300],
                        summary=message[:16000],
                        author=commit_author,
                        pkey=pr_key,
                    )

                dev_logins: set[str] = set()
                if author:
                    dev_logins.add(author.lower())
                for commit in commits or []:
                    commit_author = commit.get("author")
                    if commit_author:
                        dev_logins.add(commit_author.lower())

                for login in dev_logins:
                    dev_key = f"github:user:{login}"
                    result = session.run(
                        """
                        MATCH (e:Entity {group_id: $gid})
                        WHERE 'Developer' IN labels(e)
                          AND toLower(coalesce(e.name, '')) = toLower($login)
                        SET e.entity_key = $key
                        RETURN count(e) AS cnt
                        """,
                        gid=pot_id,
                        login=login,
                        key=dev_key,
                    )
                    counts["stamped_developers"] += _int(result)

                for thread in review_threads or []:
                    thread_id = thread.get("thread_id")
                    if thread_id is None:
                        continue
                    decision_key = (
                        f"github:decision:{repo_name}:{pr_number}:{thread_id!s}"
                    )

                    first_body = ""
                    comments = thread.get("comments") or []
                    if comments:
                        first_body = (
                            (comments[0].get("body") or "").strip()[:120].lower()
                        )

                    if not first_body:
                        continue

                    result = session.run(
                        """
                        MATCH (e:Entity {group_id: $gid})
                        WHERE 'Decision' IN labels(e)
                          AND (e.entity_key IS NULL OR e.entity_key = '')
                          AND toLower(
                            coalesce(e.name,'') + ' ' +
                            coalesce(e.summary,'') + ' ' +
                            coalesce(e.decision_made,'')
                          ) CONTAINS $needle
                        WITH e LIMIT 1
                        SET e.entity_key = $key
                        RETURN count(e) AS cnt
                        """,
                        gid=pot_id,
                        needle=first_body,
                        key=decision_key,
                    )
                    counts["stamped_decisions"] += _int(result)

                # Deterministic Decision nodes + PR linkage for review discussions (not only Graphiti fuzzy match).
                for thread in review_threads or []:
                    tid = thread.get("thread_id")
                    if tid is None:
                        continue
                    tid_s = str(tid)
                    decision_key = f"github:decision:{repo_name}:{pr_number}:{tid_s}"
                    comments = thread.get("comments") or []
                    lines: list[str] = []
                    for c in comments:
                        who = c.get("author") or "unknown"
                        body = (c.get("body") or "").strip()
                        if body:
                            lines.append(f"{who}: {body}")
                    full_text = "\n\n".join(lines).strip()
                    if not full_text:
                        full_text = "(empty thread)"
                    if len(full_text) > 16000:
                        full_text = full_text[:16000] + "\n…"
                    first_excerpt = ""
                    if comments:
                        first_body = (comments[0].get("body") or "").strip()
                        first_excerpt = first_body[:500] if first_body else ""
                    if not first_excerpt:
                        first_excerpt = (
                            full_text[:240]
                            if full_text != "(empty thread)"
                            else "(no text in thread)"
                        )
                    path = thread.get("path")
                    line = thread.get("line")
                    session.run(
                        """
                        MERGE (d:Entity:Decision {group_id: $gid, entity_key: $dkey})
                        ON CREATE SET
                          d.uuid = randomUUID(),
                          d.alternatives_rejected = '',
                          d.rationale = '',
                          d.created_at = timestamp()
                        SET d.name = $dname,
                            d.summary = $full,
                            d.decision_made = $excerpt,
                            d.thread_id = $tid,
                            d.review_path = $path,
                            d.review_line = $line
                        WITH d
                        MATCH (pr:Entity:PullRequest {group_id: $gid, entity_key: $pkey})
                        MERGE (pr)-[r:HAS_REVIEW_DECISION]->(d)
                        SET r.thread_id = $tid
                        """,
                        gid=pot_id,
                        dkey=decision_key,
                        dname=(f"PR #{pr_number} review thread {tid_s}")[:200],
                        full=full_text,
                        excerpt=first_excerpt,
                        tid=tid_s,
                        path=path,
                        line=line,
                        pkey=pr_key,
                    )
                    counts["review_threads_linked"] += 1

                # Main PR conversation (issue comments) — not the same as line-level review threads.
                conv_lines: list[str] = []
                for c in issue_comments or []:
                    user = c.get("user") or {}
                    login = user.get("login") if isinstance(user, dict) else None
                    who = login or "unknown"
                    body = (c.get("body") or "").strip()
                    if body:
                        conv_lines.append(f"{who}: {body}")
                conv_full = "\n\n".join(conv_lines).strip()
                if conv_full:
                    if len(conv_full) > 16000:
                        conv_full = conv_full[:16000] + "\n…"
                    conv_key = (
                        f"github:decision:{repo_name}:{pr_number}:pr_conversation"
                    )
                    conv_excerpt = conv_lines[0][:500] if conv_lines else ""
                    session.run(
                        """
                        MERGE (d:Entity:Decision {group_id: $gid, entity_key: $dkey})
                        ON CREATE SET
                          d.uuid = randomUUID(),
                          d.alternatives_rejected = '',
                          d.rationale = '',
                          d.created_at = timestamp()
                        SET d.name = $dname,
                            d.summary = $full,
                            d.decision_made = $excerpt,
                            d.thread_id = 'pr_conversation',
                            d.review_path = NULL,
                            d.review_line = NULL
                        WITH d
                        MATCH (pr:Entity:PullRequest {group_id: $gid, entity_key: $pkey})
                        MERGE (pr)-[r:HAS_REVIEW_DECISION]->(d)
                        SET r.thread_id = 'pr_conversation'
                        """,
                        gid=pot_id,
                        dkey=conv_key,
                        dname=(f"PR #{pr_number} conversation")[:200],
                        full=conv_full,
                        excerpt=conv_excerpt or "PR timeline comments",
                        pkey=pr_key,
                    )
                    counts["pr_conversation_linked"] = 1

        except Exception:
            logger.exception(
                "stamp_pr_entities failed project=%s episode=%s", pot_id, episode_uuid
            )
            raise
        finally:
            drv.close()

        logger.info(
            "Entity key stamping complete project=%s pr=%s counts=%s",
            pot_id,
            pr_number,
            counts,
        )
        return counts

    def get_change_history(
        self,
        pot_id: str,
        function_name: str | None,
        file_path: str | None,
        limit: int,
        repo_name: str | None = None,
        pr_number: int | None = None,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        _ = repo_name  # reserved for future repo-scoped history
        if pr_number is not None:
            query = """
            MATCH (pr:Entity {group_id: $pot_id})
            WHERE 'PullRequest' IN labels(pr)
              AND (
                coalesce(pr.pr_number, toIntegerOrNull(pr.number)) = $pr_number
                OR toIntegerOrNull(reverse(split(coalesce(pr.entity_key, ''), ':'))[0]) = $pr_number
              )
              AND CASE
                WHEN $as_of IS NULL THEN pr.valid_to IS NULL
                ELSE (pr.valid_to IS NULL OR pr.valid_to > $as_of)
              END
            OPTIONAL MATCH (pr)-[:HAS_REVIEW_DECISION]->(prdec:Entity)
            WHERE prdec IS NULL OR 'Decision' IN labels(prdec)
            WITH pr, collect(DISTINCT coalesce(prdec.decision_made, prdec.name, prdec.summary)) AS prdecs
            RETURN
              coalesce(pr.pr_number, toIntegerOrNull(reverse(split(coalesce(pr.entity_key, ''), ':'))[0])) AS pr_number,
              coalesce(pr.title, pr.name) AS title,
              coalesce(pr.why_summary, pr.summary, pr.description, '') AS why_summary,
              coalesce(pr.change_type, '') AS change_type,
              coalesce(pr.feature_area, '') AS feature_area,
              [] AS fixed_issues,
              [x IN prdecs WHERE x IS NOT NULL AND toString(x) <> ''] AS decisions
            LIMIT 1
            """
            return self._run_read(
                query,
                pot_id,
                file_path,
                function_name,
                1,
                pr_number=pr_number,
                as_of=as_of,
            )

        query = """
        MATCH (n:NODE {repoId: $pot_id})
        WHERE ($file_path IS NULL OR n.file_path = $file_path)
          AND ($function_name IS NULL OR toLower(coalesce(n.name, '')) CONTAINS toLower($function_name))
        OPTIONAL MATCH (n)-[mod:MODIFIED_IN]->(pr_direct:Entity)
        WHERE 'PullRequest' IN labels(pr_direct)
          AND CASE
            WHEN $as_of IS NULL THEN mod.valid_to IS NULL
            ELSE (mod.valid_from IS NULL OR mod.valid_from <= $as_of)
             AND (mod.valid_to IS NULL OR mod.valid_to > $as_of)
          END
        OPTIONAL MATCH (f:FILE {repoId: $pot_id})
        WHERE f.file_path = n.file_path OR f.file_path ENDS WITH ('/' + n.file_path)
        OPTIONAL MATCH (f)-[tb:TOUCHED_BY]->(pr_file:Entity)
        WHERE 'PullRequest' IN labels(pr_file)
          AND CASE
            WHEN $as_of IS NULL THEN tb.valid_to IS NULL
            ELSE (tb.valid_from IS NULL OR tb.valid_from <= $as_of)
             AND (tb.valid_to IS NULL OR tb.valid_to > $as_of)
          END
        WITH n, [x IN collect(DISTINCT pr_direct) + collect(DISTINCT pr_file) WHERE x IS NOT NULL] AS prs
        UNWIND prs AS pr
        OPTIONAL MATCH (pr)-[:Fixes]->(iss:Entity)
        OPTIONAL MATCH (n)-[:HAS_DECISION]->(dec:Entity)
        OPTIONAL MATCH (pr)-[:HAS_REVIEW_DECISION]->(prdec:Decision)
        WITH n, pr,
             collect(DISTINCT coalesce(iss.issue_number, iss.number, iss.name)) AS fixed_issues,
             collect(DISTINCT coalesce(dec.decision_made, dec.name, dec.summary)) AS node_decisions,
             collect(DISTINCT coalesce(prdec.decision_made, prdec.name, prdec.summary)) AS pr_review_decisions
        WHERE pr IS NOT NULL
          AND CASE
            WHEN $as_of IS NULL THEN pr.valid_to IS NULL
            ELSE (pr.valid_to IS NULL OR pr.valid_to > $as_of)
          END
        RETURN coalesce(pr.pr_number, pr.number) AS pr_number,
               coalesce(pr.title, pr.name) AS title,
               coalesce(pr.why_summary, pr.summary, pr.description, '') AS why_summary,
               coalesce(pr.change_type, '') AS change_type,
               coalesce(pr.feature_area, '') AS feature_area,
               fixed_issues,
               [x IN node_decisions WHERE x IS NOT NULL AND toString(x) <> '']
               + [x IN pr_review_decisions WHERE x IS NOT NULL AND toString(x) <> ''] AS decisions
        ORDER BY pr_number DESC
        LIMIT $limit
        """
        return self._run_read(query, pot_id, file_path, function_name, limit, as_of=as_of)

    def get_pr_diff(
        self,
        pot_id: str,
        pr_number: int,
        file_path: str | None,
        limit: int,
        repo_name: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
        MATCH (f:FILE {repoId: $pot_id})-[r:TOUCHED_BY {pr_number: $pr_number}]->(pr:Entity)
        WHERE 'PullRequest' IN labels(pr)
          AND ($file_path IS NULL OR f.file_path = $file_path OR f.file_path ENDS WITH ('/' + $file_path))
          AND ($repo_name IS NULL OR coalesce(pr.entity_key, '') CONTAINS $repo_name)
        RETURN f.file_path AS file_path,
               coalesce(r.status, '') AS status,
               coalesce(r.additions, 0) AS additions,
               coalesce(r.deletions, 0) AS deletions,
               coalesce(r.patch_excerpt, '') AS patch_excerpt,
               coalesce(pr.title, pr.name) AS pr_title
        ORDER BY f.file_path ASC
        LIMIT $limit
        """
        return self._run_read(
            query,
            pot_id,
            file_path,
            None,
            limit,
            pr_number=pr_number,
            repo_name=repo_name,
        )

    def get_file_owners(
        self,
        pot_id: str,
        file_path: str,
        limit: int,
        repo_name: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
        MATCH (f:FILE {repoId: $pot_id, file_path: $file_path})-[:TOUCHED_BY]->(pr:Entity)
        WHERE 'PullRequest' IN labels(pr)
          AND ($repo_name IS NULL OR coalesce(pr.entity_key, '') CONTAINS $repo_name)
        OPTIONAL MATCH (pr)-[:AuthoredBy]->(dev:Entity)
        WITH coalesce(dev.github_login, dev.name, pr.author, 'unknown') AS github_login,
             count(DISTINCT pr) AS pr_count,
             max(coalesce(pr.merged_at, pr.updated_at, pr.created_at)) AS last_touched
        RETURN github_login, pr_count, last_touched
        ORDER BY pr_count DESC, last_touched DESC
        LIMIT $limit
        """
        drv = self._open()
        if drv is None:
            return []
        try:
            with drv.session() as session:
                res = session.run(
                    query,
                    pot_id=pot_id,
                    file_path=file_path,
                    repo_name=repo_name,
                    limit=max(1, min(limit, 50)),
                )
                return [record.data() for record in res]
        finally:
            drv.close()

    def get_decisions(
        self,
        pot_id: str,
        file_path: str | None,
        function_name: str | None,
        limit: int,
        repo_name: str | None = None,
        pr_number: int | None = None,
    ) -> list[dict[str, Any]]:
        lim = max(1, min(limit, 100))
        if pr_number is not None:
            pr_only = """
            MATCH (pr:Entity {group_id: $pot_id})-[:HAS_REVIEW_DECISION]->(d:Entity)
            WHERE 'PullRequest' IN labels(pr) AND 'Decision' IN labels(d)
              AND (
                coalesce(pr.pr_number, toIntegerOrNull(pr.number)) = $pr_number
                OR toIntegerOrNull(reverse(split(coalesce(pr.entity_key, ''), ':'))[0]) = $pr_number
              )
              AND ($file_path IS NULL OR coalesce(d.review_path, '') = $file_path)
            RETURN DISTINCT
                coalesce(d.entity_key, elementId(d)) AS _dedupe,
                coalesce(d.decision_made, d.name, d.summary) AS decision_made,
                coalesce(d.alternatives_rejected, '') AS alternatives_rejected,
                coalesce(d.rationale, d.summary, '') AS rationale,
                coalesce(pr.pr_number, pr.number) AS pr_number
            LIMIT $limit
            """
            drv = self._open()
            if drv is None:
                return []
            try:
                with drv.session() as session:
                    rows = [
                        r.data()
                        for r in session.run(
                            pr_only,
                            pot_id=pot_id,
                            file_path=file_path,
                            pr_number=pr_number,
                            limit=lim,
                        )
                    ]
                    for r in rows:
                        r.pop("_dedupe", None)
                    return rows
            finally:
                drv.close()

        code_query = """
        MATCH (n:NODE {repoId: $pot_id})
        WHERE ($file_path IS NULL OR n.file_path = $file_path)
          AND ($function_name IS NULL OR toLower(coalesce(n.name, '')) CONTAINS toLower($function_name))
        OPTIONAL MATCH (n)-[:HAS_DECISION]->(d:Entity)
        OPTIONAL MATCH (n)-[:MODIFIED_IN]->(pr:Entity)
        WHERE d IS NOT NULL AND 'Decision' IN labels(d)
        RETURN DISTINCT
            coalesce(d.entity_key, elementId(d)) AS _dedupe,
            coalesce(d.decision_made, d.name, d.summary) AS decision_made,
            coalesce(d.alternatives_rejected, '') AS alternatives_rejected,
            coalesce(d.rationale, d.summary, '') AS rationale,
            coalesce(pr.pr_number, pr.number) AS pr_number
        LIMIT $limit
        """
        pr_review_query = """
        MATCH (pr:Entity {group_id: $pot_id})-[:HAS_REVIEW_DECISION]->(d:Entity)
        WHERE 'PullRequest' IN labels(pr) AND 'Decision' IN labels(d)
          AND ($file_path IS NULL OR coalesce(d.review_path, '') = $file_path)
          AND ($repo_name IS NULL OR coalesce(pr.entity_key, '') CONTAINS $repo_name)
        RETURN DISTINCT
            coalesce(d.entity_key, elementId(d)) AS _dedupe,
            coalesce(d.decision_made, d.name, d.summary) AS decision_made,
            coalesce(d.alternatives_rejected, '') AS alternatives_rejected,
            coalesce(d.rationale, d.summary, '') AS rationale,
            coalesce(pr.pr_number, pr.number) AS pr_number
        LIMIT $limit
        """
        drv = self._open()
        if drv is None:
            return []
        try:
            with drv.session() as session:
                code_rows = [
                    r.data()
                    for r in session.run(
                        code_query,
                        pot_id=pot_id,
                        file_path=file_path,
                        function_name=function_name,
                        limit=lim,
                    )
                ]
                if function_name:
                    for r in code_rows:
                        r.pop("_dedupe", None)
                    return code_rows
                pr_rows = [
                    r.data()
                    for r in session.run(
                        pr_review_query,
                        pot_id=pot_id,
                        file_path=file_path,
                        repo_name=repo_name,
                        limit=lim,
                    )
                ]
                merged = _merge_decision_result_rows(code_rows, pr_rows, lim)
                for r in merged:
                    r.pop("_dedupe", None)
                return merged
        finally:
            drv.close()

    def get_pr_review_context(
        self,
        pot_id: str,
        pr_number: int,
        repo_name: str | None = None,
    ) -> dict[str, Any]:
        """Return PR metadata plus review threads linked via HAS_REVIEW_DECISION."""
        query = """
        MATCH (pr:Entity {group_id: $pid})
        WHERE 'PullRequest' IN labels(pr)
          AND (
            coalesce(pr.pr_number, pr.number) = $num
            OR (
              coalesce(pr.entity_key, '') <> ''
              AND reverse(split(pr.entity_key, ':'))[0] = toString($num)
            )
          )
          AND ($repo_name IS NULL OR coalesce(pr.entity_key, '') CONTAINS $repo_name)
        WITH pr
        ORDER BY CASE WHEN coalesce(pr.entity_key, '') <> '' THEN 0 ELSE 1 END ASC
        LIMIT 1
        OPTIONAL MATCH (pr)-[:HAS_COMMIT]->(c:Entity:Commit)
        OPTIONAL MATCH (pr)-[r:HAS_REVIEW_DECISION]->(d:Entity)
        WHERE d IS NOT NULL AND 'Decision' IN labels(d)
        RETURN coalesce(pr.pr_number, pr.number) AS pr_number,
               coalesce(pr.title, pr.name) AS pr_title,
               coalesce(pr.summary, '') AS pr_summary,
               coalesce(pr.description, '') AS pr_description,
               coalesce(pr.author, '') AS pr_author,
               coalesce(pr.merged_at, '') AS pr_merged_at,
               coalesce(pr.entity_key, '') AS pr_entity_key,
               [x IN collect(DISTINCT {
                 sha: c.sha,
                 message: c.summary,
                 author: c.author
               }) WHERE x.sha IS NOT NULL OR x.message IS NOT NULL] AS commits,
               collect(DISTINCT {
                 thread_id: coalesce(r.thread_id, d.thread_id),
                 file_path: d.review_path,
                 line: d.review_line,
                 headline: coalesce(d.decision_made, d.name),
                 full_discussion: coalesce(d.summary, '')
               }) AS review_threads
        """
        drv = self._open()
        if drv is None:
            return {
                "found": False,
                "pr_number": pr_number,
                "pr_title": None,
                "pr_summary": None,
                "pr_description": "",
                "pr_author": "",
                "pr_merged_at": "",
                "pr_entity_key": "",
                "commits": [],
                "review_threads": [],
            }
        try:
            with drv.session() as session:
                rec = session.run(
                    query, pid=pot_id, num=pr_number, repo_name=repo_name
                ).single()
                if not rec:
                    return {
                        "found": False,
                        "pr_number": pr_number,
                        "pr_title": None,
                        "pr_summary": None,
                        "pr_description": "",
                        "pr_author": "",
                        "pr_merged_at": "",
                        "pr_entity_key": "",
                        "commits": [],
                        "review_threads": [],
                    }
                row = rec.data()
                raw_threads = row.get("review_threads") or []
                threads = [
                    t
                    for t in raw_threads
                    if t
                    and (
                        t.get("thread_id")
                        or (t.get("full_discussion") or "").strip()
                        or (t.get("headline") or "").strip()
                    )
                ]
                return {
                    "found": True,
                    "pr_number": row.get("pr_number"),
                    "pr_title": row.get("pr_title"),
                    "pr_summary": row.get("pr_summary") or "",
                    "pr_description": row.get("pr_description") or "",
                    "pr_author": row.get("pr_author") or "",
                    "pr_merged_at": row.get("pr_merged_at") or "",
                    "pr_entity_key": row.get("pr_entity_key") or "",
                    "commits": row.get("commits") or [],
                    "review_threads": threads,
                }
        finally:
            drv.close()

    def get_project_graph(
        self,
        pot_id: str,
        pr_number: int | None,
        limit: int,
        scope: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a bounded canonical project-map projection for a pot.

        This intentionally returns compact node summaries plus relationship
        references. Full source payloads remain behind source integrations.
        """
        scope = scope or {}
        labels = _project_map_labels_for_includes(include)
        lim = max(1, min(limit, 100))
        drv = self._open()
        if drv is None:
            return {
                "pot_id": pot_id,
                "pr_number": pr_number,
                "limit": lim,
                "nodes": [],
                "edges": [],
                "message": "neo4j_unavailable",
            }

        node_query = """
        MATCH (n:Entity {group_id: $pot_id})
        WHERE any(lbl IN labels(n) WHERE lbl IN $labels)
          AND (
            $repo_name IS NULL
            OR coalesce(n.repo_name, '') = $repo_name
            OR coalesce(n.repository_key, '') CONTAINS $repo_name
            OR coalesce(n.entity_key, '') CONTAINS $repo_name
          )
          AND (
            size($services) = 0
            OR toLower(coalesce(n.name, '')) IN $services_lc
            OR toLower(coalesce(n.entity_key, '')) IN $services_lc
            OR any(service IN $services_lc WHERE toLower(coalesce(n.entity_key, '')) CONTAINS service)
          )
          AND (
            size($features) = 0
            OR toLower(coalesce(n.name, '')) IN $features_lc
            OR toLower(coalesce(n.entity_key, '')) IN $features_lc
            OR any(feature IN $features_lc WHERE toLower(coalesce(n.entity_key, '')) CONTAINS feature)
          )
          AND (
            $environment IS NULL
            OR toLower(coalesce(n.name, '')) = $environment_lc
            OR toLower(coalesce(n.environment_type, '')) = $environment_lc
            OR toLower(coalesce(n.entity_key, '')) CONTAINS $environment_lc
          )
          AND (
            $user IS NULL
            OR toLower(coalesce(n.name, '')) = $user_lc
            OR toLower(coalesce(n.github_login, '')) = $user_lc
            OR toLower(coalesce(n.email, '')) = $user_lc
          )
        WITH n
        ORDER BY
          CASE
            WHEN 'Service' IN labels(n) THEN 0
            WHEN 'Feature' IN labels(n) THEN 1
            WHEN 'Component' IN labels(n) THEN 2
            WHEN 'Document' IN labels(n) THEN 3
            ELSE 4
          END,
          coalesce(n.updated_at, n.created_at, 0) DESC,
          coalesce(n.name, n.title, n.entity_key, '') ASC
        LIMIT $limit
        OPTIONAL MATCH (n)-[out]->(m:Entity {group_id: $pot_id})
        WHERE any(lbl IN labels(m) WHERE lbl IN $labels)
        WITH n, collect(DISTINCT {
          type: type(out),
          direction: 'out',
          target_key: coalesce(m.entity_key, elementId(m)),
          target_labels: labels(m),
          target_name: coalesce(m.name, m.title, m.statement, m.entity_key)
        }) AS out_rels
        OPTIONAL MATCH (src:Entity {group_id: $pot_id})-[inr]->(n)
        WHERE any(lbl IN labels(src) WHERE lbl IN $labels)
        WITH n, out_rels, collect(DISTINCT {
          type: type(inr),
          direction: 'in',
          source_key: coalesce(src.entity_key, elementId(src)),
          source_labels: labels(src),
          source_name: coalesce(src.name, src.title, src.statement, src.entity_key)
        }) AS in_rels
        WITH n, out_rels + in_rels AS rels
        RETURN
          elementId(n) AS id,
          coalesce(n.entity_key, elementId(n)) AS entity_key,
          labels(n) AS labels,
          properties(n) AS properties,
          [rel IN rels WHERE rel.type IS NOT NULL][..12] AS relationships
        """
        edge_query = """
        MATCH (a:Entity {group_id: $pot_id})-[r]->(b:Entity {group_id: $pot_id})
        WHERE any(lbl IN labels(a) WHERE lbl IN $labels)
          AND any(lbl IN labels(b) WHERE lbl IN $labels)
        RETURN
          coalesce(a.entity_key, elementId(a)) AS from,
          type(r) AS type,
          coalesce(b.entity_key, elementId(b)) AS to,
          properties(r) AS properties
        LIMIT $limit
        """
        try:
            with drv.session() as session:
                params = _project_graph_params(pot_id, pr_number, lim, labels, scope)
                nodes = [record.data() for record in session.run(node_query, **params)]
                edges = [record.data() for record in session.run(edge_query, **params)]
                return {
                    "pot_id": pot_id,
                    "pr_number": pr_number,
                    "limit": lim,
                    "nodes": nodes,
                    "edges": edges,
                    "message": "ok",
                }
        except Exception as exc:
            logger.exception("get_project_graph failed pot=%s", pot_id)
            return {
                "pot_id": pot_id,
                "pr_number": pr_number,
                "limit": lim,
                "nodes": [],
                "edges": [],
                "message": str(exc),
            }
        finally:
            drv.close()

    def get_graph_overview(
        self,
        pot_id: str,
        *,
        top_entities_limit: int = 20,
    ) -> dict[str, Any]:
        """Aggregate per-label / per-edge / drift stats for a pot's structural graph.

        Returns counts keyed by raw Neo4j labels and edge types; the caller
        (``query_context.get_graph_overview``) merges ontology metadata
        (category, required properties, predicate family) on top.
        """
        empty: dict[str, Any] = {
            "pot_id": pot_id,
            "totals": {"entities": 0, "edges": 0, "entities_without_canonical_label": 0},
            "label_counts": {},
            "edge_counts": {},
            "lifecycle_distribution": {},
            "top_entities_by_degree": [],
            "message": "ok",
        }
        drv = self._open()
        if drv is None:
            empty["message"] = "neo4j_unavailable"
            return empty
        canonical_labels = list(ENTITY_TYPES.keys())
        try:
            with drv.session() as session:
                totals_row = session.run(
                    "MATCH (n:Entity {group_id: $pid}) RETURN count(n) AS cnt",
                    pid=pot_id,
                ).single()
                entity_total = int(totals_row["cnt"]) if totals_row else 0

                edge_total_row = session.run(
                    "MATCH (:Entity {group_id: $pid})-[r]->(:Entity {group_id: $pid}) "
                    "RETURN count(r) AS cnt",
                    pid=pot_id,
                ).single()
                edge_total = int(edge_total_row["cnt"]) if edge_total_row else 0

                label_rows = session.run(
                    "MATCH (n:Entity {group_id: $pid}) "
                    "UNWIND labels(n) AS lbl "
                    "RETURN lbl AS label, count(*) AS cnt",
                    pid=pot_id,
                ).data()
                label_counts: dict[str, int] = {}
                for row in label_rows:
                    lbl = row.get("label")
                    if lbl:
                        label_counts[str(lbl)] = int(row.get("cnt") or 0)

                no_canonical_row = session.run(
                    "MATCH (n:Entity {group_id: $pid}) "
                    "WHERE NONE(l IN labels(n) WHERE l IN $canon) "
                    "RETURN count(n) AS cnt",
                    pid=pot_id,
                    canon=canonical_labels,
                ).single()
                no_canonical = (
                    int(no_canonical_row["cnt"]) if no_canonical_row else 0
                )

                edge_rows = session.run(
                    "MATCH (:Entity {group_id: $pid})-[r]->(:Entity {group_id: $pid}) "
                    "RETURN type(r) AS etype, count(*) AS cnt",
                    pid=pot_id,
                ).data()
                edge_counts: dict[str, int] = {}
                for row in edge_rows:
                    et = row.get("etype")
                    if et:
                        edge_counts[str(et)] = int(row.get("cnt") or 0)

                lifecycle_rows = session.run(
                    "MATCH (:Entity {group_id: $pid})-[r]->(:Entity {group_id: $pid}) "
                    "WHERE r.lifecycle_status IS NOT NULL "
                    "RETURN r.lifecycle_status AS status, count(*) AS cnt",
                    pid=pot_id,
                ).data()
                lifecycle: dict[str, int] = {}
                for row in lifecycle_rows:
                    st = row.get("status")
                    if st:
                        lifecycle[str(st)] = int(row.get("cnt") or 0)

                top_limit = max(1, min(top_entities_limit, 100))
                top_rows = session.run(
                    """
                    MATCH (n:Entity {group_id: $pid})
                    OPTIONAL MATCH (n)-[out]->(:Entity {group_id: $pid})
                    WITH n, count(out) AS out_degree
                    OPTIONAL MATCH (:Entity {group_id: $pid})-[inr]->(n)
                    WITH n, out_degree, count(inr) AS in_degree
                    WITH n, out_degree + in_degree AS degree
                    WHERE degree > 0
                    RETURN
                      coalesce(n.entity_key, elementId(n)) AS entity_key,
                      labels(n) AS labels,
                      coalesce(n.name, n.title, n.statement, n.entity_key) AS name,
                      degree
                    ORDER BY degree DESC
                    LIMIT $lim
                    """,
                    pid=pot_id,
                    lim=top_limit,
                ).data()
                top_entities = [
                    {
                        "entity_key": r.get("entity_key"),
                        "labels": list(r.get("labels") or []),
                        "name": r.get("name"),
                        "degree": int(r.get("degree") or 0),
                    }
                    for r in top_rows
                ]

                return {
                    "pot_id": pot_id,
                    "totals": {
                        "entities": entity_total,
                        "edges": edge_total,
                        "entities_without_canonical_label": no_canonical,
                    },
                    "label_counts": label_counts,
                    "edge_counts": edge_counts,
                    "lifecycle_distribution": lifecycle,
                    "top_entities_by_degree": top_entities,
                    "message": "ok",
                }
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_graph_overview failed pot=%s", pot_id)
            empty["message"] = str(exc)
            return empty
        finally:
            drv.close()

    def get_debugging_memory(
        self,
        pot_id: str,
        limit: int,
        scope: dict[str, Any] | None = None,
        include: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Return compact reusable debugging memory for a pot."""
        scope = scope or {}
        labels = _debugging_memory_labels_for_includes(include)
        lim = max(1, min(limit, 100))
        drv = self._open()
        if drv is None:
            return {
                "pot_id": pot_id,
                "limit": lim,
                "nodes": [],
                "edges": [],
                "message": "neo4j_unavailable",
            }

        node_query = """
        MATCH (n:Entity {group_id: $pot_id})
        WHERE any(lbl IN labels(n) WHERE lbl IN $labels)
          AND (
            $query_text = ''
            OR toLower(
              coalesce(n.name, '') + ' ' +
              coalesce(n.title, '') + ' ' +
              coalesce(n.summary, '') + ' ' +
              coalesce(n.description, '') + ' ' +
              coalesce(n.root_cause, '') + ' ' +
              coalesce(n.signal_type, '') + ' ' +
              coalesce(n.fingerprint, '')
            ) CONTAINS $query_text
          )
          AND (
            size($services) = 0
            OR any(service IN $services_lc WHERE toLower(coalesce(n.entity_key, '')) CONTAINS service)
            OR any(service IN $services_lc WHERE toLower(coalesce(n.service, '')) CONTAINS service)
          )
          AND (
            $environment IS NULL
            OR toLower(coalesce(n.environment, '')) = $environment_lc
            OR toLower(coalesce(n.entity_key, '')) CONTAINS $environment_lc
          )
        WITH n
        ORDER BY
          CASE
            WHEN 'Fix' IN labels(n) THEN 0
            WHEN 'BugPattern' IN labels(n) THEN 1
            WHEN 'Investigation' IN labels(n) THEN 2
            WHEN 'Incident' IN labels(n) THEN 3
            WHEN 'Alert' IN labels(n) THEN 4
            ELSE 5
          END,
          coalesce(n.resolved_at, n.last_observed_at, n.updated_at, n.created_at, 0) DESC,
          coalesce(n.name, n.title, n.entity_key, '') ASC
        LIMIT $limit
        OPTIONAL MATCH (n)-[out]->(m:Entity {group_id: $pot_id})
        WHERE any(lbl IN labels(m) WHERE lbl IN $related_labels)
        WITH n, collect(DISTINCT {
          type: type(out),
          direction: 'out',
          target_key: coalesce(m.entity_key, elementId(m)),
          target_labels: labels(m),
          target_name: coalesce(m.name, m.title, m.summary, m.entity_key),
          properties: properties(out)
        }) AS out_rels
        OPTIONAL MATCH (src:Entity {group_id: $pot_id})-[inr]->(n)
        WHERE any(lbl IN labels(src) WHERE lbl IN $related_labels)
        WITH n, out_rels, collect(DISTINCT {
          type: type(inr),
          direction: 'in',
          source_key: coalesce(src.entity_key, elementId(src)),
          source_labels: labels(src),
          source_name: coalesce(src.name, src.title, src.summary, src.entity_key),
          properties: properties(inr)
        }) AS in_rels
        WITH n, out_rels + in_rels AS rels
        RETURN
          elementId(n) AS id,
          coalesce(n.entity_key, elementId(n)) AS entity_key,
          labels(n) AS labels,
          properties(n) AS properties,
          [rel IN rels WHERE rel.type IS NOT NULL][..16] AS relationships
        """
        edge_query = """
        MATCH (a:Entity {group_id: $pot_id})-[r]->(b:Entity {group_id: $pot_id})
        WHERE any(lbl IN labels(a) WHERE lbl IN $labels)
          AND any(lbl IN labels(b) WHERE lbl IN $related_labels)
        RETURN
          coalesce(a.entity_key, elementId(a)) AS from,
          type(r) AS type,
          coalesce(b.entity_key, elementId(b)) AS to,
          properties(r) AS properties
        LIMIT $limit
        """
        try:
            with drv.session() as session:
                params = _debugging_memory_params(
                    pot_id,
                    lim,
                    labels,
                    scope,
                    query,
                )
                nodes = [record.data() for record in session.run(node_query, **params)]
                edges = [record.data() for record in session.run(edge_query, **params)]
                return {
                    "pot_id": pot_id,
                    "limit": lim,
                    "nodes": nodes,
                    "edges": edges,
                    "message": "ok",
                }
        except Exception as exc:
            logger.exception("get_debugging_memory failed pot=%s", pot_id)
            return {
                "pot_id": pot_id,
                "limit": lim,
                "nodes": [],
                "edges": [],
                "message": str(exc),
            }
        finally:
            drv.close()

    def upsert_entities(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        if not items:
            return 0
        drv = self._open()
        if drv is None:
            return 0
        count = 0
        try:
            with drv.session() as session:
                for item in items:
                    props = dict(item.properties)
                    props["group_id"] = pot_id
                    props["provenance_source_event"] = provenance.source_event_id
                    session.run(
                        "MERGE (e:Entity {group_id: $gid, entity_key: $key}) "
                        "ON CREATE SET e.uuid = randomUUID(), e.created_at = timestamp() "
                        "SET e += $props",
                        gid=pot_id,
                        key=item.entity_key,
                        props=props,
                    )
                    for lbl in item.labels:
                        if lbl == "Entity":
                            continue
                        if not is_canonical_entity_label(lbl) or lbl not in ENTITY_TYPES:
                            continue
                        session.run(
                            f"MATCH (e:Entity {{group_id: $gid, entity_key: $key}}) SET e:{lbl}",  # pyright: ignore[reportArgumentType]
                            gid=pot_id,
                            key=item.entity_key,
                        )
                    count += 1
        finally:
            drv.close()
        return count

    def upsert_edges(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        if not items:
            return 0
        drv = self._open()
        if drv is None:
            return 0
        count = 0
        now_iso = _utc_now_iso()
        try:
            with drv.session() as session:
                for item in items:
                    props = dict(item.properties)
                    props.setdefault("valid_from", now_iso)
                    props["provenance_source_event"] = provenance.source_event_id
                    res = session.run(
                        f"MATCH (a:Entity {{group_id: $gid, entity_key: $from_key}}) "  # pyright: ignore[reportArgumentType]
                        f"MATCH (b:Entity {{group_id: $gid, entity_key: $to_key}}) "
                        f"MERGE (a)-[r:{item.edge_type}]->(b) "
                        "SET r += $props "
                        "RETURN count(r) AS cnt",
                        gid=pot_id,
                        from_key=item.from_entity_key,
                        to_key=item.to_entity_key,
                        props=props,
                    )
                    count += _count(res)
        finally:
            drv.close()
        return count

    def delete_edges(
        self,
        pot_id: str,
        items: list[EdgeDelete],
        provenance: ProvenanceRef,
    ) -> int:
        del provenance
        if not items:
            return 0
        drv = self._open()
        if drv is None:
            return 0
        count = 0
        try:
            with drv.session() as session:
                for item in items:
                    res = session.run(
                        f"MATCH (a:Entity {{group_id: $gid, entity_key: $from_key}})"  # pyright: ignore[reportArgumentType]
                        f"-[r:{item.edge_type}]->"
                        f"(b:Entity {{group_id: $gid, entity_key: $to_key}}) "
                        "DELETE r RETURN count(r) AS cnt",
                        gid=pot_id,
                        from_key=item.from_entity_key,
                        to_key=item.to_entity_key,
                    )
                    count += _count(res)
        finally:
            drv.close()
        return count

    def apply_invalidations(
        self,
        pot_id: str,
        items: list[InvalidationOp],
        provenance: ProvenanceRef,
    ) -> int:
        if not items:
            return 0
        drv = self._open()
        if drv is None:
            return 0
        count = 0
        now_iso = _utc_now_iso()
        try:
            with drv.session() as session:
                for item in items:
                    valid_to = item.valid_to or now_iso
                    if item.target_entity_key:
                        res = session.run(
                            "MATCH (e:Entity {group_id: $gid, entity_key: $key}) "
                            "SET e.valid_to = $valid_to, "
                            "    e.invalidation_reason = $reason, "
                            "    e.invalidated_by = $by "
                            "RETURN count(e) AS cnt",
                            gid=pot_id,
                            key=item.target_entity_key,
                            valid_to=valid_to,
                            reason=item.reason,
                            by=provenance.source_event_id,
                        )
                        matched = _count(res)
                        if matched and item.superseded_by_key:
                            session.run(
                                "MATCH (new:Entity {group_id: $gid, entity_key: $new_key}) "
                                "MATCH (old:Entity {group_id: $gid, entity_key: $old_key}) "
                                "MERGE (new)-[r:SUPERSEDES]->(old) "
                                "SET r.reason = $reason, r.superseded_at = $valid_to",
                                gid=pot_id,
                                new_key=item.superseded_by_key,
                                old_key=item.target_entity_key,
                                reason=item.reason,
                                valid_to=valid_to,
                            )
                        count += matched
                    elif item.target_edge:
                        edge_type, from_key, to_key = item.target_edge
                        if not _is_safe_cypher_identifier(edge_type):
                            logger.warning(
                                "apply_invalidations: skipping non-canonical edge_type %r", edge_type
                            )
                            continue
                        res = session.run(
                            f"MATCH (a:Entity {{group_id: $gid, entity_key: $from_key}})"  # pyright: ignore[reportArgumentType]
                            f"-[r:{edge_type}]->"
                            f"(b:Entity {{group_id: $gid, entity_key: $to_key}}) "
                            "SET r.valid_to = $valid_to, "
                            "    r.invalidation_reason = $reason "
                            "RETURN count(r) AS cnt",
                            gid=pot_id,
                            from_key=from_key,
                            to_key=to_key,
                            valid_to=valid_to,
                            reason=item.reason,
                        )
                        matched = _count(res)
                        if matched and item.superseded_by_key:
                            session.run(
                                "MATCH (new:Entity {group_id: $gid, entity_key: $new_key}) "
                                "MATCH (old:Entity {group_id: $gid, entity_key: $to_key}) "
                                "MERGE (new)-[r:SUPERSEDES]->(old) "
                                "SET r.reason = $reason, r.superseded_at = $valid_to",
                                gid=pot_id,
                                new_key=item.superseded_by_key,
                                to_key=to_key,
                                reason=item.reason,
                                valid_to=valid_to,
                            )
                        count += matched
        finally:
            drv.close()
        return count

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        """Remove structural ``Entity`` / ``FILE`` / ``NODE`` data for this pot (default Neo4j database)."""
        drv = self._open()
        if drv is None:
            return {
                "ok": False,
                "entity_deleted": 0,
                "file_deleted": 0,
                "node_deleted": 0,
                "error": "neo4j_unavailable",
            }
        q_entity = """
        MATCH (n:Entity {group_id: $pid})
        CALL (n) {
            DETACH DELETE n
        } IN TRANSACTIONS OF 500 ROWS
        """
        q_file = """
        MATCH (n:FILE {repoId: $pid})
        CALL (n) {
            DETACH DELETE n
        } IN TRANSACTIONS OF 500 ROWS
        """
        q_node = """
        MATCH (n:NODE {repoId: $pid})
        CALL (n) {
            DETACH DELETE n
        } IN TRANSACTIONS OF 500 ROWS
        """
        entity_deleted = 0
        file_deleted = 0
        node_deleted = 0
        try:
            with drv.session() as session:
                r = session.run(q_entity, pid=pot_id)
                entity_deleted = int(r.consume().counters.nodes_deleted)
                r = session.run(q_file, pid=pot_id)
                file_deleted = int(r.consume().counters.nodes_deleted)
                r = session.run(q_node, pid=pot_id)
                node_deleted = int(r.consume().counters.nodes_deleted)
        except Exception as exc:
            logger.warning("reset_pot structural: %s", exc)
            return {
                "ok": False,
                "entity_deleted": entity_deleted,
                "file_deleted": file_deleted,
                "node_deleted": node_deleted,
                "error": str(exc),
            }
        finally:
            drv.close()
        return {
            "ok": True,
            "entity_deleted": entity_deleted,
            "file_deleted": file_deleted,
            "node_deleted": node_deleted,
        }

    def expand_causal_neighbours(
        self,
        pot_id: str,
        node_uuids: list[str],
        *,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """One-hop episodic ``RELATES_TO`` expansion for hybrid search (see 04-causal-multihop)."""
        del depth  # reserved for future variable-length hops
        seeds = [str(u) for u in node_uuids if u][:12]
        if not seeds:
            return []
        drv = self._open()
        if drv is None:
            return []
        forward = sorted(
            {"CAUSED", "CAUSES", "PRECEDES", "TRIGGERED_BY"},
        )
        backward = sorted(
            {"CAUSED", "CAUSES", "PRECEDES", "TRIGGERED_BY", "DECIDES_FOR", "FIXES"},
        )
        query = """
        MATCH (seed:Entity {group_id: $gid})
        WHERE seed.uuid IN $seeds
        MATCH (seed)-[e:RELATES_TO]->(nb:Entity {group_id: $gid})
        WHERE toUpper(trim(e.name)) IN $forward_names
        RETURN nb.uuid AS neighbor_uuid,
               nb.name AS name,
               coalesce(nb.summary, nb.description, '') AS summary,
               e.uuid AS edge_uuid,
               toUpper(trim(e.name)) AS edge_name,
               seed.uuid AS seed_uuid,
               labels(nb) AS labels
        UNION
        MATCH (seed:Entity {group_id: $gid})
        WHERE seed.uuid IN $seeds
        MATCH (pred:Entity {group_id: $gid})-[e:RELATES_TO]->(seed)
        WHERE toUpper(trim(e.name)) IN $backward_names
        RETURN pred.uuid AS neighbor_uuid,
               pred.name AS name,
               coalesce(pred.summary, pred.description, '') AS summary,
               e.uuid AS edge_uuid,
               toUpper(trim(e.name)) AS edge_name,
               seed.uuid AS seed_uuid,
               labels(pred) AS labels
        """
        try:
            with drv.session() as session:
                res = session.run(
                    query,
                    gid=pot_id,
                    seeds=seeds,
                    forward_names=forward,
                    backward_names=backward,
                )
                return [record.data() for record in res]
        except Exception as exc:
            logger.exception("expand_causal_neighbours failed pot=%s: %s", pot_id, exc)
            return []
        finally:
            drv.close()

    @staticmethod
    def _causal_step_in_as_of_window(
        row: dict[str, Any],
        *,
        window_lo: datetime | None,
        window_hi: datetime | None,
    ) -> bool:
        """If ``as_of`` bounds are set, keep steps whose edge (or predecessor) time lies in-window."""
        if window_lo is None or window_hi is None:
            return True
        raw = row.get("valid_at")
        if raw is None:
            raw = row.get("pred_valid_at")
        if raw is None:
            return True
        try:
            if isinstance(raw, datetime):
                t = raw
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
            else:
                s = str(raw).replace("Z", "+00:00")
                t = datetime.fromisoformat(s)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return True
        return window_lo <= t <= window_hi

    @staticmethod
    def _as_of_window_bounds(
        as_of_iso: str | None, window_days: int
    ) -> tuple[datetime | None, datetime | None]:
        if not as_of_iso or not str(as_of_iso).strip() or window_days < 1:
            return None, None
        try:
            s = str(as_of_iso).strip().replace("Z", "+00:00")
            as_of = datetime.fromisoformat(s)
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            delta = timedelta(days=window_days)
            return as_of - delta, as_of + delta
        except (ValueError, TypeError):
            return None, None

    def walk_causal_chain_backward(
        self,
        pot_id: str,
        focal_node_uuid: str,
        *,
        max_depth: int = 6,
        as_of_iso: str | None = None,
        window_days: int = 180,
    ) -> list[dict[str, Any]]:
        """Walk predecessors along causal edge names; returns rows root → focal order."""
        cur = str(focal_node_uuid).strip()
        if not cur:
            return []
        drv = self._open()
        if drv is None:
            return []
        window_lo, window_hi = self._as_of_window_bounds(as_of_iso, window_days)
        allowed = sorted({"CAUSED", "CAUSES", "TRIGGERED_BY", "PRECEDES"})
        step_query = """
        MATCH (pred:Entity {group_id: $gid})-[e:RELATES_TO]->(cur:Entity {uuid: $cur, group_id: $gid})
        WHERE toUpper(trim(e.name)) IN $allowed
        RETURN pred.uuid AS uuid,
               pred.name AS name,
               coalesce(pred.summary, pred.description, '') AS summary,
               pred.source_ref AS source_ref,
               e.uuid AS edge_uuid,
               toUpper(trim(e.name)) AS edge_name,
               e.valid_at AS valid_at,
               e.invalid_at AS invalid_at,
               pred.valid_at AS pred_valid_at
        ORDER BY e.valid_at DESC NULLS LAST, pred.valid_at DESC NULLS LAST
        LIMIT 24
        """
        chain: list[dict[str, Any]] = []
        visited: set[str] = {cur}
        try:
            with drv.session() as session:
                for _ in range(max(1, min(max_depth, 20))):
                    rec = session.run(
                        step_query,
                        gid=pot_id,
                        cur=cur,
                        allowed=allowed,
                    )
                    rows = [record.data() for record in rec]
                    chosen: dict[str, Any] | None = None
                    for data in rows:
                        if not self._causal_step_in_as_of_window(
                            data, window_lo=window_lo, window_hi=window_hi
                        ):
                            continue
                        nxt = str(data.get("uuid") or "")
                        if not nxt or nxt in visited:
                            continue
                        chosen = data
                        break
                    if chosen is None:
                        break
                    data = chosen
                    nxt = str(data.get("uuid") or "")
                    visited.add(nxt)
                    chain.append(data)
                    cur = nxt
        except Exception as exc:
            logger.exception("walk_causal_chain_backward failed pot=%s: %s", pot_id, exc)
            return []
        finally:
            drv.close()
        chain.reverse()
        return chain

    def resolve_entity_uuid_for_service_hint(
        self,
        pot_id: str,
        service_hint: str,
    ) -> str | None:
        hint = (service_hint or "").strip()
        if not hint:
            return None
        drv = self._open()
        if drv is None:
            return None
        q = """
        MATCH (n:Entity {group_id: $gid})
        WHERE toLower(n.name) = $hl
           OR toLower(coalesce(n.entity_key, '')) CONTAINS $hp
        RETURN n.uuid AS uuid
        LIMIT 1
        """
        try:
            with drv.session() as session:
                rec = session.run(
                    q,
                    gid=pot_id,
                    hl=hint.lower(),
                    hp=hint.lower().replace(" ", "_"),
                )
                one = rec.single()
                if one is None:
                    return None
                uid = one.data().get("uuid")
                return str(uid) if uid else None
        except Exception as exc:
            logger.exception("resolve_entity_uuid_for_service_hint failed: %s", exc)
            return None
        finally:
            drv.close()

    def get_episodic_entity_node(
        self,
        pot_id: str,
        entity_uuid: str,
    ) -> dict[str, Any] | None:
        uid = str(entity_uuid).strip()
        if not uid:
            return None
        drv = self._open()
        if drv is None:
            return None
        q = """
        MATCH (n:Entity {group_id: $gid, uuid: $uid})
        RETURN n.uuid AS uuid,
               n.name AS name,
               coalesce(n.summary, n.description, '') AS summary,
               n.valid_at AS valid_at,
               n.source_ref AS source_ref
        LIMIT 1
        """
        try:
            with drv.session() as session:
                rec = session.run(q, gid=pot_id, uid=uid)
                row = rec.single()
                return row.data() if row is not None else None
        except Exception as exc:
            logger.exception("get_episodic_entity_node failed: %s", exc)
            return None
        finally:
            drv.close()

    def _run_read(
        self,
        query: str,
        pot_id: str,
        file_path: str | None,
        function_name: str | None,
        limit: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        drv = self._open()
        if drv is None:
            return []
        try:
            with drv.session() as session:
                params: dict[str, Any] = {
                    "pot_id": pot_id,
                    "file_path": file_path,
                    "function_name": function_name,
                    "limit": max(1, min(limit, 100)),
                }
                params.update(kwargs)
                res = session.run(
                    query,  # pyright: ignore[reportArgumentType]
                    **params,
                )
                return [record.data() for record in res]
        finally:
            drv.close()


_SAFE_CYPHER_IDENTIFIER_RE = None


def _is_safe_cypher_identifier(value: str) -> bool:
    """Return True only if value is a canonical ontology edge type (no Cypher injection risk)."""
    import re

    global _SAFE_CYPHER_IDENTIFIER_RE
    if _SAFE_CYPHER_IDENTIFIER_RE is None:
        _SAFE_CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
    if not _SAFE_CYPHER_IDENTIFIER_RE.match(value):
        return False
    from domain.ontology import is_canonical_edge_type

    return is_canonical_edge_type(value)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _count(neo4j_result) -> int:
    record = neo4j_result.single()
    return int(record["cnt"]) if record and record.get("cnt") is not None else 0


def _int(result) -> int:
    record = result.single()
    return int(record["cnt"]) if record and record.get("cnt") is not None else 0


def _project_map_labels_for_includes(include: list[str] | None) -> list[str]:
    if not include:
        include = [
            "purpose",
            "repo_map",
            "service_map",
            "feature_map",
            "docs",
            "deployments",
            "runbooks",
            "local_workflows",
            "scripts",
            "config",
            "preferences",
            "agent_instructions",
        ]
    labels: list[str] = []
    seen: set[str] = set()
    for raw in include:
        for label in _PROJECT_MAP_LABELS_BY_INCLUDE.get(raw, ()):
            if label in seen:
                continue
            labels.append(label)
            seen.add(label)
    return labels or ["Service", "Feature", "Repository", "Document"]


def _project_graph_params(
    pot_id: str,
    pr_number: int | None,
    limit: int,
    labels: list[str],
    scope: dict[str, Any],
) -> dict[str, Any]:
    services = [str(value) for value in scope.get("services") or [] if value]
    features = [str(value) for value in scope.get("features") or [] if value]
    environment = scope.get("environment")
    user = scope.get("user")
    return {
        "pot_id": pot_id,
        "pr_number": pr_number,
        "limit": limit,
        "labels": labels,
        "repo_name": scope.get("repo_name"),
        "services": services,
        "services_lc": [value.lower() for value in services],
        "features": features,
        "features_lc": [value.lower() for value in features],
        "environment": environment,
        "environment_lc": str(environment).lower() if environment else None,
        "user": user,
        "user_lc": str(user).lower() if user else None,
    }


def _debugging_memory_labels_for_includes(include: list[str] | None) -> list[str]:
    if not include:
        include = ["prior_fixes", "diagnostic_signals", "incidents", "alerts"]
    labels: list[str] = []
    seen: set[str] = set()
    for raw in include:
        for label in _DEBUGGING_MEMORY_LABELS_BY_INCLUDE.get(raw, ()):
            if label in seen:
                continue
            labels.append(label)
            seen.add(label)
    return labels or ["Fix", "BugPattern", "Investigation", "DiagnosticSignal"]


def _debugging_memory_params(
    pot_id: str,
    limit: int,
    labels: list[str],
    scope: dict[str, Any],
    query: str | None,
) -> dict[str, Any]:
    services = [str(value) for value in scope.get("services") or [] if value]
    environment = scope.get("environment")
    related = set(labels)
    related.update(
        {
            "Service",
            "Environment",
            "Component",
            "Repository",
            "PullRequest",
            "Commit",
            "Runbook",
            "DiagnosticSignal",
            "Incident",
            "Alert",
            "BugPattern",
            "Investigation",
            "Fix",
            "SourceReference",
        }
    )
    return {
        "pot_id": pot_id,
        "limit": limit,
        "labels": labels,
        "related_labels": sorted(related),
        "services": services,
        "services_lc": [value.lower() for value in services],
        "environment": environment,
        "environment_lc": str(environment).lower() if environment else None,
        "query_text": (query or "").strip().lower()[:240],
    }

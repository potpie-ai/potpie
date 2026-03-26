"""Neo4j structural graph: bridges, entity stamping, agent queries."""

from __future__ import annotations

import logging
from typing import Any, Optional

from neo4j import Driver, GraphDatabase

from domain.deterministic_extractors import parse_diff_hunks
from domain.ingestion import BridgeResult
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.structural_graph import StructuralGraphPort

logger = logging.getLogger(__name__)


def _merge_decision_result_rows(
    code_rows: list[dict[str, Any]],
    pr_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in code_rows + pr_rows:
        dedupe = row.pop("_dedupe", None)
        key = str(dedupe) if dedupe is not None else f"{row.get('decision_made')}|{row.get('pr_number')}"
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

    def write_bridges(
        self,
        project_id: str,
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
                        MATCH (file:FILE {repoId: $project_id})
                        WHERE file.file_path = $file_path
                           OR file.file_path ENDS WITH ('/' + $file_path)
                        MATCH (pr:Entity {group_id: $project_id, entity_key: $pr_key})
                        MERGE (file)-[r:TOUCHED_BY {pr_number: $pr_number}]->(pr)
                        SET r.updated_at = timestamp(),
                            r.status = $status,
                            r.additions = $additions,
                            r.deletions = $deletions,
                            r.patch_excerpt = $patch_excerpt
                        RETURN count(r) AS cnt
                        """,
                        project_id=project_id,
                        file_path=file_path,
                        pr_key=pr_entity_key,
                        pr_number=pr_number,
                        status=status,
                        additions=additions,
                        deletions=deletions,
                        patch_excerpt=patch_excerpt,
                    )
                    result.touched_by += _count(res)

                    for start_line, end_line in parse_diff_hunks(f.get("patch")):
                        res = session.run(
                            """
                            MATCH (n:NODE {repoId: $project_id})
                            WHERE (n.file_path = $file_path
                                   OR n.file_path ENDS WITH ('/' + $file_path))
                              AND any(lbl IN labels(n) WHERE lbl IN ['FUNCTION', 'CLASS'])
                              AND n.start_line <= $end_line
                              AND n.end_line >= $start_line
                            MATCH (pr:Entity {group_id: $project_id, entity_key: $pr_key})
                            MERGE (n)-[r:MODIFIED_IN {pr_number: $pr_number}]->(pr)
                            SET r.merged_at = $merged_at,
                                r.updated_at = timestamp(),
                                r.is_approximate = $is_approximate
                            RETURN count(r) AS cnt
                            """,
                            project_id=project_id,
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

                    decision_key = f"github:decision:{repo_name}:{pr_number}:{thread_id!s}"

                    res = session.run(
                        """
                        MATCH (n:NODE {repoId: $project_id})
                        WHERE (n.file_path = $file_path
                               OR n.file_path ENDS WITH ('/' + $file_path))
                          AND any(lbl IN labels(n) WHERE lbl IN ['FUNCTION', 'CLASS'])
                          AND n.start_line <= $line
                          AND n.end_line >= $line
                        WITH n
                        MATCH (d:Entity {group_id: $project_id, entity_key: $decision_key})
                        MERGE (n)-[r:HAS_DECISION]->(d)
                        SET r.updated_at = timestamp()
                        RETURN count(r) AS cnt
                        """,
                        project_id=project_id,
                        file_path=path,
                        line=line,
                        decision_key=decision_key,
                    )
                    result.has_decision += _count(res)

        except Exception:
            logger.exception("write_bridges failed project=%s pr=%s", project_id, pr_number)
            raise
        finally:
            drv.close()

        logger.info(
            "Bridge write complete project=%s pr=%s edges=%s",
            project_id,
            pr_number,
            result.as_dict(),
        )
        return result

    def stamp_pr_entities(
        self,
        project_id: str,
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
                        gid=project_id,
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
                        gid=project_id,
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
                        gid=project_id,
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
                            gid=project_id,
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
                    gid=project_id,
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
                    gid=project_id,
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
                        gid=project_id,
                        sha=sha[:12],
                        key=commit_key,
                    )
                    counts["stamped_commits"] += _int(result)
                    message = (commit.get("message") or "").strip()
                    first_line = message.splitlines()[0] if message else f"Commit {sha[:12]}"
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
                        gid=project_id,
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
                        gid=project_id,
                        login=login,
                        key=dev_key,
                    )
                    counts["stamped_developers"] += _int(result)

                for thread in review_threads or []:
                    thread_id = thread.get("thread_id")
                    if thread_id is None:
                        continue
                    decision_key = f"github:decision:{repo_name}:{pr_number}:{thread_id!s}"

                    first_body = ""
                    comments = thread.get("comments") or []
                    if comments:
                        first_body = (comments[0].get("body") or "").strip()[:120].lower()

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
                        gid=project_id,
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
                            full_text[:240] if full_text != "(empty thread)" else "(no text in thread)"
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
                        gid=project_id,
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
                    conv_key = f"github:decision:{repo_name}:{pr_number}:pr_conversation"
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
                        gid=project_id,
                        dkey=conv_key,
                        dname=(f"PR #{pr_number} conversation")[:200],
                        full=conv_full,
                        excerpt=conv_excerpt or "PR timeline comments",
                        pkey=pr_key,
                    )
                    counts["pr_conversation_linked"] = 1

        except Exception:
            logger.exception("stamp_pr_entities failed project=%s episode=%s", project_id, episode_uuid)
            raise
        finally:
            drv.close()

        logger.info(
            "Entity key stamping complete project=%s pr=%s counts=%s",
            project_id,
            pr_number,
            counts,
        )
        return counts

    def get_change_history(
        self,
        project_id: str,
        function_name: str | None,
        file_path: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = """
        MATCH (n:NODE {repoId: $project_id})
        WHERE ($file_path IS NULL OR n.file_path = $file_path)
          AND ($function_name IS NULL OR toLower(coalesce(n.name, '')) CONTAINS toLower($function_name))
        OPTIONAL MATCH (n)-[:MODIFIED_IN]->(pr_direct:Entity)
        WHERE 'PullRequest' IN labels(pr_direct)
        OPTIONAL MATCH (f:FILE {repoId: $project_id})
        WHERE f.file_path = n.file_path OR f.file_path ENDS WITH ('/' + n.file_path)
        OPTIONAL MATCH (f)-[:TOUCHED_BY]->(pr_file:Entity)
        WHERE 'PullRequest' IN labels(pr_file)
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
        return self._run_read(query, project_id, file_path, function_name, limit)

    def get_pr_diff(
        self,
        project_id: str,
        pr_number: int,
        file_path: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = """
        MATCH (f:FILE {repoId: $project_id})-[r:TOUCHED_BY {pr_number: $pr_number}]->(pr:Entity)
        WHERE 'PullRequest' IN labels(pr)
          AND ($file_path IS NULL OR f.file_path = $file_path OR f.file_path ENDS WITH ('/' + $file_path))
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
            project_id,
            file_path,
            None,
            limit,
            pr_number=pr_number,
        )

    def get_file_owners(
        self,
        project_id: str,
        file_path: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = """
        MATCH (f:FILE {repoId: $project_id, file_path: $file_path})-[:TOUCHED_BY]->(pr:Entity)
        WHERE 'PullRequest' IN labels(pr)
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
                    project_id=project_id,
                    file_path=file_path,
                    limit=max(1, min(limit, 50)),
                )
                return [record.data() for record in res]
        finally:
            drv.close()

    def get_decisions(
        self,
        project_id: str,
        file_path: str | None,
        function_name: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        lim = max(1, min(limit, 100))
        code_query = """
        MATCH (n:NODE {repoId: $project_id})
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
        MATCH (pr:Entity {group_id: $project_id})-[:HAS_REVIEW_DECISION]->(d:Entity)
        WHERE 'PullRequest' IN labels(pr) AND 'Decision' IN labels(d)
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
                code_rows = [r.data() for r in session.run(code_query, project_id=project_id, file_path=file_path, function_name=function_name, limit=lim)]
                if function_name:
                    for r in code_rows:
                        r.pop("_dedupe", None)
                    return code_rows
                pr_rows = [r.data() for r in session.run(pr_review_query, project_id=project_id, file_path=file_path, limit=lim)]
                merged = _merge_decision_result_rows(code_rows, pr_rows, lim)
                for r in merged:
                    r.pop("_dedupe", None)
                return merged
        finally:
            drv.close()

    def get_pr_review_context(self, project_id: str, pr_number: int) -> dict[str, Any]:
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
                rec = session.run(query, pid=project_id, num=pr_number).single()
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

    def _run_read(
        self,
        query: str,
        project_id: str,
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
                    "project_id": project_id,
                    "file_path": file_path,
                    "function_name": function_name,
                    "limit": max(1, min(limit, 100)),
                }
                params.update(kwargs)
                res = session.run(
                    query,
                    **params,
                )
                return [record.data() for record in res]
        finally:
            drv.close()


def _count(neo4j_result) -> int:
    record = neo4j_result.single()
    return int(record["cnt"]) if record and record.get("cnt") is not None else 0


def _int(result) -> int:
    record = result.single()
    return int(record["cnt"]) if record and record.get("cnt") is not None else 0

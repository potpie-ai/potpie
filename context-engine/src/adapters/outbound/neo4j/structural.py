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

                    res = session.run(
                        """
                        MATCH (file:FILE {repoId: $project_id})
                        WHERE file.file_path = $file_path
                           OR file.file_path ENDS WITH ('/' + $file_path)
                        MATCH (pr:Entity {group_id: $project_id, entity_key: $pr_key})
                        MERGE (file)-[r:TOUCHED_BY {pr_number: $pr_number}]->(pr)
                        SET r.updated_at = timestamp()
                        RETURN count(r) AS cnt
                        """,
                        project_id=project_id,
                        file_path=file_path,
                        pr_key=pr_entity_key,
                        pr_number=pr_number,
                    )
                    result.touched_by += _count(res)

                    if is_live:
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
                                    r.updated_at = timestamp()
                                RETURN count(r) AS cnt
                                """,
                                project_id=project_id,
                                file_path=file_path,
                                start_line=start_line,
                                end_line=end_line,
                                pr_key=pr_entity_key,
                                pr_number=pr_number,
                                merged_at=merged_at,
                            )
                            result.modified_in += _count(res)

                for thread in review_threads or []:
                    thread_id = thread.get("thread_id")
                    path = thread.get("path")
                    line = thread.get("line")
                    if not path or line is None or thread_id is None:
                        continue

                    decision_key = f"github:decision:{repo_name}:{pr_number}:{thread_id}"

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
        author: str | None,
    ) -> dict[str, int]:
        pr_key = f"github:pr:{repo_name}:{pr_number}"
        counts: dict[str, int] = {
            "stamped_pr": 0,
            "stamped_commits": 0,
            "stamped_developers": 0,
            "stamped_decisions": 0,
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
                    decision_key = f"github:decision:{repo_name}:{pr_number}:{thread_id}"

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
        OPTIONAL MATCH (n)-[:MODIFIED_IN]->(pr:Entity)
        WHERE 'PullRequest' IN labels(pr)
        OPTIONAL MATCH (pr)-[:Fixes]->(iss:Entity)
        OPTIONAL MATCH (n)-[:HAS_DECISION]->(dec:Entity)
        WITH n, pr,
             collect(DISTINCT coalesce(iss.issue_number, iss.number, iss.name)) AS fixed_issues,
             collect(DISTINCT coalesce(dec.decision_made, dec.name, dec.summary)) AS decisions
        WHERE pr IS NOT NULL
        RETURN coalesce(pr.pr_number, pr.number) AS pr_number,
               coalesce(pr.title, pr.name) AS title,
               coalesce(pr.why_summary, pr.summary, '') AS why_summary,
               coalesce(pr.change_type, '') AS change_type,
               coalesce(pr.feature_area, '') AS feature_area,
               fixed_issues,
               decisions
        ORDER BY pr_number DESC
        LIMIT $limit
        """
        return self._run_read(query, project_id, file_path, function_name, limit)

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
        query = """
        MATCH (n:NODE {repoId: $project_id})
        WHERE ($file_path IS NULL OR n.file_path = $file_path)
          AND ($function_name IS NULL OR toLower(coalesce(n.name, '')) CONTAINS toLower($function_name))
        OPTIONAL MATCH (n)-[:HAS_DECISION]->(d:Entity)
        OPTIONAL MATCH (n)-[:MODIFIED_IN]->(pr:Entity)
        WHERE d IS NOT NULL AND 'Decision' IN labels(d)
        RETURN DISTINCT
            coalesce(d.decision_made, d.name, d.summary) AS decision_made,
            coalesce(d.alternatives_rejected, '') AS alternatives_rejected,
            coalesce(d.rationale, d.summary, '') AS rationale,
            coalesce(pr.pr_number, pr.number) AS pr_number
        LIMIT $limit
        """
        return self._run_read(query, project_id, file_path, function_name, limit)

    def _run_read(
        self,
        query: str,
        project_id: str,
        file_path: str | None,
        function_name: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        drv = self._open()
        if drv is None:
            return []
        try:
            with drv.session() as session:
                res = session.run(
                    query,
                    project_id=project_id,
                    file_path=file_path,
                    function_name=function_name,
                    limit=max(1, min(limit, 100)),
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

"""Stamp Graphiti-created entities with deterministic entity_key values.

After Graphiti ingests an episode and creates Entity nodes via LLM extraction,
we post-stamp those entities with stable, predictable keys so that the bridge
writer (and any future consumer) can match them without depending on
LLM-generated names.

Key format:
  github:pr:<repo_name>:<pr_number>
  github:commit:<repo_name>:<sha>
  github:user:<login>
  github:decision:<repo_name>:<pr_number>:<thread_id>
"""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from app.core.config_provider import config_provider
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _driver():
    cfg = config_provider.get_neo4j_config()
    return GraphDatabase.driver(
        cfg.get("uri"),
        auth=(cfg.get("username"), cfg.get("password")),
    )


def _ensure_entity_key_index(session) -> None:
    session.run(
        "CREATE INDEX entity_key_group_idx IF NOT EXISTS "
        "FOR (e:Entity) ON (e.entity_key, e.group_id)"
    )


def stamp_pr_entities(
    project_id: str,
    episode_uuid: str,
    repo_name: str,
    pr_number: int,
    commits: list[dict[str, Any]],
    review_threads: list[dict[str, Any]],
    author: str | None = None,
) -> dict[str, int]:
    """Stamp all entities reachable from an episode with deterministic keys.

    Returns a dict of counts: {stamped_pr, stamped_commits, stamped_developers, stamped_decisions}.
    """
    pr_key = f"github:pr:{repo_name}:{pr_number}"
    counts: dict[str, int] = {
        "stamped_pr": 0,
        "stamped_commits": 0,
        "stamped_developers": 0,
        "stamped_decisions": 0,
    }

    drv = _driver()
    try:
        with drv.session() as session:
            _ensure_entity_key_index(session)

            # --- PullRequest entity ---
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

            # --- Commit entities ---
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

            # --- Developer entities ---
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

            # --- Decision entities ---
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
        logger.exception(
            "Failed stamping entity keys for project=%s episode=%s",
            project_id,
            episode_uuid,
        )
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


def _int(result) -> int:
    record = result.single()
    return int(record["cnt"]) if record and record.get("cnt") is not None else 0

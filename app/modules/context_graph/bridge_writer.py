"""Write bridge relationships between code-graph nodes and context entities.

All matching uses deterministic ``entity_key`` (set by entity_key_stamper)
rather than the LLM-generated ``name`` field.

Bridge types:
  FILE  -[:TOUCHED_BY]->  Entity:PullRequest
  NODE  -[:MODIFIED_IN]-> Entity:PullRequest   (live/webhook only, hunk-backed)
  NODE  -[:HAS_DECISION]-> Entity:Decision      (thread-id-backed)
"""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from app.core.config_provider import config_provider
from app.modules.context_graph.deterministic_extractors import parse_diff_hunks
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _driver():
    cfg = config_provider.get_neo4j_config()
    return GraphDatabase.driver(
        cfg.get("uri"),
        auth=(cfg.get("username"), cfg.get("password")),
    )


def _ensure_indexes(session) -> None:
    session.run(
        "CREATE INDEX node_file_repo_idx IF NOT EXISTS "
        "FOR (n:NODE) ON (n.file_path, n.repoId)"
    )
    session.run(
        "CREATE INDEX entity_key_group_idx IF NOT EXISTS "
        "FOR (e:Entity) ON (e.entity_key, e.group_id)"
    )


class BridgeResult:
    """Counts of edges written per type — for logging and diagnostics."""

    __slots__ = ("touched_by", "modified_in", "has_decision")

    def __init__(self) -> None:
        self.touched_by = 0
        self.modified_in = 0
        self.has_decision = 0

    def total(self) -> int:
        return self.touched_by + self.modified_in + self.has_decision

    def as_dict(self) -> dict[str, int]:
        return {
            "touched_by": self.touched_by,
            "modified_in": self.modified_in,
            "has_decision": self.has_decision,
        }


def write_bridges(
    project_id: str,
    pr_entity_key: str,
    pr_number: int,
    repo_name: str,
    files_with_patches: list[dict[str, Any]],
    review_threads: list[dict[str, Any]],
    merged_at: str | None = None,
    is_live: bool = False,
) -> BridgeResult:
    """Create TOUCHED_BY / MODIFIED_IN / HAS_DECISION edges.

    Matches the PR entity via ``entity_key`` (deterministic, set by stamper).
    """
    result = BridgeResult()
    drv = _driver()
    try:
        with drv.session() as session:
            _ensure_indexes(session)

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
        logger.exception(
            "Failed writing bridges for project=%s pr=%s",
            project_id,
            pr_number,
        )
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


def _count(neo4j_result) -> int:
    record = neo4j_result.single()
    return int(record["cnt"]) if record and record.get("cnt") is not None else 0

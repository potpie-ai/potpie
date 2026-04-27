"""Diagnose context-graph bridge edges and entity_key stamping in Neo4j.

Usage:
  uv run python scripts/check_context_graph_bridges.py --project-id <project_id>
  uv run python scripts/check_context_graph_bridges.py --project-id <project_id> --pr-number 1
  uv run python scripts/check_context_graph_bridges.py --project-id <project_id> --repo-name owner/repo --pr-number 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neo4j import GraphDatabase  # noqa: E402

from app.core.config_provider import config_provider  # noqa: E402


def _connect():
    cfg = config_provider.get_neo4j_config()
    return GraphDatabase.driver(
        cfg.get("uri"),
        auth=(cfg.get("username"), cfg.get("password")),
    )


def _val(session, query: str, **params: Any) -> int:
    record = session.run(query, **params).single()
    return int(record[0]) if record and record[0] is not None else 0


def _rows(session, query: str, **params: Any) -> list[dict]:
    return [dict(r) for r in session.run(query, **params)]


def run(project_id: str, repo_name: str | None, pr_number: int | None) -> int:
    print("=" * 56)
    print("  Context Graph Bridge Diagnostics (v2 — entity_key)")
    print("=" * 56)
    print(f"  project_id : {project_id}")
    print(f"  repo_name  : {repo_name or '(auto-detect)'}")
    print(f"  pr_number  : {pr_number if pr_number is not None else '(auto-detect)'}")
    print()

    with _connect() as driver:
        with driver.session() as s:

            # --- 1. Node counts ---
            file_cnt = _val(s, "MATCH (:FILE {repoId: $p}) RETURN count(*)", p=project_id)
            node_cnt = _val(s, "MATCH (:NODE {repoId: $p}) RETURN count(*)", p=project_id)
            entity_cnt = _val(
                s, "MATCH (:Entity {group_id: $p}) RETURN count(*)", p=project_id
            )
            pr_cnt = _val(
                s,
                "MATCH (e:Entity {group_id:$p}) WHERE 'PullRequest' IN labels(e) RETURN count(e)",
                p=project_id,
            )
            episodic_cnt = _val(
                s, "MATCH (:Episodic {group_id: $p}) RETURN count(*)", p=project_id
            )

            print("1. NODE COUNTS")
            print(f"   FILE nodes        : {file_cnt}")
            print(f"   NODE nodes        : {node_cnt}")
            print(f"   Entity nodes      : {entity_cnt}")
            print(f"   PullRequest       : {pr_cnt}")
            print(f"   Episodic          : {episodic_cnt}")
            print()

            # --- 2. Bridge edge counts ---
            tb = _val(
                s,
                "MATCH (:FILE {repoId:$p})-[r:TOUCHED_BY]->(:Entity {group_id:$p}) RETURN count(r)",
                p=project_id,
            )
            mi = _val(
                s,
                "MATCH (:NODE {repoId:$p})-[r:MODIFIED_IN]->(:Entity {group_id:$p}) RETURN count(r)",
                p=project_id,
            )
            hd = _val(
                s,
                "MATCH (:NODE {repoId:$p})-[r:HAS_DECISION]->(:Entity {group_id:$p}) RETURN count(r)",
                p=project_id,
            )
            print("2. BRIDGE EDGES")
            print(f"   TOUCHED_BY        : {tb}")
            print(f"   MODIFIED_IN       : {mi}")
            print(f"   HAS_DECISION      : {hd}")
            print()

            # --- 3. entity_key coverage ---
            stamped = _val(
                s,
                """MATCH (e:Entity {group_id:$p})
                   WHERE e.entity_key IS NOT NULL AND e.entity_key <> ''
                   RETURN count(e)""",
                p=project_id,
            )
            unstamped = _val(
                s,
                """MATCH (e:Entity {group_id:$p})
                   WHERE e.entity_key IS NULL OR e.entity_key = ''
                   RETURN count(e)""",
                p=project_id,
            )
            print("3. ENTITY_KEY COVERAGE")
            print(f"   Stamped           : {stamped}")
            print(f"   Unstamped         : {unstamped}")
            print()

            entities = _rows(
                s,
                """MATCH (e:Entity {group_id:$p})
                   RETURN labels(e) AS labels,
                          e.name AS name,
                          e.entity_key AS entity_key
                   ORDER BY e.entity_key, e.name
                   LIMIT 30""",
                p=project_id,
            )
            print("   Entity details:")
            for e in entities:
                key = e.get("entity_key") or "(none)"
                name = e.get("name") or "(unnamed)"
                lbls = ", ".join(l for l in (e.get("labels") or []) if l != "Entity")
                print(f"   - [{lbls:20s}] name={name:30s}  entity_key={key}")
            print()

            # --- 4. PR entity key check ---
            if pr_number is not None and repo_name:
                expected_key = f"github:pr:{repo_name}:{pr_number}"
                found = _val(
                    s,
                    "MATCH (e:Entity {group_id:$p, entity_key:$k}) RETURN count(e)",
                    p=project_id,
                    k=expected_key,
                )
                print("4. PR ENTITY_KEY MATCH")
                print(f"   Expected key      : {expected_key}")
                print(f"   Found             : {found}")
                if found == 0:
                    print("   STATUS            : MISSING — stamper did not run or failed")
                else:
                    print("   STATUS            : OK")
                print()

            # --- 5. Diagnosis ---
            print("5. DIAGNOSIS")
            issues: list[str] = []
            if entity_cnt == 0:
                issues.append("No Entity nodes — Graphiti ingestion has not run.")
            if stamped == 0 and entity_cnt > 0:
                issues.append(
                    "No entity_key stamped — stamper did not run after ingestion. "
                    "Re-ingest or run stamper manually."
                )
            if tb == 0 and stamped > 0 and file_cnt > 0:
                issues.append(
                    "TOUCHED_BY edges are 0 despite stamped entities and FILE nodes. "
                    "Check that PR file paths match code graph file_path values."
                )

            if not issues:
                print("   No issues detected.")
            else:
                for i, msg in enumerate(issues, 1):
                    print(f"   [{i}] {msg}")
            print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose context graph bridge edges.")
    parser.add_argument("--project-id", required=True, help="Project / group_id")
    parser.add_argument("--repo-name", default=None, help="GitHub repo full name (owner/repo)")
    parser.add_argument("--pr-number", type=int, default=None, help="PR number to check")
    args = parser.parse_args()

    try:
        return run(
            project_id=args.project_id,
            repo_name=args.repo_name,
            pr_number=args.pr_number,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

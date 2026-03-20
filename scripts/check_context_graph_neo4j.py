#!/usr/bin/env python3
"""
Check what context-graph (Graphiti) data exists in Neo4j: Episodic nodes, commits, comments.

Usage (from repo root):
  uv run python scripts/check_context_graph_neo4j.py

Uses NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD from .env (same as the app).
"""

import os
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USERNAME") or ""
    password = os.getenv("NEO4J_PASSWORD") or ""

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("Install neo4j: uv add neo4j")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password) if user else None)

    def run(query: str, **kwargs):
        with driver.session() as session:
            result = session.run(query, **kwargs)
            return list(result)

    print("=== Context graph (Graphiti) in Neo4j ===\n")
    print(f"Neo4j: {uri}\n")

    # 1. All labels in the DB
    labels_result = run("CALL db.labels() YIELD label RETURN label")
    labels = [r["label"] for r in labels_result]
    print("Labels in DB:", ", ".join(sorted(labels)))
    print()

    if "Episodic" not in labels:
        print("No 'Episodic' label found. Context graph (Graphiti) has not written any episodes yet.")
        print("  - Ensure CONTEXT_GRAPH_ENABLED=true and a Celery worker is consuming context-graph-etl.")
        print("  - Trigger sync from Sources page (Sync now) or run: uv run python -m scripts.context_graph_backfill_all")
        driver.close()
        return

    # 2. Count Episodic nodes
    count_result = run("MATCH (e:Episodic) RETURN count(e) AS c")
    total = count_result[0]["c"] if count_result else 0
    print(f"Episodic nodes (episodes): {total}")
    if total == 0:
        print("  No episodes in the graph. Run a context-graph sync (Sources → Sync now).")
        driver.close()
        return

    # 3. Property keys on Episodic nodes
    keys_result = run(
        "MATCH (e:Episodic) WITH e LIMIT 1 RETURN keys(e) AS k"
    )
    raw_keys = keys_result[0]["k"] if keys_result else []
    prop_keys = list(raw_keys) if isinstance(raw_keys, (list, tuple)) else [raw_keys]
    print("Episodic property keys:", ", ".join(prop_keys))
    print()

    # 4. Use 'content' for episode body (Graphiti stores episode_body there); fallback to summary/name
    body_prop = "content" if "content" in prop_keys else ("summary" if "summary" in prop_keys else "name")

    # 5. Count by content: PR-like (PR #), comments (Comments:), commits (Commit / on branch)
    pr_count = comments_count = commit_count = 0
    try:
        pr_like = run(
            f"MATCH (e:Episodic) WHERE e.{body_prop} IS NOT NULL AND e.{body_prop} CONTAINS 'PR #' RETURN count(e) AS c"
        )
        with_comments = run(
            f"MATCH (e:Episodic) WHERE e.{body_prop} IS NOT NULL AND (e.{body_prop} CONTAINS 'Comments:' OR e.{body_prop} CONTAINS 'Review comments:') RETURN count(e) AS c"
        )
        commit_like = run(
            f"MATCH (e:Episodic) WHERE e.{body_prop} IS NOT NULL AND (e.{body_prop} CONTAINS 'on branch' OR e.{body_prop} CONTAINS 'Commit ') RETURN count(e) AS c"
        )
        pr_count = pr_like[0]["c"] if pr_like else 0
        comments_count = with_comments[0]["c"] if with_comments else 0
        commit_count = commit_like[0]["c"] if commit_like else 0
        print(f"Episode body property: '{body_prop}'")
        print(f"  Episodes that look like PRs (contain 'PR #'):     {pr_count}")
        print(f"  Episodes that contain comments text:             {comments_count}")
        print(f"  Episodes that look like commits (on branch/Commit): {commit_count}")
        if comments_count == 0 and commit_count == 0 and total > 0:
            print("  -> If you expected comments/commits: backfill may have run before we added them,")
            print("     or comments/commits are in a different property. Sample content below.")
        print()
    except Exception as e:
        print(f"  (Could not query property '{body_prop}': {e})\n")

    # 6. group_id (project) distribution
    if "group_id" in prop_keys:
        groups = run(
            "MATCH (e:Episodic) RETURN e.group_id AS g, count(*) AS c ORDER BY c DESC LIMIT 10"
        )
        print("Episodes per group_id (project_id):")
        for r in groups:
            print(f"  {r['g']}: {r['c']}")
        print()

    # 7. Sample episode names
    sample_names = run(
        "MATCH (e:Episodic) RETURN e.name AS name LIMIT 5"
    )
    if sample_names:
        print("Sample episode names:")
        for r in sample_names:
            print(f"  - {r.get('name', '')}")
        print()

    # 8. If no commits/comments found in body, show a raw content snippet
    if total > 0 and (comments_count == 0 and commit_count == 0):
        snippet = run(
            "MATCH (e:Episodic) RETURN e LIMIT 1"
        )
        if snippet and snippet[0].get("e"):
            node = snippet[0]["e"]
            print("Sample Episodic node (first 500 chars of string props):")
            for k, v in node.items():
                if isinstance(v, str):
                    print(f"  {k}: {v[:500]}...")
                else:
                    print(f"  {k}: {v}")
    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()

from typing import Dict, List, Set
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)


class Neo4jStateService:
    """Service for querying Neo4j parsing state"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )

    def close(self):
        self.driver.close()

    def get_parsed_files(self, project_id: str) -> Set[str]:
        """
        Get set of file paths that have been parsed (have nodes in Neo4j).

        WARNING: This fetches ALL files for the project. For large repos (100K+ files),
        this can be slow and memory-intensive. If you only need to check specific files,
        use get_parsed_files_for_paths() instead.

        Returns:
            Set of file paths (relative to repo root)
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:NODE {repoId: $project_id})
                WHERE n.file_path IS NOT NULL AND n.file_path <> ''
                RETURN DISTINCT n.file_path AS file_path
            """, project_id=project_id)

            return {record['file_path'] for record in result}

    def get_parsed_files_for_paths(
        self,
        project_id: str,
        file_paths: List[str],
        batch_size: int = 1000
    ) -> Set[str]:
        """
        Check which of the specified file paths have been parsed.

        More efficient than get_parsed_files() when you only need to check
        a subset of files (e.g., files in a single work unit).

        IMPORTANT: Uses batching even for work unit queries to avoid overwhelming
        Neo4j with large IN clauses (5000 files would be too many).

        Args:
            project_id: Project identifier
            file_paths: List of file paths to check
            batch_size: Number of paths to check per query (default 1000)

        Returns:
            Set of file paths that exist in Neo4j (subset of file_paths)
        """
        if not file_paths:
            return set()

        parsed_files = set()
        num_batches = (len(file_paths) + batch_size - 1) // batch_size

        # Reuse a single session for all batches to avoid connection overhead
        with self.driver.session() as session:
            # Batch the queries to avoid large IN clauses
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]

                # Use parameterized query with IN clause
                result = session.run("""
                    MATCH (n:NODE {repoId: $project_id})
                    WHERE n.file_path IN $file_paths
                    RETURN DISTINCT n.file_path AS file_path
                """, project_id=project_id, file_paths=batch)

                parsed_files.update(record['file_path'] for record in result)

        logger.info(
            f"Checked {len(file_paths)} files in {num_batches} batches, "
            f"found {len(parsed_files)} already parsed"
        )

        return parsed_files

    def get_parsing_statistics(self, project_id: str) -> Dict[str, int]:
        """
        Get comprehensive statistics about parsing state.

        Returns:
            Dict with: total_nodes, total_files, nodes_with_docstring,
                      nodes_with_embedding, parsing_percentage, inference_percentage
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:NODE {repoId: $project_id})
                OPTIONAL MATCH (f:FILE {repoId: $project_id})
                WITH
                  count(DISTINCT n) AS total_nodes,
                  count(DISTINCT f) AS total_files,
                  count(DISTINCT n.file_path) AS unique_file_paths,
                  count(CASE WHEN n.text IS NOT NULL THEN 1 END) AS nodes_with_text,
                  count(CASE WHEN n.docstring IS NOT NULL THEN 1 END) AS nodes_with_docstring,
                  count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS nodes_with_embedding
                RETURN
                  total_nodes,
                  total_files,
                  unique_file_paths,
                  nodes_with_text,
                  nodes_with_docstring,
                  nodes_with_embedding,
                  round(100.0 * nodes_with_text / CASE WHEN total_nodes > 0 THEN total_nodes ELSE 1 END, 2) AS parsing_percentage,
                  round(100.0 * nodes_with_docstring / CASE WHEN total_nodes > 0 THEN total_nodes ELSE 1 END, 2) AS inference_percentage
            """, project_id=project_id)

            record = result.single()
            if not record:
                return {
                    'total_nodes': 0,
                    'total_files': 0,
                    'unique_file_paths': 0,
                    'nodes_with_text': 0,
                    'nodes_with_docstring': 0,
                    'nodes_with_embedding': 0,
                    'parsing_percentage': 0.0,
                    'inference_percentage': 0.0
                }

            return dict(record)

    def get_files_by_inference_status(
        self,
        project_id: str
    ) -> Dict[str, List[str]]:
        """
        Categorize files by their inference completion status.

        Returns:
            Dict with keys: 'complete', 'partial', 'none'
            Values: Lists of file paths
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:NODE {repoId: $project_id})
                WHERE n.file_path IS NOT NULL AND n.file_path <> ''
                WITH n.file_path AS file_path,
                     count(n) AS total_nodes,
                     count(n.docstring) AS nodes_with_inference
                RETURN file_path,
                       total_nodes,
                       nodes_with_inference,
                       CASE
                         WHEN total_nodes = nodes_with_inference THEN 'complete'
                         WHEN nodes_with_inference > 0 THEN 'partial'
                         ELSE 'none'
                       END AS status
                ORDER BY file_path
            """, project_id=project_id)

            categorized = {
                'complete': [],
                'partial': [],
                'none': []
            }

            for record in result:
                status = record['status']
                file_path = record['file_path']
                categorized[status].append(file_path)

            return categorized

    def has_any_nodes(self, project_id: str) -> bool:
        """Check if project has any nodes in Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:NODE {repoId: $project_id})
                RETURN count(n) > 0 AS has_nodes
            """, project_id=project_id)

            record = result.single()
            return record['has_nodes'] if record else False

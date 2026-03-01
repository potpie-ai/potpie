import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.search.search_service import SearchService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CodeGraphService:
    """Manage graph persistence and maintenance operations for repository parsing."""

    def __init__(
        self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, db: Session
    ) -> None:
        """Initialize a Neo4j-backed graph service.

        Args:
            neo4j_uri: Neo4j connection URI.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
            db: Active SQLAlchemy session for secondary services.

        Raises:
            Exception: Re-raises any Neo4j driver initialization failure.
        """
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
        except Exception:
            logger.exception("Failed to initialize Neo4j driver", uri=neo4j_uri)
            raise
        self.db = db

    def __enter__(self) -> "CodeGraphService":
        """Enter context manager scope for deterministic driver cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the Neo4j driver when leaving context manager scope."""
        self.close()

    def _require_driver(self):
        """Return the active Neo4j driver or raise if closed/uninitialized."""
        if self.driver is None:
            raise RuntimeError("Neo4j driver is not available")
        return self.driver

    @contextmanager
    def _session(self) -> Generator:
        """Yield a Neo4j session from the active driver.

        Yields:
            neo4j.Session: Open Neo4j session.
        """
        with self._require_driver().session() as session:
            yield session

    @staticmethod
    def generate_node_id(path: str, user_id: str) -> str:
        """Generate deterministic node identifier scoped to a user.

        Args:
            path: Source path or node signature.
            user_id: Owning user identifier.

        Returns:
            MD5 hex digest used as ``node_id``.
        """
        combined_string = f"{user_id}:{path}"

        # usedforsecurity=False: MD5 is used for non-cryptographic node ID generation only
        hash_object = hashlib.md5(usedforsecurity=False)  # noqa: S324
        hash_object.update(combined_string.encode("utf-8"))
        node_id = hash_object.hexdigest()
        return node_id

    def close(self) -> None:
        """Close the Neo4j driver safely and idempotently."""
        if self.driver is None:
            return

        try:
            self.driver.close()
        finally:
            self.driver = None

    def delete_nodes_by_file_paths(self, project_id: str, file_paths: list[str]) -> None:
        """Delete all graph nodes that belong to the provided file paths.

        Args:
            project_id: Project/repository identifier in the graph.
            file_paths: Relative file paths to remove from graph storage.
        """
        normalized_paths = [
            str(Path(file_path).as_posix()).lstrip("./")
            for file_path in file_paths
            if file_path
        ]
        if not normalized_paths:
            return

        with self._session() as session:
            session.run(
                """
                MATCH (n:NODE {repoId: $project_id})
                WHERE n.file_path IN $file_paths
                DETACH DELETE n
                """,
                project_id=project_id,
                file_paths=normalized_paths,
            )

    def create_and_store_graph(
        self,
        repo_dir: str,
        project_id: str,
        user_id: str,
        changed_files: Optional[list[str]] = None,
    ) -> None:
        """Parse repository structure and persist graph nodes/relationships in Neo4j.

        Args:
            repo_dir: Local repository directory to parse.
            project_id: Project/repository identifier.
            user_id: User identifier for node ID generation.
            changed_files: Optional subset of changed files for incremental updates.
        """
        graph_start_time = time.time()
        changed_files_set = None
        if changed_files:
            changed_files_set = {
                str(Path(file_path).as_posix()).lstrip("./")
                for file_path in changed_files
                if file_path
            }

        # Ensure repo_dir is a string and absolute path
        repo_dir = str(Path(repo_dir).resolve())
        logger.info(
            f"[GRAPH GENERATION] Starting graph creation for project {project_id}",
            project_id=project_id,
        )
        logger.info(
            f"[GRAPH GENERATION] Using repository directory: {repo_dir} (exists: {os.path.exists(repo_dir)}, isdir: {os.path.isdir(repo_dir)})",
            project_id=project_id,
        )

        # Step 1: Create RepoMap and parse repository
        repo_map_start = time.time()
        logger.info(
            f"[GRAPH GENERATION] Step 1/4: Initializing RepoMap parser",
            project_id=project_id,
        )
        self.repo_map = RepoMap(
            root=repo_dir,
            verbose=True,
            main_model=SimpleTokenCounter(),
            io=SimpleIO(),
        )

        parse_start = time.time()
        logger.info(
            f"[GRAPH GENERATION] Step 2/4: Parsing repository structure",
            project_id=project_id,
        )
        nx_graph = self.repo_map.create_graph(repo_dir)
        parse_time = time.time() - parse_start
        total_node_count = nx_graph.number_of_nodes()
        total_relationship_count = nx_graph.number_of_edges()
        node_count = total_node_count
        relationship_count = total_relationship_count
        if changed_files_set:
            node_count = sum(
                1
                for _, node_data in nx_graph.nodes(data=True)
                if node_data.get("file") in changed_files_set
            )
            relationship_count = sum(
                1
                for source, target, _ in nx_graph.edges(data=True)
                if (
                    nx_graph.nodes[source].get("file") in changed_files_set
                    or nx_graph.nodes[target].get("file") in changed_files_set
                )
            )
        logger.info(
            f"[GRAPH GENERATION] Parsed repository: {total_node_count} nodes, {total_relationship_count} relationships in {parse_time:.2f}s",
            project_id=project_id,
            node_count=total_node_count,
            relationship_count=total_relationship_count,
            parse_time_seconds=parse_time,
        )
        if changed_files_set:
            logger.info(
                "[GRAPH GENERATION] Incremental mode",
                project_id=project_id,
                changed_files_count=len(changed_files_set),
                incremental_node_count=node_count,
                incremental_relationship_count=relationship_count,
            )

        with self._session() as session:
            db_start_time = time.time()

            # Step 2: Create indices
            index_start = time.time()
            logger.info(
                f"[GRAPH GENERATION] Step 3/4: Creating Neo4j indices",
                project_id=project_id,
            )
            session.run(
                """
                CREATE INDEX node_id_repo_idx IF NOT EXISTS
                FOR (n:NODE) ON (n.node_id, n.repoId)
            """
            )
            index_time = time.time() - index_start
            logger.info(
                f"[GRAPH GENERATION] Created indices in {index_time:.2f}s",
                project_id=project_id,
                index_time_seconds=index_time,
            )

            # Step 3: Batch insert nodes
            node_insert_start = time.time()
            logger.info(
                f"[GRAPH GENERATION] Step 4/4: Inserting {node_count} nodes into Neo4j",
                project_id=project_id,
                total_nodes=node_count,
            )
            batch_size = 1000
            total_batches = (node_count + batch_size - 1) // batch_size
            nodes_inserted = 0

            for batch_idx, i in enumerate(range(0, node_count, batch_size), 1):
                batch_start = time.time()
                batch_nodes = list(nx_graph.nodes(data=True))[i : i + batch_size]
                nodes_to_create = []

                for node_id, node_data in batch_nodes:
                    if changed_files_set and node_data.get("file") not in changed_files_set:
                        continue

                    # Get the node type and ensure it's one of our expected types
                    node_type = node_data.get("type", "UNKNOWN")
                    if node_type == "UNKNOWN":
                        continue
                    # Initialize labels with NODE
                    labels = ["NODE"]

                    # Add specific type label if it's a valid type
                    if node_type in ["FILE", "CLASS", "FUNCTION", "INTERFACE"]:
                        labels.append(node_type)

                    # Prepare node data
                    processed_node = {
                        "name": node_data.get(
                            "name", node_id
                        ),  # Use node_id as fallback
                        "file_path": node_data.get("file", ""),
                        "start_line": node_data.get("line", -1),
                        "end_line": node_data.get("end_line", -1),
                        "repoId": project_id,
                        "node_id": CodeGraphService.generate_node_id(node_id, user_id),
                        "entityId": user_id,
                        "type": node_type,
                        "text": node_data.get("text", ""),
                        "labels": labels,
                    }

                    # Remove None values
                    processed_node = {
                        k: v for k, v in processed_node.items() if v is not None
                    }
                    nodes_to_create.append(processed_node)

                # Create nodes with labels
                insert_start = time.time()
                session.run(
                    """
                    UNWIND $nodes AS node
                    CALL apoc.create.node(node.labels, node) YIELD node AS n
                    RETURN count(*) AS created_count
                    """,
                    nodes=nodes_to_create,
                )
                insert_time = time.time() - insert_start
                nodes_inserted += len(nodes_to_create)
                batch_time = time.time() - batch_start

                if batch_idx % 10 == 0 or batch_idx == total_batches:
                    logger.info(
                        f"[GRAPH GENERATION] Node batch {batch_idx}/{total_batches}: inserted {len(nodes_to_create)} nodes in {batch_time:.2f}s (insert: {insert_time:.2f}s)",
                        project_id=project_id,
                        batch_index=batch_idx,
                        total_batches=total_batches,
                        batch_size=len(nodes_to_create),
                        batch_time_seconds=batch_time,
                        insert_time_seconds=insert_time,
                    )

            node_insert_time = time.time() - node_insert_start
            logger.info(
                f"[GRAPH GENERATION] Completed node insertion: {nodes_inserted} nodes in {node_insert_time:.2f}s",
                project_id=project_id,
                nodes_inserted=nodes_inserted,
                node_insert_time_seconds=node_insert_time,
            )

            # Step 4: Insert relationships
            rel_insert_start = time.time()
            logger.info(
                f"[GRAPH GENERATION] Inserting {relationship_count} relationships into Neo4j",
                project_id=project_id,
                total_relationships=relationship_count,
            )

            # Pre-calculate common relationship types to avoid dynamic relationship creation
            rel_types = set()
            for source, target, data in nx_graph.edges(data=True):
                if changed_files_set:
                    source_file = nx_graph.nodes[source].get("file")
                    target_file = nx_graph.nodes[target].get("file")
                    if (
                        source_file not in changed_files_set
                        and target_file not in changed_files_set
                    ):
                        continue
                rel_type = data.get("type", "REFERENCES")
                rel_types.add(rel_type)

            logger.info(
                f"[GRAPH GENERATION] Found {len(rel_types)} relationship types: {', '.join(sorted(rel_types))}",
                project_id=project_id,
                relationship_types=list(rel_types),
            )

            # Process relationships with huge batch size and type-specific queries
            batch_size = 1000
            total_rels_inserted = 0

            for rel_type in rel_types:
                rel_type_start = time.time()
                # Filter edges by relationship type
                type_edges = [
                    (s, t, d)
                    for s, t, d in nx_graph.edges(data=True)
                    if d.get("type", "REFERENCES") == rel_type
                    and (
                        not changed_files_set
                        or (
                            nx_graph.nodes[s].get("file") in changed_files_set
                            or nx_graph.nodes[t].get("file") in changed_files_set
                        )
                    )
                ]

                type_batch_count = (len(type_edges) + batch_size - 1) // batch_size
                logger.info(
                    f"[GRAPH GENERATION] Processing {len(type_edges)} {rel_type} relationships in {type_batch_count} batches",
                    project_id=project_id,
                    relationship_type=rel_type,
                    relationship_count=len(type_edges),
                    batch_count=type_batch_count,
                )

                for batch_idx, i in enumerate(range(0, len(type_edges), batch_size), 1):
                    batch_start = time.time()
                    batch_edges = type_edges[i : i + batch_size]
                    edges_to_create = []

                    for source, target, data in batch_edges:
                        edges_to_create.append(
                            {
                                "source_id": CodeGraphService.generate_node_id(
                                    source, user_id
                                ),
                                "target_id": CodeGraphService.generate_node_id(
                                    target, user_id
                                ),
                                "repoId": project_id,
                            }
                        )

                    # Type-specific relationship creation in one transaction
                    insert_start = time.time()
                    query = f"""
                        UNWIND $edges AS edge
                        MATCH (source:NODE {{node_id: edge.source_id, repoId: edge.repoId}})
                        MATCH (target:NODE {{node_id: edge.target_id, repoId: edge.repoId}})
                        CREATE (source)-[r:{rel_type} {{repoId: edge.repoId}}]->(target)
                    """
                    session.run(query, edges=edges_to_create)
                    insert_time = time.time() - insert_start
                    batch_time = time.time() - batch_start
                    total_rels_inserted += len(edges_to_create)

                    if batch_idx % 10 == 0 or batch_idx == type_batch_count:
                        logger.info(
                            f"[GRAPH GENERATION] {rel_type} batch {batch_idx}/{type_batch_count}: inserted {len(edges_to_create)} relationships in {batch_time:.2f}s (insert: {insert_time:.2f}s)",
                            project_id=project_id,
                            relationship_type=rel_type,
                            batch_index=batch_idx,
                            total_batches=type_batch_count,
                            batch_size=len(edges_to_create),
                            batch_time_seconds=batch_time,
                            insert_time_seconds=insert_time,
                        )

                rel_type_time = time.time() - rel_type_start
                logger.info(
                    f"[GRAPH GENERATION] Completed {rel_type} relationships: {len(type_edges)} in {rel_type_time:.2f}s",
                    project_id=project_id,
                    relationship_type=rel_type,
                    relationship_count=len(type_edges),
                    rel_type_time_seconds=rel_type_time,
                )

            rel_insert_time = time.time() - rel_insert_start
            logger.info(
                f"[GRAPH GENERATION] Completed relationship insertion: {total_rels_inserted} relationships in {rel_insert_time:.2f}s",
                project_id=project_id,
                relationships_inserted=total_rels_inserted,
                rel_insert_time_seconds=rel_insert_time,
            )

            db_time = time.time() - db_start_time
            total_time = time.time() - graph_start_time

            logger.info(
                f"[GRAPH GENERATION] Graph creation completed in {total_time:.2f}s (DB operations: {db_time:.2f}s, Parsing: {parse_time:.2f}s)",
                project_id=project_id,
                total_time_seconds=total_time,
                db_time_seconds=db_time,
                parse_time_seconds=parse_time,
                nodes_inserted=nodes_inserted,
                relationships_inserted=total_rels_inserted,
            )

    def cleanup_graph(self, project_id: str) -> None:
        """Remove all graph data and search index state for a project.

        Args:
            project_id: Project/repository identifier.
        """
        with self._session() as session:
            session.run(
                """
                MATCH (n {repoId: $project_id})
                DETACH DELETE n
                """,
                project_id=project_id,
            )

        # Clean up search index
        search_service = SearchService(self.db)
        search_service.delete_project_index(project_id)

    async def get_node_by_id(self, node_id: str, project_id: str) -> Optional[Dict]:
        """Fetch a single node by logical node ID and project identifier.

        Args:
            node_id: Graph node ID.
            project_id: Project/repository identifier.

        Returns:
            Node property dictionary if found, otherwise ``None``.
        """
        with self._session() as session:
            result = session.run(
                """
                MATCH (n:NODE {node_id: $node_id, repoId: $project_id})
                RETURN n
                """,
                node_id=node_id,
                project_id=project_id,
            )
            record = result.single()
            return dict(record["n"]) if record else None

    def query_graph(self, query: str) -> list[Dict[str, Any]]:
        """Execute an arbitrary Cypher query and return record dictionaries.

        Args:
            query: Cypher query string.

        Returns:
            List of row dictionaries from query results.
        """
        with self._session() as session:
            result = session.run(query)
            return [record.data() for record in result]


class SimpleIO:
    """Minimal file IO adapter consumed by ``RepoMap`` parsing routines."""

    def read_text(self, fname):
        """
        Read file with multiple encoding fallbacks.

        Tries encodings in order:
        1. utf-8 (most common)
        2. utf-8-sig (UTF-8 with BOM)
        3. utf-16 (common in Windows files)
        4. latin-1 (fallback that accepts all bytes)
        """
        encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]

        for encoding in encodings:
            try:
                with open(fname, "r", encoding=encoding) as f:
                    content = f.read()
                    if encoding != "utf-8":
                        logger.info(f"Read {fname} using {encoding} encoding")
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                logger.exception(f"Error reading {fname}")
                return ""

        logger.warning(
            f"Could not read {fname} with any supported encoding. Skipping this file."
        )
        return ""

    def tool_error(self, message):
        """Emit parser tool error message through the shared logger."""
        logger.error(f"Error: {message}")

    def tool_output(self, message):
        """Emit parser tool output message through the shared logger."""
        logger.info(message)


class SimpleTokenCounter:
    """Approximate token counter used by RepoMap parsing."""

    def token_count(self, text):
        """Return a whitespace-token approximation for the provided text."""
        return len(text.split())

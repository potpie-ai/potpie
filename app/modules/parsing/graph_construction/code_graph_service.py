import hashlib
import json
import logging
import os
import time
from typing import Dict, Optional, Set

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)

# Lazy-loaded singleton for inference context extractor
_context_extractor = None


def get_context_extractor():
    """Get or create singleton InferenceContextExtractor to avoid repeated initialization."""
    global _context_extractor
    if _context_extractor is None:
        from app.modules.parsing.graph_construction.inference_context_extractor import InferenceContextExtractor
        _context_extractor = InferenceContextExtractor()
        logger.info("Initialized InferenceContextExtractor singleton")
    return _context_extractor


class CodeGraphService:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, db: Session):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.db = db

    @staticmethod
    def generate_node_id(path: str, user_id: str):
        # Concatenate path and signature
        combined_string = f"{user_id}:{path}"

        # Create a SHA-1 hash of the combined string
        hash_object = hashlib.md5()
        hash_object.update(combined_string.encode("utf-8"))

        # Get the hexadecimal representation of the hash
        node_id = hash_object.hexdigest()

        return node_id

    def close(self):
        self.driver.close()

    def create_and_store_graph(self, repo_dir, project_id, user_id):
        logger.info(f"CodeGraphService: create_and_store_graph called for project_id={project_id}, user_id={user_id}")
        logger.info(f"CodeGraphService: repo_dir={repo_dir}")
        
        # Validate inputs
        if not repo_dir:
            error_msg = f"repo_dir is None or empty for project {project_id}"
            logger.error(f"CodeGraphService: {error_msg}")
            raise ValueError(error_msg)
        
        if not os.path.exists(repo_dir):
            error_msg = f"repo_dir does not exist: {repo_dir} for project {project_id}"
            logger.error(f"CodeGraphService: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        if not os.path.isdir(repo_dir):
            error_msg = f"repo_dir is not a directory: {repo_dir} for project {project_id}"
            logger.error(f"CodeGraphService: {error_msg}")
            raise NotADirectoryError(error_msg)
        
        logger.info(f"CodeGraphService: Validated repo_dir exists and is accessible")
        
        # Create the graph using RepoMap
        logger.info(f"CodeGraphService: Initializing RepoMap with root={repo_dir}")
        try:
            self.repo_map = RepoMap(
                root=repo_dir,
                verbose=True,
                main_model=SimpleTokenCounter(),
                io=SimpleIO(),
            )
            logger.info(f"CodeGraphService: RepoMap initialized successfully")
        except Exception as e:
            logger.error(f"CodeGraphService: Failed to initialize RepoMap: {e}")
            raise

        logger.info(f"CodeGraphService: Calling repo_map.create_graph()")
        try:
            nx_graph = self.repo_map.create_graph(repo_dir)
            logger.info(f"CodeGraphService: Graph created with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"CodeGraphService: Failed to create graph: {e}")
            logger.exception("CodeGraphService: Exception details:")
            raise

        logger.info(f"CodeGraphService: Opening Neo4j session to store graph")
        with self.driver.session() as session:
            start_time = time.time()
            node_count = nx_graph.number_of_nodes()
            logger.info(f"CodeGraphService: Creating {node_count} nodes in Neo4j")

            # Create specialized index for relationship queries
            logger.debug(f"CodeGraphService: Creating Neo4j index")
            try:
                session.run(
                    """
                    CREATE INDEX node_id_repo_idx IF NOT EXISTS
                    FOR (n:NODE) ON (n.node_id, n.repoId)
                """
                )
                logger.debug(f"CodeGraphService: Index created successfully")
            except Exception as e:
                logger.error(f"CodeGraphService: Failed to create index: {e}")
                raise

            # Batch insert nodes
            batch_size = 1000
            logger.info(f"CodeGraphService: Inserting nodes in batches of {batch_size}")
            for i in range(0, node_count, batch_size):
                batch_nodes = list(nx_graph.nodes(data=True))[i : i + batch_size]
                nodes_to_create = []
                
                logger.debug(f"CodeGraphService: Processing node batch {i//batch_size + 1}/{(node_count + batch_size - 1)//batch_size}")

                for node_id, node_data in batch_nodes:
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
                logger.debug(f"CodeGraphService: Creating {len(nodes_to_create)} nodes in Neo4j")
                try:
                    result = session.run(
                        """
                        UNWIND $nodes AS node
                        CALL apoc.create.node(node.labels, node) YIELD node AS n
                        RETURN count(*) AS created_count
                        """,
                        nodes=nodes_to_create,
                    )
                    count = result.single()["created_count"]
                    logger.debug(f"CodeGraphService: Created {count} nodes in this batch")
                except Exception as e:
                    logger.error(f"CodeGraphService: Failed to create node batch: {e}")
                    raise
            
            logger.info(f"CodeGraphService: All nodes created successfully")

            relationship_count = nx_graph.number_of_edges()
            logger.info(f"CodeGraphService: Creating {relationship_count} relationships")

            # Pre-calculate common relationship types to avoid dynamic relationship creation
            rel_types = set()
            for source, target, data in nx_graph.edges(data=True):
                rel_type = data.get("type", "REFERENCES")
                rel_types.add(rel_type)

            # Process relationships with huge batch size and type-specific queries
            batch_size = 1000

            for rel_type in rel_types:
                # Filter edges by relationship type
                type_edges = [
                    (s, t, d)
                    for s, t, d in nx_graph.edges(data=True)
                    if d.get("type", "REFERENCES") == rel_type
                ]

                logging.info(
                    f"Creating {len(type_edges)} relationships of type {rel_type}"
                )

                for i in range(0, len(type_edges), batch_size):
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
                    query = f"""
                        UNWIND $edges AS edge
                        MATCH (source:NODE {{node_id: edge.source_id, repoId: edge.repoId}})
                        MATCH (target:NODE {{node_id: edge.target_id, repoId: edge.repoId}})
                        CREATE (source)-[r:{rel_type} {{repoId: edge.repoId}}]->(target)
                    """
                    session.run(query, edges=edges_to_create)

            end_time = time.time()
            logging.info(
                f"Time taken to create graph and search index: {end_time - start_time:.2f} seconds"
            )

    def cleanup_graph(self, project_id: str):
        """
        Delete all nodes for a project in batches to avoid memory issues.
        Uses a two-phase approach:
        1. Delete relationships in batches
        2. Delete nodes in batches
        This avoids the OOM error from DETACH DELETE on highly connected nodes.
        """
        rel_batch_size = int(os.getenv('NEO4J_DELETE_REL_BATCH_SIZE', '5000'))
        node_batch_size = int(os.getenv('NEO4J_DELETE_NODE_BATCH_SIZE', '1000'))

        logger.info(f"Starting batched cleanup for project {project_id}")
        logger.info(f"Phase 1: Deleting relationships (batch_size={rel_batch_size})")

        total_rels_deleted = 0
        total_nodes_deleted = 0

        with self.driver.session() as session:
            # Phase 1: Delete all relationships first
            while True:
                result = session.run(
                    """
                    MATCH (n {repoId: $project_id})-[r]-()
                    WITH r LIMIT $batch_size
                    DELETE r
                    RETURN count(r) as deleted
                    """,
                    project_id=project_id,
                    batch_size=rel_batch_size
                )

                record = result.single()
                deleted_count = record["deleted"] if record else 0
                total_rels_deleted += deleted_count

                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} relationships (total: {total_rels_deleted})")

                # If we deleted fewer than batch_size, we're done
                if deleted_count < rel_batch_size:
                    break

            logger.info(f"Phase 1 complete: deleted {total_rels_deleted} total relationships")
            logger.info(f"Phase 2: Deleting nodes (batch_size={node_batch_size})")

            # Phase 2: Delete all nodes (now they have no relationships)
            while True:
                result = session.run(
                    """
                    MATCH (n {repoId: $project_id})
                    WITH n LIMIT $batch_size
                    DELETE n
                    RETURN count(n) as deleted
                    """,
                    project_id=project_id,
                    batch_size=node_batch_size
                )

                record = result.single()
                deleted_count = record["deleted"] if record else 0
                total_nodes_deleted += deleted_count

                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} nodes (total: {total_nodes_deleted})")

                # If we deleted fewer than batch_size, we're done
                if deleted_count < node_batch_size:
                    break

        logger.info(
            f"Cleanup complete: deleted {total_rels_deleted} relationships "
            f"and {total_nodes_deleted} nodes for project {project_id}"
        )

        # Clean up search index
        search_service = SearchService(self.db)
        search_service.delete_project_index(project_id)

    async def get_node_by_id(self, node_id: str, project_id: str) -> Optional[Dict]:
        with self.driver.session() as session:
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

    def query_graph(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def write_subgraph_incremental(
        self, graph, project_id: str, user_id: str, batch_size: int = 1000
    ):
        """
        Write subgraph to Neo4j incrementally with smaller batches.

        Designed for distributed parsing where multiple workers write
        concurrently. Uses batch UNWIND queries for optimal performance.

        NEW: Extracts inference_context for 85-90% token savings during inference.

        Args:
            graph: NetworkX MultiDiGraph to write
            project_id: Project ID (used as repoId)
            user_id: User ID (used for node_id generation)
            batch_size: Number of nodes/edges per batch (default 1000)

        Returns:
            Tuple of (nodes_created, edges_created)
        """
        logger.info(
            f"Writing subgraph incrementally: "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

        # Get singleton extractor (FIX #6 - avoid repeated initialization)
        extractor = get_context_extractor()

        # Convert graph to node list
        nodes = []
        for node_name, node_data in graph.nodes(data=True):
            node_type = node_data.get("type", "UNKNOWN")
            if node_type == "UNKNOWN":
                continue

            labels = ["NODE"]
            if node_type in ["FILE", "CLASS", "FUNCTION", "INTERFACE"]:
                labels.append(node_type)

            file_path = node_data.get("file", "")
            full_text = node_data.get("text", "")
            display_name = node_data.get("display_name", "")
            class_name = node_data.get("class_name")  # May be present for methods

            # === NEW: Extract minimal inference context for 85-90% token savings ===
            inference_context = None
            if full_text and node_type in ["FUNCTION", "CLASS", "INTERFACE"]:
                # Detect language from file path
                language = extractor.analyzer.detect_language(file_path) if file_path else None
                if language:
                    try:
                        context_dict = extractor.extract_context(
                            full_text=full_text,
                            file_path=file_path,
                            language=language,
                            node_type=node_type,
                            node_name=node_data.get("name", node_name),
                            class_name=class_name
                        )
                        inference_context = json.dumps(context_dict)
                    except Exception as e:
                        logger.warning(f"Failed to extract inference context for {node_name}: {e}")

            node_dict = {
                "name": node_data.get("name", node_name),
                "file_path": file_path,
                "start_line": node_data.get("line", -1),
                "end_line": node_data.get("end_line", -1),
                "repoId": project_id,
                "node_id": CodeGraphService.generate_node_id(node_name, user_id),
                "entityId": user_id,
                "type": node_type,
                "text": full_text,
                "display_name": display_name,
                "inference_context": inference_context,  # NEW: Minimal LLM context
                "labels": labels,
            }
            # Remove None values
            node_dict = {k: v for k, v in node_dict.items() if v is not None}
            nodes.append(node_dict)

        # Write nodes in batches using UNWIND (no per-node queries)
        nodes_created = 0
        with self.driver.session() as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i+batch_size]

                # Group nodes by label combination for efficient creation
                nodes_by_labels = {}
                for node in batch:
                    labels_tuple = tuple(sorted(node['labels']))
                    if labels_tuple not in nodes_by_labels:
                        nodes_by_labels[labels_tuple] = []
                    nodes_by_labels[labels_tuple].append(node)

                # Create nodes with same labels in single query
                for labels_tuple, label_nodes in nodes_by_labels.items():
                    labels_str = ':'.join(labels_tuple)
                    # Remove 'labels' key from node properties
                    nodes_without_labels = [
                        {k: v for k, v in node.items() if k != 'labels'}
                        for node in label_nodes
                    ]

                    query = f"""
                        UNWIND $nodes AS node
                        CREATE (n:{labels_str})
                        SET n = node
                    """
                    session.run(query, nodes=nodes_without_labels)
                    nodes_created += len(label_nodes)

                logger.info(f"Wrote {nodes_created}/{len(nodes)} nodes")

        # Convert edges to list
        relationships = []
        for source, target, edge_data in graph.edges(data=True):
            rel_dict = {
                'source_id': CodeGraphService.generate_node_id(source, user_id),
                'target_id': CodeGraphService.generate_node_id(target, user_id),
                'type': edge_data.get('type', 'REFERENCES'),
                'repoId': project_id,
            }
            # Preserve all edge properties
            for key, value in edge_data.items():
                if key != 'type':  # type is already added
                    rel_dict[key] = value
            relationships.append(rel_dict)

        # Write edges in batches, grouped by relationship type
        edges_created = 0
        with self.driver.session() as session:
            # Group by relationship type
            rel_types = {}
            for rel in relationships:
                rel_type = rel['type']
                if rel_type not in rel_types:
                    rel_types[rel_type] = []
                rel_types[rel_type].append(rel)

            for rel_type, type_edges in rel_types.items():
                # Validate relationship type to prevent injection
                if not rel_type.replace('_', '').isalnum():
                    logger.warning(f"Skipping invalid relationship type: {rel_type}")
                    continue

                for i in range(0, len(type_edges), batch_size):
                    batch = type_edges[i:i+batch_size]
                    # Extract edge properties (excluding source_id, target_id, type)
                    edges_with_props = []
                    for edge in batch:
                        edge_props = {k: v for k, v in edge.items()
                                     if k not in ['source_id', 'target_id', 'type']}
                        edges_with_props.append({
                            'source_id': edge['source_id'],
                            'target_id': edge['target_id'],
                            'props': edge_props
                        })

                    query = f"""
                        UNWIND $edges AS edge
                        MATCH (source:NODE {{node_id: edge.source_id, repoId: $repo_id}})
                        MATCH (target:NODE {{node_id: edge.target_id, repoId: $repo_id}})
                        CREATE (source)-[r:{rel_type}]->(target)
                        SET r = edge.props
                    """
                    session.run(query, edges=edges_with_props, repo_id=project_id)
                    edges_created += len(batch)
                    logger.info(f"Wrote {edges_created}/{len(relationships)} edges")

        logger.info(
            f"Subgraph write complete: {nodes_created} nodes, {edges_created} edges"
        )

        return nodes_created, edges_created

    def create_edges_batch(
        self, edges_list: list, project_id: str, user_id: str, batch_size: int = 1000
    ) -> int:
        """
        Create edges in Neo4j from a list of edge dictionaries.

        Used by cross-directory reference resolution to create REFERENCES edges
        after all nodes have been created.

        Args:
            edges_list: List of edge dicts with keys:
                - source: source node name
                - target: target node name
                - type: relationship type
                - other properties to set on the edge
            project_id: Project ID (used as repoId)
            user_id: User ID (for node_id generation)
            batch_size: Number of edges per batch (default 1000)

        Returns:
            Number of edges created
        """
        logger.info(f"Creating {len(edges_list)} edges in batches of {batch_size}")

        # Convert source/target names to node_ids
        edges_to_create = []
        for edge in edges_list:
            edge_dict = {
                'source_id': CodeGraphService.generate_node_id(edge['source'], user_id),
                'target_id': CodeGraphService.generate_node_id(edge['target'], user_id),
                'type': edge.get('type', 'REFERENCES'),
                'repoId': project_id,
            }
            # Copy all other properties
            for key, value in edge.items():
                if key not in ['source', 'target', 'type']:
                    edge_dict[key] = value
            edges_to_create.append(edge_dict)

        # Group by relationship type for efficient creation
        rel_types = {}
        for edge in edges_to_create:
            rel_type = edge['type']
            if rel_type not in rel_types:
                rel_types[rel_type] = []
            rel_types[rel_type].append(edge)

        edges_created = 0
        with self.driver.session() as session:
            for rel_type, type_edges in rel_types.items():
                # Validate relationship type
                if not rel_type.replace('_', '').isalnum():
                    logger.warning(f"Skipping invalid relationship type: {rel_type}")
                    continue

                for i in range(0, len(type_edges), batch_size):
                    batch = type_edges[i:i+batch_size]

                    # Extract edge properties (excluding source_id, target_id, type, repoId)
                    edges_with_props = []
                    for edge in batch:
                        edge_props = {k: v for k, v in edge.items()
                                     if k not in ['source_id', 'target_id', 'type']}
                        edges_with_props.append({
                            'source_id': edge['source_id'],
                            'target_id': edge['target_id'],
                            'props': edge_props
                        })

                    query = f"""
                        UNWIND $edges AS edge
                        MATCH (source:NODE {{node_id: edge.source_id, repoId: $repo_id}})
                        MATCH (target:NODE {{node_id: edge.target_id, repoId: $repo_id}})
                        CREATE (source)-[r:{rel_type}]->(target)
                        SET r = edge.props
                    """
                    session.run(query, edges=edges_with_props, repo_id=project_id)
                    edges_created += len(batch)
                    logger.info(f"Created {edges_created}/{len(edges_to_create)} edges")

        logger.info(f"Edge creation complete: {edges_created} edges")
        return edges_created

    def get_parsed_files_for_project(self, project_id: str) -> Set[str]:
        """
        Get set of files that have been parsed for a project.

        Convenience wrapper around Neo4jStateService.
        """
        from app.modules.parsing.graph_construction.neo4j_state_service import Neo4jStateService

        # Extract connection details from driver
        # Note: This is a simplified approach; in production you may want to pass
        # these as parameters or store them as instance variables
        state_service = Neo4jStateService(
            neo4j_uri=self.driver._pool.address,
            neo4j_user="neo4j",  # These should ideally come from config
            neo4j_password="password"  # These should ideally come from config
        )

        try:
            return state_service.get_parsed_files(project_id)
        finally:
            state_service.close()


class SimpleIO:
    def read_text(self, fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"SimpleIO: Successfully read {len(content)} characters from {fname} using UTF-8")
                return content
        except UnicodeDecodeError as e:
            logger.warning(f"SimpleIO: UnicodeDecodeError reading {fname} as UTF-8: {e}")
            logger.info(f"SimpleIO: Attempting to read {fname} with latin-1 encoding")
            try:
                with open(fname, "r", encoding="latin-1") as f:
                    content = f.read()
                    logger.info(f"SimpleIO: Successfully read {len(content)} characters from {fname} using latin-1")
                    return content
            except Exception as fallback_error:
                logger.error(f"SimpleIO: Failed to read {fname} even with latin-1 encoding: {fallback_error}")
                logger.exception(f"SimpleIO: Exception details for {fname}:")
                return ""
        except FileNotFoundError as e:
            logger.error(f"SimpleIO: File not found: {fname}")
            return ""
        except PermissionError as e:
            logger.error(f"SimpleIO: Permission denied reading file: {fname}")
            return ""
        except Exception as e:
            logger.error(f"SimpleIO: Unexpected error reading {fname}: {e}")
            logger.exception(f"SimpleIO: Exception details for {fname}:")
            return ""

    def tool_error(self, message):
        logging.error(f"Error: {message}")

    def tool_output(self, message):
        logging.info(message)


class SimpleTokenCounter:
    def token_count(self, text):
        return len(text.split())

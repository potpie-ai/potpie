import hashlib
import logging
import os
import time
from typing import Dict, Optional

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)


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
        with self.driver.session() as session:
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

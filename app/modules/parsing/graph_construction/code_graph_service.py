import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple

from neo4j import GraphDatabase, Session
from sqlalchemy.orm import Session as SQLSession

from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)

class CacheMetrics:
    def __init__(self):
        self.total_nodes = 0
        self.cached_nodes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.partial_hits = 0  # Has some cached data but not all
        
    def log_metrics(self, project_id: str):
        cache_hit_rate = (self.cached_nodes / self.total_nodes) * 100 if self.total_nodes > 0 else 0
        partial_hit_rate = (self.partial_hits / self.total_nodes) * 100 if self.total_nodes > 0 else 0
        
        logger.info(
            f"Cache metrics for project {project_id}:\n"
            f"Total nodes: {self.total_nodes}\n"
            f"Fully cached nodes: {self.cached_nodes}\n"
            f"Partial cache hits: {self.partial_hits}\n"
            f"Cache hit rate: {cache_hit_rate:.2f}%\n"
            f"Partial hit rate: {partial_hit_rate:.2f}%"
        )

class CodeGraphService:
    HASH_VERSION = "v1"  # For future schema changes
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, db: SQLSession):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.db = db
        self.cache_metrics = CacheMetrics()

    @staticmethod
    def generate_node_id(path: str, user_id: str) -> str:
        # Concatenate path and signature
        combined_string = f"{user_id}:{path}"

        # Create a SHA-1 hash of the combined string
        hash_object = hashlib.md5()
        hash_object.update(combined_string.encode("utf-8"))

        # Get the hexadecimal representation of the hash
        node_id = hash_object.hexdigest()

        return node_id

    @staticmethod
    def validate_content_hash(content_hash: str) -> bool:
        """Validate the structure and format of a content hash.
        
        Args:
            content_hash (str): The content hash to validate
            
        Returns:
            bool: True if the hash is valid, False otherwise
        """
        if not content_hash or not isinstance(content_hash, str):
            return False
            
        # Check version prefix
        if not content_hash.startswith(f"{CodeGraphService.HASH_VERSION}:"):
            return False
            
        # Check hash structure
        try:
            version, digest = content_hash.split(":", 1)
            # SHA-256 produces 64 hex characters
            if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest.lower()):
                return False
        except ValueError:
            return False
            
        return True

    @staticmethod
    def generate_content_hash(node_data: Dict) -> str:
        """Generate a deterministic hash of node content.
        
        Args:
            node_data (Dict): Node data including text, name, type, etc.
            
        Returns:
            str: Version-prefixed SHA-256 hash of node content
        
        Raises:
            ValueError: If node_data is None or empty
        """
        if not node_data:
            raise ValueError("node_data cannot be None or empty")
            
        # Sort keys for consistent ordering
        content_parts = []
        for key in sorted(['text', 'name', 'type', 'file', 'line', 'end_line']):
            value = node_data.get(key)
            if value is not None:
                # Normalize string representation
                str_value = str(value).strip()
                content_parts.append(str_value)
            else:
                content_parts.append('')
                
        content_string = '|'.join(content_parts)
        if not content_string:
            raise ValueError("No valid content to hash")
            
        hash_object = hashlib.sha256()
        hash_object.update(content_string.encode("utf-8"))
        digest = hash_object.hexdigest()
        
        return f"{CodeGraphService.HASH_VERSION}:{digest}"

    def validate_cached_data(self, cached_data: Dict) -> Tuple[bool, bool]:
        """Validate cached node data.
        
        Args:
            cached_data (Dict): Node data from cache
            
        Returns:
            Tuple[bool, bool]: (has_valid_docstring, has_valid_embedding)
        """
        # Validate content hash first
        if not self.validate_content_hash(cached_data.get("hash", "")):
            return False, False
            
        has_valid_docstring = bool(cached_data.get("docstring"))
        
        embedding = cached_data.get("embedding")
        has_valid_embedding = (
            isinstance(embedding, list) and 
            len(embedding) == 384 and 
            all(isinstance(x, (int, float)) for x in embedding)
        )
        
        return has_valid_docstring, has_valid_embedding

    def get_existing_node_hashes(self, session: Session, project_id: str) -> Dict:
        """Fetch and validate existing node hashes and their cached data.
        
        Args:
            session (Session): Neo4j session
            project_id (str): Project ID to fetch nodes for
            
        Returns:
            Dict: Mapping of node_id to cached data
        """
        try:
            result = session.run(
                """
                MATCH (n:NODE {repoId: $project_id})
                RETURN n.node_id, n.content_hash, n.docstring, n.embedding, n.tags
                """,
                project_id=project_id
            )
            
            existing_hashes = {}
            invalid_hashes = 0
            for record in result:
                if record["n.content_hash"]:
                    node_id = record["n.node_id"]
                    content_hash = record["n.content_hash"]
                    
                    # Validate hash structure first
                    if not self.validate_content_hash(content_hash):
                        invalid_hashes += 1
                        continue
                        
                    cached_data = {
                        "hash": content_hash,
                        "docstring": record["n.docstring"],
                        "embedding": record["n.embedding"],
                        "tags": record["n.tags"]
                    }
                    
                    # Validate cached data
                    has_docstring, has_embedding = self.validate_cached_data(cached_data)
                    
                    if has_docstring and has_embedding:
                        self.cache_metrics.cache_hits += 1
                        existing_hashes[node_id] = cached_data
                    elif has_docstring or has_embedding:
                        self.cache_metrics.partial_hits += 1
                        # Still cache partial data
                        existing_hashes[node_id] = cached_data
                    else:
                        self.cache_metrics.cache_misses += 1
            
            if invalid_hashes > 0:
                logger.warning(f"Found {invalid_hashes} nodes with invalid content hashes")
                        
            return existing_hashes
            
        except Exception as e:
            logger.error(f"Error fetching existing hashes: {str(e)}")
            return {}

    def create_content_hash_index(self, session: Session):
        """Create and verify content hash index."""
        try:
            # Create composite index for better performance
            content_hash_query = """
                CREATE INDEX content_hash_repo_NODE IF NOT EXISTS 
                FOR (n:NODE) ON (n.repoId, n.content_hash)
            """
            session.run(content_hash_query)
            logger.info("Successfully created content hash index")
            
            # Log index statistics
            stats_query = """
                CALL db.indexes() 
                YIELD name, type, labelsOrTypes, properties, state 
                WHERE name = 'content_hash_repo_NODE'
                RETURN *
            """
            stats = session.run(stats_query).single()
            if stats:
                logger.info(f"Index stats: {dict(stats)}")
            else:
                logger.warning("Content hash index not found after creation")
                
        except Exception as e:
            logger.error(f"Error creating content hash index: {str(e)}")

    def close(self):
        self.driver.close()

    def create_and_store_graph(self, repo_dir, project_id, user_id):
        # Create the graph using RepoMap
        self.repo_map = RepoMap(
            root=repo_dir,
            verbose=True,
            main_model=SimpleTokenCounter(),
            io=SimpleIO(),
        )

        nx_graph = self.repo_map.create_graph(repo_dir)

        with self.driver.session() as session:
            start_time = time.time()
            node_count = nx_graph.number_of_nodes()
            self.cache_metrics.total_nodes = node_count
            logging.info(f"Creating {node_count} nodes")

            # Create content hash index
            self.create_content_hash_index(session)

            # Get existing node hashes
            existing_hashes = self.get_existing_node_hashes(session, project_id)

            # Batch insert nodes
            batch_size = 300
            failed_batches = []
            
            def process_node_batch(tx, nodes_to_create):
                """Process a batch of nodes within a transaction."""
                result = tx.run(
                    """
                    UNWIND $nodes AS node
                    CALL apoc.create.node(node.labels, node) YIELD node AS n
                    RETURN count(*) AS created_count
                    """,
                    nodes=nodes_to_create,
                )
                return result.single()["created_count"]

            for i in range(0, node_count, batch_size):
                batch_nodes = list(nx_graph.nodes(data=True))[i : i + batch_size]
                nodes_to_create = []

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

                    # Generate content hash
                    content_hash = self.generate_content_hash(node_data)
                    node_id_str = CodeGraphService.generate_node_id(node_id, user_id)

                    # Prepare node data
                    processed_node = {
                        "name": node_data.get(
                            "name", node_id
                        ),  # Use node_id as fallback
                        "file_path": node_data.get("file", ""),
                        "start_line": node_data.get("line", -1),
                        "end_line": node_data.get("end_line", -1),
                        "repoId": project_id,
                        "node_id": node_id_str,
                        "entityId": user_id,
                        "type": node_type,
                        "text": node_data.get("text", ""),
                        "content_hash": content_hash,
                        "labels": labels,
                    }

                    # Reuse existing node data if hash matches
                    if node_id_str in existing_hashes and existing_hashes[node_id_str]["hash"] == content_hash:
                        cached_data = existing_hashes[node_id_str]
                        has_docstring, has_embedding = self.validate_cached_data(cached_data)
                        
                        if has_docstring:
                            processed_node["docstring"] = cached_data["docstring"]
                        if has_embedding:
                            processed_node["embedding"] = cached_data["embedding"]
                        if cached_data.get("tags"):
                            processed_node["tags"] = cached_data["tags"]
                            
                        if has_docstring and has_embedding:
                            self.cache_metrics.cached_nodes += 1

                    # Remove None values
                    processed_node = {
                        k: v for k, v in processed_node.items() if v is not None
                    }
                    nodes_to_create.append(processed_node)

                try:
                    # Create nodes with labels in a transaction
                    created_count = session.execute_write(process_node_batch, nodes_to_create)
                    logger.info(f"Successfully created {created_count} nodes in batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error creating nodes batch {i//batch_size + 1}: {str(e)}")
                    failed_batches.append((i, nodes_to_create))
                    continue

            if failed_batches:
                logger.error(f"Failed to create {len(failed_batches)} node batches. Attempting retry...")
                for batch_index, nodes in failed_batches:
                    try:
                        created_count = session.execute_write(process_node_batch, nodes)
                        logger.info(f"Retry successful: Created {created_count} nodes in batch {batch_index//batch_size + 1}")
                    except Exception as e:
                        logger.error(f"Retry failed for batch {batch_index//batch_size + 1}: {str(e)}")

            relationship_count = nx_graph.number_of_edges()
            logging.info(f"Creating {relationship_count} relationships")

            # Create relationships in batches
            failed_rel_batches = []
            
            def process_relationship_batch(tx, edges):
                """Process a batch of relationships within a transaction."""
                result = tx.run(
                    """
                    UNWIND $edges AS edge
                    MATCH (source:NODE {node_id: edge.source_id, repoId: edge.repoId})
                    MATCH (target:NODE {node_id: edge.target_id, repoId: edge.repoId})
                    CALL apoc.create.relationship(source, edge.type, {repoId: edge.repoId}, target) YIELD rel
                    RETURN count(rel) AS created_count
                    """,
                    edges=edges,
                )
                return result.single()["created_count"]

            for i in range(0, relationship_count, batch_size):
                batch_edges = list(nx_graph.edges(data=True))[i : i + batch_size]
                edges_to_create = []
                for source, target, data in batch_edges:
                    edge_data = {
                        "source_id": CodeGraphService.generate_node_id(source, user_id),
                        "target_id": CodeGraphService.generate_node_id(target, user_id),
                        "type": data.get("type", "REFERENCES"),
                        "repoId": project_id,
                    }
                    # Remove any null values from edge_data
                    edge_data = {k: v for k, v in edge_data.items() if v is not None}
                    edges_to_create.append(edge_data)

                try:
                    # Create relationships in a transaction
                    created_count = session.execute_write(process_relationship_batch, edges_to_create)
                    logger.info(f"Successfully created {created_count} relationships in batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error creating relationships batch {i//batch_size + 1}: {str(e)}")
                    failed_rel_batches.append((i, edges_to_create))
                    continue

            if failed_rel_batches:
                logger.error(f"Failed to create {len(failed_rel_batches)} relationship batches. Attempting retry...")
                for batch_index, edges in failed_rel_batches:
                    try:
                        created_count = session.execute_write(process_relationship_batch, edges)
                        logger.info(f"Retry successful: Created {created_count} relationships in batch {batch_index//batch_size + 1}")
                    except Exception as e:
                        logger.error(f"Retry failed for batch {batch_index//batch_size + 1}: {str(e)}")

            end_time = time.time()
            
            # Log cache metrics
            self.cache_metrics.log_metrics(project_id)
            
            # Log final status
            logger.info(
                f"Graph creation completed:\n"
                f"Time taken: {end_time - start_time:.2f} seconds\n"
                f"Failed node batches: {len(failed_batches)}\n"
                f"Failed relationship batches: {len(failed_rel_batches)}"
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
                return f.read()
        except UnicodeDecodeError:
            logging.warning(f"Could not read {fname} as UTF-8. Skipping this file.")
            return ""

    def tool_error(self, message):
        logging.error(f"Error: {message}")

    def tool_output(self, message):
        logging.info(message)


class SimpleTokenCounter:
    def token_count(self, text):
        return len(text.split())

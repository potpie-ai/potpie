import logging
from typing import Dict, List, Set, Tuple
from neo4j import GraphDatabase
from sqlalchemy.orm import Session
import time

from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.parsing.graph_construction.parsing_repomap import RepoMap

logger = logging.getLogger(__name__)

class IncrementalUpdateService:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, db: Session):
        """Initialize the incremental update service."""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.db = db
        self.code_graph_service = CodeGraphService(neo4j_uri, neo4j_user, neo4j_password, db)
        self.inference_service = InferenceService(db)

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def _get_file_nodes(self, project_id: str, file_path: str) -> List[Dict]:
        """Get all nodes associated with a specific file."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE {repoId: $project_id})
                WHERE n.file_path = $file_path
                RETURN n
                """,
                project_id=project_id,
                file_path=file_path
            )
            return [dict(record["n"]) for record in result]

    def _get_affected_nodes(self, project_id: str, file_nodes: List[Dict]) -> Set[str]:
        """Get all nodes that are connected to the modified nodes."""
        affected_nodes = set()
        with self.driver.session() as session:
            for node in file_nodes:
                result = session.run(
                    """
                    MATCH (n:NODE {node_id: $node_id})-[*1..2]-(m:NODE)
                    WHERE m.repoId = $project_id
                    RETURN DISTINCT m.node_id AS node_id
                    """,
                    node_id=node["node_id"],
                    project_id=project_id
                )
                affected_nodes.update(record["node_id"] for record in result)
        return affected_nodes

    def _remove_file_nodes(self, project_id: str, file_path: str):
        """Remove all nodes and their relationships for a specific file."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n:NODE {repoId: $project_id, file_path: $file_path})
                DETACH DELETE n
                """,
                project_id=project_id,
                file_path=file_path
            )

    def _create_change_log(self, project_id: str, file_path: str, change_type: str, nodes_affected: int, user_id: str):
        """Record a change in the graph for tracking purposes."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (c:CHANGE {
                    repoId: $project_id,
                    file_path: $file_path,
                    change_type: $change_type,
                    nodes_affected: $nodes_affected,
                    user_id: $user_id,
                    timestamp: datetime()
                })
                """,
                project_id=project_id,
                file_path=file_path,
                change_type=change_type,
                nodes_affected=nodes_affected,
                user_id=user_id
            )

    def get_change_history(self, project_id: str, file_path: str = None) -> List[Dict]:
        """
        Get the history of changes for a project or specific file.
        
        Args:
            project_id: Project identifier
            file_path: Optional file path to filter changes
            
        Returns:
            List[Dict]: List of change records
        """
        with self.driver.session() as session:
            query = """
                MATCH (c:CHANGE {repoId: $project_id})
                WHERE $file_path IS NULL OR c.file_path = $file_path
                RETURN c
                ORDER BY c.timestamp DESC
                LIMIT 100
            """
            result = session.run(query, project_id=project_id, file_path=file_path)
            return [dict(record["c"]) for record in result]

    def create_snapshot(self, project_id: str, snapshot_name: str) -> str:
        """
        Create a named snapshot of the current graph state.
        
        Args:
            project_id: Project identifier
            snapshot_name: Name for the snapshot
            
        Returns:
            str: Snapshot identifier
        """
        snapshot_id = f"{project_id}_{snapshot_name}_{int(time.time())}"
        
        with self.driver.session() as session:
            # Create snapshot metadata
            session.run(
                """
                CREATE (s:SNAPSHOT {
                    id: $snapshot_id,
                    repoId: $project_id,
                    name: $name,
                    created_at: datetime(),
                    node_count: 0,
                    relationship_count: 0
                })
                """,
                snapshot_id=snapshot_id,
                project_id=project_id,
                name=snapshot_name
            )
            
            # Store nodes in snapshot
            session.run(
                """
                MATCH (n:NODE {repoId: $project_id})
                WITH collect(n) as nodes
                MATCH (s:SNAPSHOT {id: $snapshot_id})
                SET s.nodes = [node IN nodes | {
                    node_id: node.node_id,
                    name: node.name,
                    file_path: node.file_path,
                    start_line: node.start_line,
                    end_line: node.end_line,
                    type: node.type,
                    text: node.text,
                    docstring: node.docstring,
                    embedding: node.embedding,
                    tags: node.tags,
                    labels: labels(node)
                }]
                """,
                project_id=project_id,
                snapshot_id=snapshot_id
            )
            
            # Store relationships in snapshot
            session.run(
                """
                MATCH (n:NODE {repoId: $project_id})-[r]->(m:NODE {repoId: $project_id})
                WITH collect({
                    source_id: n.node_id,
                    target_id: m.node_id,
                    type: type(r),
                    properties: properties(r)
                }) as relationships
                MATCH (s:SNAPSHOT {id: $snapshot_id})
                SET s.relationships = relationships,
                    s.node_count = size(s.nodes),
                    s.relationship_count = size(relationships)
                """,
                project_id=project_id,
                snapshot_id=snapshot_id
            )
            
            # Get counts for logging
            result = session.run(
                """
                MATCH (s:SNAPSHOT {id: $snapshot_id})
                RETURN s.node_count as nodes, s.relationship_count as rels
                """,
                snapshot_id=snapshot_id
            )
            
            record = result.single()
            logger.info(
                f"Created snapshot {snapshot_name} with {record['nodes']} nodes and {record['rels']} relationships"
            )
            
        return snapshot_id

    def list_snapshots(self, project_id: str) -> List[Dict]:
        """List all snapshots for a project."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:SNAPSHOT {repoId: $project_id})
                RETURN s
                ORDER BY s.created_at DESC
                """,
                project_id=project_id
            )
            return [dict(record["s"]) for record in result]

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore the graph to a previous snapshot state.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            bool: True if restore was successful
        """
        try:
            with self.driver.session() as session:
                # Get snapshot info
                result = session.run(
                    """
                    MATCH (s:SNAPSHOT {id: $snapshot_id})
                    RETURN s.repoId as project_id, s.nodes as nodes, s.relationships as relationships
                    """,
                    snapshot_id=snapshot_id
                )
                snapshot = result.single()
                if not snapshot:
                    logger.error(f"Snapshot {snapshot_id} not found")
                    return False
                
                project_id = snapshot["project_id"]
                
                # Begin transaction for atomic restore
                tx = session.begin_transaction()
                try:
                    # Remove existing nodes and relationships
                    tx.run(
                        """
                        MATCH (n:NODE {repoId: $project_id})
                        DETACH DELETE n
                        """,
                        project_id=project_id
                    )
                    
                    # Restore nodes
                    tx.run(
                        """
                        UNWIND $nodes as node
                        CALL apoc.create.node(
                            node.labels,
                            apoc.map.merge(node, {repoId: $project_id})
                        ) YIELD node as n
                        """,
                        nodes=snapshot["nodes"],
                        project_id=project_id
                    )
                    
                    # Restore relationships
                    tx.run(
                        """
                        UNWIND $relationships as rel
                        MATCH (source:NODE {repoId: $project_id, node_id: rel.source_id})
                        MATCH (target:NODE {repoId: $project_id, node_id: rel.target_id})
                        CALL apoc.create.relationship(
                            source,
                            rel.type,
                            rel.properties,
                            target
                        ) YIELD rel as r
                        """,
                        relationships=snapshot["relationships"]
                    )
                    
                    # Commit transaction
                    tx.commit()
                    
                    logger.info(
                        f"Successfully restored snapshot {snapshot_id} with "
                        f"{len(snapshot['nodes'])} nodes and {len(snapshot['relationships'])} relationships"
                    )
                    return True
                    
                except Exception as e:
                    # Rollback transaction on error
                    tx.rollback()
                    logger.error(f"Error during snapshot restore: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Failed to restore snapshot {snapshot_id}: {str(e)}")
            return False

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:SNAPSHOT {id: $snapshot_id})
                    DELETE s
                    RETURN count(*) as deleted
                    """,
                    snapshot_id=snapshot_id
                )
                deleted = result.single()["deleted"]
                return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete snapshot {snapshot_id}: {str(e)}")
            return False

    def get_snapshot_info(self, snapshot_id: str) -> Dict:
        """
        Get detailed information about a snapshot.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            Dict: Snapshot information including metadata and statistics
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:SNAPSHOT {id: $snapshot_id})
                RETURN s
                """,
                snapshot_id=snapshot_id
            )
            record = result.single()
            if not record:
                return None
            
            snapshot = dict(record["s"])
            return {
                "id": snapshot["id"],
                "name": snapshot["name"],
                "project_id": snapshot["repoId"],
                "created_at": snapshot["created_at"],
                "node_count": snapshot["node_count"],
                "relationship_count": snapshot["relationship_count"]
            }

    async def update_file(self, repo_dir: str, project_id: str, file_path: str, user_id: str) -> Tuple[int, int]:
        """
        Update the knowledge graph for a single modified file.
        
        Returns:
            Tuple[int, int]: Number of nodes and relationships updated
        """
        logger.info(f"Incrementally updating file: {file_path}")
        
        # Get existing nodes for the file
        existing_nodes = self._get_file_nodes(project_id, file_path)
        affected_nodes = self._get_affected_nodes(project_id, existing_nodes)
        
        # Remove existing nodes for the file
        self._remove_file_nodes(project_id, file_path)
        
        # Create new graph for the file
        repo_map = RepoMap(root=repo_dir)
        new_graph = repo_map.create_graph_for_file(file_path)
        
        # Store new nodes and relationships
        nodes_updated = 0
        relationships_updated = 0
        
        with self.driver.session() as session:
            # Create new nodes
            for node_id, node_data in new_graph.nodes(data=True):
                node_type = node_data.get("type", "UNKNOWN")
                if node_type == "UNKNOWN":
                    continue
                
                labels = ["NODE"]
                if node_type in ["FILE", "CLASS", "FUNCTION", "INTERFACE"]:
                    labels.append(node_type)
                
                processed_node = {
                    "name": node_data.get("name", node_id),
                    "file_path": node_data.get("file", ""),
                    "start_line": node_data.get("line", -1),
                    "end_line": node_data.get("end_line", -1),
                    "repoId": project_id,
                    "node_id": CodeGraphService.generate_node_id(node_id, user_id),
                    "entityId": user_id,
                    "type": node_type,
                    "text": node_data.get("text", ""),
                    "labels": labels
                }
                
                session.run(
                    """
                    CALL apoc.create.node($labels, $node) YIELD node
                    RETURN count(*) as count
                    """,
                    labels=labels,
                    node={k: v for k, v in processed_node.items() if v is not None}
                )
                nodes_updated += 1
            
            # Create new relationships
            for source, target, data in new_graph.edges(data=True):
                edge_data = {
                    "source_id": CodeGraphService.generate_node_id(source, user_id),
                    "target_id": CodeGraphService.generate_node_id(target, user_id),
                    "type": data.get("type", "REFERENCES"),
                    "repoId": project_id
                }
                
                session.run(
                    """
                    MATCH (source:NODE {node_id: $source_id, repoId: $repo_id})
                    MATCH (target:NODE {node_id: $target_id, repoId: $repo_id})
                    CALL apoc.create.relationship(source, $type, {repoId: $repo_id}, target) YIELD rel
                    RETURN count(rel) as count
                    """,
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    type=edge_data["type"],
                    repo_id=project_id
                )
                relationships_updated += 1
        
        # Update inferences for affected nodes
        await self._update_inferences(project_id, affected_nodes)
        
        # Record the change
        self._create_change_log(
            project_id,
            file_path,
            "UPDATE",
            nodes_updated,
            user_id
        )
        
        logger.info(f"Updated {nodes_updated} nodes and {relationships_updated} relationships for {file_path}")
        return nodes_updated, relationships_updated

    async def _update_inferences(self, project_id: str, affected_node_ids: Set[str]):
        """Update inferences only for affected nodes."""
        if not affected_node_ids:
            return
            
        with self.driver.session() as session:
            # Get nodes that need inference updates
            result = session.run(
                """
                MATCH (n:NODE)
                WHERE n.repoId = $project_id AND n.node_id IN $node_ids
                RETURN n.node_id as node_id, n.text as text
                """,
                project_id=project_id,
                node_ids=list(affected_node_ids)
            )
            
            nodes_to_update = [
                {"node_id": record["node_id"], "text": record["text"]}
                for record in result
            ]
            
            # Update inferences in batches
            batch_size = 50
            for i in range(0, len(nodes_to_update), batch_size):
                batch = nodes_to_update[i:i + batch_size]
                await self.inference_service.generate_response(batch, project_id)

    async def update_files(self, repo_dir: str, project_id: str, file_paths: List[str], user_id: str) -> Dict[str, Tuple[int, int]]:
        """
        Update multiple files in the knowledge graph.
        
        Args:
            repo_dir: Repository directory path
            project_id: Project identifier
            file_paths: List of file paths to update
            user_id: User identifier
            
        Returns:
            Dict[str, Tuple[int, int]]: Map of file paths to (nodes_updated, relationships_updated)
        """
        results = {}
        for file_path in file_paths:
            try:
                nodes_updated, relationships_updated = await self.update_file(
                    repo_dir, project_id, file_path, user_id
                )
                results[file_path] = (nodes_updated, relationships_updated)
            except Exception as e:
                logger.error(f"Failed to update file {file_path}: {str(e)}")
                results[file_path] = (0, 0)
        return results

    async def delete_files(self, project_id: str, file_paths: List[str], user_id: str) -> int:
        """
        Remove files and their nodes from the knowledge graph.
        
        Args:
            project_id: Project identifier
            file_paths: List of file paths to delete
            
        Returns:
            int: Number of nodes deleted
        """
        nodes_deleted = 0
        
        for file_path in file_paths:
            try:
                # Get affected nodes before deletion
                existing_nodes = self._get_file_nodes(project_id, file_path)
                affected_nodes = self._get_affected_nodes(project_id, existing_nodes)
                
                # Remove file nodes
                self._remove_file_nodes(project_id, file_path)
                nodes_deleted += len(existing_nodes)
                
                # Update inferences for affected nodes
                await self._update_inferences(project_id, affected_nodes)
                
                # Record the change
                self._create_change_log(
                    project_id,
                    file_path,
                    "DELETE",
                    nodes_deleted,
                    user_id
                )
                
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {str(e)}")
                
        return nodes_deleted

    def get_file_status(self, project_id: str, file_path: str) -> Dict:
        """
        Get the current status of a file in the knowledge graph.
        
        Args:
            project_id: Project identifier
            file_path: File path to check
            
        Returns:
            Dict containing:
                - num_nodes: Number of nodes for the file
                - node_types: Types of nodes present
                - last_updated: Timestamp of last update
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE {repoId: $project_id, file_path: $file_path})
                RETURN 
                    count(n) as num_nodes,
                    collect(DISTINCT n.type) as node_types,
                    max(n.updated_at) as last_updated
                """,
                project_id=project_id,
                file_path=file_path
            )
            record = result.single()
            if not record:
                return {"num_nodes": 0, "node_types": [], "last_updated": None}
                
            return {
                "num_nodes": record["num_nodes"],
                "node_types": record["node_types"],
                "last_updated": record["last_updated"]
            } 
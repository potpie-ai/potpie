from dataclasses import dataclass
import hashlib
import json
import re
import time
import uuid
from typing import Dict, Optional, Tuple

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.modules.parsing.models.kg_ingest_model import KgIngestRun, KgLatestSuccessfulRun
from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.search.search_service import SearchService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class ArtifactKindDiff:
    added_keys: set[str]
    deleted_keys: set[str]
    changed_keys: set[str]

    @property
    def upsert_keys(self) -> set[str]:
        return self.added_keys | self.changed_keys

    @property
    def delete_keys(self) -> set[str]:
        return self.deleted_keys | self.changed_keys

    def summary(self) -> Dict[str, int]:
        return {
            "added": len(self.added_keys),
            "deleted": len(self.deleted_keys),
            "changed": len(self.changed_keys),
        }


@dataclass(frozen=True)
class ArtifactRunDiff:
    nodes: ArtifactKindDiff
    edges: ArtifactKindDiff


class CodeGraphService:
    _REL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    @classmethod
    def _sanitize_rel_type(cls, rel_type: object) -> str:
        if not isinstance(rel_type, str):
            return "REFERENCES"
        rel_type = rel_type.strip()
        if not rel_type or not cls._REL_TYPE_RE.match(rel_type):
            logger.debug(
                "Invalid relationship type; falling back to REFERENCES: {}", rel_type
            )
            return "REFERENCES"
        return rel_type

    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, db: Session):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.db = db

    @staticmethod
    def generate_node_id(path: str, user_id: str):
        # Concatenate path and signature
        combined_string = f"{user_id}:{path}"

        # Create an MD5 hash of the combined string
        hash_object = hashlib.md5()
        hash_object.update(combined_string.encode("utf-8"))

        # Get the hexadecimal representation of the hash
        node_id = hash_object.hexdigest()

        return node_id

    def build_graph(self, repo_dir):
        self.repo_map = RepoMap(
            root=repo_dir,
            verbose=True,
            main_model=SimpleTokenCounter(),
            io=SimpleIO(),
        )
        return self.repo_map.create_graph(repo_dir)

    @staticmethod
    def _canonical_json(payload: Dict) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def _record_key(parts: list) -> str:
        return json.dumps(parts, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def _record_hash(cls, repo_id: str, record_json: Dict) -> str:
        canonical = cls._canonical_json(record_json)
        hash_input = f"{repo_id}:{canonical}".encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()

    @staticmethod
    def _edge_disambiguator(edge_data: Dict) -> Optional[str]:
        if edge_data.get("ref_site_hash") is not None:
            return str(edge_data["ref_site_hash"])
        if edge_data.get("ref_line") is not None or edge_data.get("end_ref_line") is not None:
            return f"{edge_data.get('ref_line')}:{edge_data.get('end_ref_line')}"
        if edge_data.get("start_line") is not None or edge_data.get("end_line") is not None:
            return f"{edge_data.get('start_line')}:{edge_data.get('end_line')}"
        return None

    @staticmethod
    def _build_processed_node(
        raw_node_id: str, node_data: Dict, project_id: str, user_id: str
    ) -> Optional[Dict]:
        node_type = node_data.get("type", "UNKNOWN")
        if node_type == "UNKNOWN":
            return None

        text = node_data.get("text", "") or ""
        if not isinstance(text, str):
            text = str(text)
        # Postgres TEXT/JSONB cannot contain NUL bytes; also treat them as binary noise.
        if "\x00" in text:
            text = text.replace("\x00", "")

        labels = ["NODE"]
        if node_type in ["FILE", "CLASS", "FUNCTION", "INTERFACE"]:
            labels.append(node_type)

        processed_node = {
            "name": node_data.get("name", raw_node_id),
            "file_path": node_data.get("file", ""),
            "start_line": node_data.get("line", -1),
            "end_line": node_data.get("end_line", -1),
            "repoId": project_id,
            "node_id": CodeGraphService.generate_node_id(raw_node_id, user_id),
            "raw_node_id": raw_node_id,
            "entityId": user_id,
            "type": node_type,
            "text": text,
            "labels": labels,
        }

        return {k: v for k, v in processed_node.items() if v is not None}

    @staticmethod
    def _build_edge_properties(edge_data: Dict, repo_id: str) -> Dict:
        properties = {"repoId": repo_id}
        for key, value in edge_data.items():
            if key == "type" or value is None:
                continue
            properties[key] = value
        return properties

    @classmethod
    def _edge_record_key(
        cls, repo_id: str, rel_type: str, source_id: str, target_id: str, edge_data: Dict
    ) -> str:
        rel_type = cls._sanitize_rel_type(rel_type)
        disambiguator = cls._edge_disambiguator(edge_data)
        key_parts = [repo_id, rel_type, source_id, target_id]
        if disambiguator is not None:
            key_parts.append(disambiguator)
        return cls._record_key(key_parts)

    def _upsert_latest_successful_run(self, repo_id: str, user_id: str, run_id) -> None:
        latest = (
            self.db.query(KgLatestSuccessfulRun)
            .filter(
                KgLatestSuccessfulRun.repo_id == repo_id,
                KgLatestSuccessfulRun.user_id == user_id,
            )
            .one_or_none()
        )
        if latest:
            latest.run_id = run_id
        else:
            self.db.add(
                KgLatestSuccessfulRun(repo_id=repo_id, user_id=user_id, run_id=run_id)
            )

    def get_latest_successful_run_id(self, repo_id: str, user_id: str):
        latest = (
            self.db.query(KgLatestSuccessfulRun)
            .filter(
                KgLatestSuccessfulRun.repo_id == repo_id,
                KgLatestSuccessfulRun.user_id == user_id,
            )
            .one_or_none()
        )
        return latest.run_id if latest else None

    def get_latest_successful_commit_id(
        self, repo_id: str, user_id: str
    ) -> Optional[str]:
        latest_run_id = self.get_latest_successful_run_id(repo_id, user_id)
        if not latest_run_id:
            return None
        return (
            self.db.query(KgIngestRun.commit_id)
            .filter(KgIngestRun.run_id == latest_run_id)
            .scalar()
        )

    def build_artifact_key_hash_maps(
        self, nx_graph, repo_id: str, user_id: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        node_map: Dict[str, str] = {}
        for raw_node_id, node_data in nx_graph.nodes(data=True):
            processed_node = self._build_processed_node(
                raw_node_id, node_data, repo_id, user_id
            )
            if not processed_node:
                continue
            record_key = self._record_key([repo_id, processed_node["node_id"]])
            record_hash = self._record_hash(repo_id, processed_node)
            node_map[record_key] = record_hash

        edge_map: Dict[str, str] = {}
        for source, target, data in nx_graph.edges(data=True):
            rel_type = self._sanitize_rel_type(data.get("type", "REFERENCES"))
            source_id = CodeGraphService.generate_node_id(source, user_id)
            target_id = CodeGraphService.generate_node_id(target, user_id)
            edge_properties = self._build_edge_properties(data, repo_id)
            record_key = self._edge_record_key(
                repo_id, rel_type, source_id, target_id, data
            )
            record_json = {
                "repoId": repo_id,
                "rel_type": rel_type,
                "source_id": source_id,
                "target_id": target_id,
                "properties": edge_properties,
            }
            record_hash = self._record_hash(repo_id, record_json)
            edge_map[record_key] = record_hash

        return node_map, edge_map

    def diff_artifact_maps(
        self,
        old_node_map: Dict[str, str],
        new_node_map: Dict[str, str],
        old_edge_map: Dict[str, str],
        new_edge_map: Dict[str, str],
    ) -> ArtifactRunDiff:
        return ArtifactRunDiff(
            nodes=self._diff_key_hash_maps(old_node_map, new_node_map),
            edges=self._diff_key_hash_maps(old_edge_map, new_edge_map),
        )

    @staticmethod
    def _node_id_from_record_key(record_key: str) -> Optional[str]:
        try:
            parts = json.loads(record_key)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(parts, list) and len(parts) >= 2:
            return parts[1]
        return None

    @staticmethod
    def _diff_key_hash_maps(old: Dict[str, str], new: Dict[str, str]) -> ArtifactKindDiff:
        old_keys = set(old.keys())
        new_keys = set(new.keys())

        added_keys = new_keys - old_keys
        deleted_keys = old_keys - new_keys

        common_keys = old_keys & new_keys
        changed_keys = {key for key in common_keys if old[key] != new[key]}

        return ArtifactKindDiff(
            added_keys=added_keys,
            deleted_keys=deleted_keys,
            changed_keys=changed_keys,
        )

    def record_successful_ingest_run(
        self, repo_id: str, user_id: str, commit_id: Optional[str]
    ) -> uuid.UUID:
        run_id = uuid.uuid4()
        run = KgIngestRun(
            run_id=run_id,
            repo_id=repo_id,
            user_id=user_id,
            commit_id=commit_id,
            status="success",
        )
        self.db.add(run)
        self._upsert_latest_successful_run(repo_id, user_id, run_id)
        self.db.commit()
        return run_id

    def set_ingest_run_status(self, run_id, status: str) -> None:
        run = (
            self.db.query(KgIngestRun)
            .filter(KgIngestRun.run_id == run_id)
            .one_or_none()
        )
        if not run:
            raise ValueError(f"KG ingest run not found: {run_id}")
        run.status = status

    def mark_latest_successful_run(self, repo_id: str, user_id: str, run_id) -> None:
        self._upsert_latest_successful_run(repo_id, user_id, run_id)

    def close(self):
        self.driver.close()

    def create_and_store_graph(
        self,
        repo_dir,
        project_id,
        user_id,
    ):
        nx_graph = self.build_graph(repo_dir)

        with self.driver.session() as session:
            start_time = time.time()
            nodes = list(nx_graph.nodes(data=True))
            node_count = len(nodes)
            logger.info(f"Creating {node_count} nodes")

            # Create specialized index for relationship queries
            session.run(
                """
                CREATE INDEX node_id_repo_idx IF NOT EXISTS
                FOR (n:NODE) ON (n.node_id, n.repoId)
            """
            )

            # Batch insert nodes
            batch_size = 1000
            for i in range(0, node_count, batch_size):
                batch_nodes = nodes[i : i + batch_size]
                nodes_to_create = []

                for node_id, node_data in batch_nodes:
                    processed_node = self._build_processed_node(
                        node_id, node_data, project_id, user_id
                    )
                    if not processed_node:
                        continue
                    nodes_to_create.append(processed_node)

                # Create nodes with labels
                session.run(
                    """
                    UNWIND $nodes AS node
                    CALL apoc.create.node(node.labels, node) YIELD node AS n
                    RETURN count(*) AS created_count
                    """,
                    nodes=nodes_to_create,
                )

            relationship_count = nx_graph.number_of_edges()
            logger.info(f"Creating {relationship_count} relationships")

            # Group edges by relationship type to avoid repeated scans.
            edges_by_type: Dict[str, list[Dict]] = {}
            for source, target, data in nx_graph.edges(data=True):
                rel_type = self._sanitize_rel_type(data.get("type", "REFERENCES"))
                source_id = CodeGraphService.generate_node_id(source, user_id)
                target_id = CodeGraphService.generate_node_id(target, user_id)
                edge_properties = self._build_edge_properties(data, project_id)
                edge_key = self._edge_record_key(
                    project_id, rel_type, source_id, target_id, data
                )
                edges_by_type.setdefault(rel_type, []).append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "repoId": project_id,
                        "edge_key": edge_key,
                        "properties": edge_properties,
                    }
                )

            # Process relationships with huge batch size and type-specific queries
            batch_size = 1000

            for rel_type, type_edges in edges_by_type.items():
                logger.debug(
                    f"Creating {len(type_edges)} relationships of type {rel_type}"
                )

                for i in range(0, len(type_edges), batch_size):
                    batch_edges = type_edges[i : i + batch_size]
                    edges_to_create = list(batch_edges)

                    # Type-specific relationship creation in one transaction
                    query = f"""
                        UNWIND $edges AS edge
                        MATCH (source:NODE {{node_id: edge.source_id, repoId: edge.repoId}})
                        MATCH (target:NODE {{node_id: edge.target_id, repoId: edge.repoId}})
                        MERGE (source)-[r:{rel_type} {{repoId: edge.repoId, edge_key: edge.edge_key}}]->(target)
                        SET r += edge.properties
                    """
                    session.run(query, edges=edges_to_create)

            end_time = time.time()
            logger.info(
                f"Time taken to create graph and search index: {end_time - start_time:.2f} seconds"
            )

    def apply_incremental_diff_in_memory(
        self,
        repo_id: str,
        nx_graph,
        diff: ArtifactRunDiff,
        user_id: str,
        *,
        node_batch_size: int = 1000,
        edge_batch_size: int = 1000,
    ) -> None:
        touched_node_ids = self._apply_node_deltas_from_graph(
            repo_id, nx_graph, diff, user_id, node_batch_size
        )

        self._apply_edge_deltas_from_graph(
            repo_id, nx_graph, diff, touched_node_ids, user_id, edge_batch_size
        )

        logger.debug(
            "Applied node deltas for incremental ingest (in-memory)",
            repo_id=repo_id,
            touched_node_count=len(touched_node_ids),
        )

    def _apply_node_deltas_from_graph(
        self,
        repo_id: str,
        nx_graph,
        diff: ArtifactRunDiff,
        user_id: str,
        node_batch_size: int,
    ) -> list[str]:
        node_ids_to_delete = []
        for record_key in diff.nodes.delete_keys:
            node_id = self._node_id_from_record_key(record_key)
            if node_id:
                node_ids_to_delete.append(node_id)
        self._delete_nodes(repo_id, node_ids_to_delete, node_batch_size)

        nodes_to_upsert: list[Dict] = []
        upsert_keys = diff.nodes.upsert_keys
        if upsert_keys:
            for raw_node_id, node_data in nx_graph.nodes(data=True):
                processed_node = self._build_processed_node(
                    raw_node_id, node_data, repo_id, user_id
                )
                if not processed_node:
                    continue
                record_key = self._record_key([repo_id, processed_node["node_id"]])
                if record_key in upsert_keys:
                    nodes_to_upsert.append(processed_node)
        self._upsert_nodes(nodes_to_upsert, node_batch_size)
        return node_ids_to_delete

    def _apply_edge_deltas_from_graph(
        self,
        repo_id: str,
        nx_graph,
        diff: ArtifactRunDiff,
        touched_node_ids: list[str],
        user_id: str,
        edge_batch_size: int,
    ) -> None:
        self._delete_edges_by_key(repo_id, diff.edges.delete_keys, edge_batch_size)

        touched_set = set(touched_node_ids)
        upsert_keys = diff.edges.upsert_keys
        edges_by_type: Dict[str, list[Dict]] = {}
        seen_edge_keys: set[str] = set()
        incident_edge_count = 0
        upsert_edge_count = 0

        if upsert_keys or touched_set:
            for source, target, data in nx_graph.edges(data=True):
                rel_type = self._sanitize_rel_type(data.get("type", "REFERENCES"))
                source_id = CodeGraphService.generate_node_id(source, user_id)
                target_id = CodeGraphService.generate_node_id(target, user_id)
                edge_key = self._edge_record_key(
                    repo_id, rel_type, source_id, target_id, data
                )
                is_incident = source_id in touched_set or target_id in touched_set
                if edge_key not in upsert_keys and not is_incident:
                    continue
                if edge_key in seen_edge_keys:
                    continue
                seen_edge_keys.add(edge_key)

                edge_properties = self._build_edge_properties(data, repo_id)
                edges_by_type.setdefault(rel_type, []).append(
                    {
                        "repoId": repo_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "record_key": edge_key,
                        "properties": edge_properties,
                    }
                )
                if edge_key in upsert_keys:
                    upsert_edge_count += 1
                if is_incident:
                    incident_edge_count += 1

        self._upsert_edges_from_records(edges_by_type, edge_batch_size)

        logger.debug(
            "Applied edge deltas for incremental ingest (in-memory)",
            repo_id=repo_id,
            deleted_edge_count=len(diff.edges.delete_keys),
            upsert_edge_count=upsert_edge_count,
            incident_edge_count=incident_edge_count,
        )

    def _upsert_edges_from_records(
        self, edges_by_type: Dict[str, list[Dict]], batch_size: int
    ) -> None:
        if not edges_by_type:
            return
        with self.driver.session() as session:
            for rel_type, edges in edges_by_type.items():
                for i in range(0, len(edges), batch_size):
                    batch = edges[i : i + batch_size]
                    query = f"""
                        UNWIND $edges AS edge
                        MATCH (source:NODE {{node_id: edge.source_id, repoId: edge.repoId}})
                        MATCH (target:NODE {{node_id: edge.target_id, repoId: edge.repoId}})
                        MERGE (source)-[r:{rel_type} {{repoId: edge.repoId, edge_key: edge.record_key}}]->(target)
                        SET r += edge.properties
                    """
                    session.run(query, edges=batch)

    def _delete_edges_by_key(
        self, repo_id: str, edge_keys: set[str], batch_size: int
    ) -> None:
        if not edge_keys:
            return
        edge_key_list = list(edge_keys)
        with self.driver.session() as session:
            for i in range(0, len(edge_key_list), batch_size):
                batch = edge_key_list[i : i + batch_size]
                session.run(
                    """
                    UNWIND $edge_keys AS edge_key
                    MATCH ()-[r {repoId: $repo_id, edge_key: edge_key}]-()
                    DELETE r
                    """,
                    repo_id=repo_id,
                    edge_keys=batch,
                )

    def _delete_nodes(self, repo_id: str, node_ids: list[str], batch_size: int) -> None:
        if not node_ids:
            return
        with self.driver.session() as session:
            for i in range(0, len(node_ids), batch_size):
                batch = node_ids[i : i + batch_size]
                session.run(
                    """
                    UNWIND $node_ids AS node_id
                    MATCH (n:NODE {repoId: $repo_id, node_id: node_id})
                    DETACH DELETE n
                    """,
                    repo_id=repo_id,
                    node_ids=batch,
                )

    def _upsert_nodes(self, nodes: list[Dict], batch_size: int) -> None:
        if not nodes:
            return
        with self.driver.session() as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                session.run(
                    """
                    UNWIND $nodes AS node
                    MERGE (n:NODE {repoId: node.repoId, node_id: node.node_id})
                    SET n += node
                    WITH n, node
                    CALL apoc.create.addLabels(n, node.labels) YIELD node AS _
                    RETURN count(*) AS upserted
                    """,
                    nodes=batch,
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
                        logger.debug(
                            "Read {} using {} encoding", str(fname), str(encoding)
                        )
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                logger.exception("Error reading {}", str(fname))
                return ""

        logger.warning(
            "Could not read {} with any supported encoding. Skipping this file.",
            str(fname),
        )
        return ""

    def tool_error(self, message):
        logger.error("Error: {}", str(message))

    def tool_output(self, message):
        logger.info("{}", str(message))


class SimpleTokenCounter:
    def token_count(self, text):
        return len(text.split())

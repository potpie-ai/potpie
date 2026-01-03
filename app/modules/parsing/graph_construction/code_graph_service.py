import hashlib
import json
import time
import uuid
from typing import Dict, Optional

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.modules.parsing.models.kg_ingest_model import (
    KgArtifactRecord,
    KgIngestRun,
    KgLatestSuccessfulRun,
)
from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.search.search_service import SearchService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class CodeGraphService:
    _ARTIFACT_KIND_NODE = "node"
    _ARTIFACT_KIND_EDGE = "edge"

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

    def _node_record(self, run_id, repo_id: str, node_json: Dict) -> Dict:
        record_key = self._record_key([repo_id, node_json["node_id"]])
        record_hash = self._record_hash(repo_id, node_json)
        return {
            "run_id": run_id,
            "kind": self._ARTIFACT_KIND_NODE,
            "record_key": record_key,
            "record_hash": record_hash,
            "record_json": node_json,
        }

    def _edge_record(
        self,
        run_id,
        repo_id: str,
        rel_type: str,
        source_id: str,
        target_id: str,
        edge_properties: Dict,
        edge_data: Dict,
    ) -> Dict:
        record_json = {
            "repoId": repo_id,
            "rel_type": rel_type,
            "source_id": source_id,
            "target_id": target_id,
            "properties": edge_properties,
        }
        disambiguator = self._edge_disambiguator(edge_data)
        key_parts = [repo_id, rel_type, source_id, target_id]
        if disambiguator is not None:
            key_parts.append(disambiguator)
        record_key = self._record_key(key_parts)
        record_hash = self._record_hash(repo_id, record_json)
        return {
            "run_id": run_id,
            "kind": self._ARTIFACT_KIND_EDGE,
            "record_key": record_key,
            "record_hash": record_hash,
            "record_json": record_json,
        }

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

    def _persist_artifacts(
        self, nx_graph, project_id: str, user_id: str, commit_id: Optional[str]
    ) -> None:
        run_id = uuid.uuid4()
        run = KgIngestRun(
            run_id=run_id,
            repo_id=project_id,
            user_id=user_id,
            commit_id=commit_id,
            status="success",
        )
        self.db.add(run)
        self.db.flush()

        batch = []
        batch_size = 2000

        def flush_batch():
            if batch:
                self.db.bulk_insert_mappings(KgArtifactRecord, batch)
                batch.clear()

        for raw_node_id, node_data in nx_graph.nodes(data=True):
            processed_node = self._build_processed_node(
                raw_node_id, node_data, project_id, user_id
            )
            if not processed_node:
                continue
            batch.append(self._node_record(run_id, project_id, processed_node))
            if len(batch) >= batch_size:
                flush_batch()

        for source, target, data in nx_graph.edges(data=True):
            rel_type = data.get("type", "REFERENCES")
            source_id = CodeGraphService.generate_node_id(source, user_id)
            target_id = CodeGraphService.generate_node_id(target, user_id)
            edge_properties = self._build_edge_properties(data, project_id)
            batch.append(
                self._edge_record(
                    run_id,
                    project_id,
                    rel_type,
                    source_id,
                    target_id,
                    edge_properties,
                    data,
                )
            )
            if len(batch) >= batch_size:
                flush_batch()

        flush_batch()
        self._upsert_latest_successful_run(project_id, user_id, run_id)
        self.db.commit()

    def close(self):
        self.driver.close()

    def create_and_store_graph(
        self,
        repo_dir,
        project_id,
        user_id,
        *,
        persist_artifacts: bool = False,
        commit_id: Optional[str] = None,
    ):
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
                batch_nodes = list(nx_graph.nodes(data=True))[i : i + batch_size]
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

                logger.info(
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
            logger.info(
                f"Time taken to create graph and search index: {end_time - start_time:.2f} seconds"
            )

        if persist_artifacts:
            try:
                self._persist_artifacts(nx_graph, project_id, user_id, commit_id)
            except Exception:
                self.db.rollback()
                logger.exception(
                    "Failed to persist KG artifact records",
                    project_id=project_id,
                    user_id=user_id,
                )
                raise

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
        logger.error(f"Error: {message}")

    def tool_output(self, message):
        logger.info(message)


class SimpleTokenCounter:
    def token_count(self, text):
        return len(text.split())

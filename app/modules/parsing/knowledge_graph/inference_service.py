import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional

import tiktoken
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.parsing.knowledge_graph.inference_schema import (
    DocstringRequest,
    DocstringResponse,
)
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)

# Global singleton for SentenceTransformer to avoid reloading
_embedding_model = None


def get_embedding_model():
    """Get the singleton SentenceTransformer model, loading it only once"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model (first time only)")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("SentenceTransformer model loaded successfully")
    return _embedding_model


class InferenceService:
    def __init__(self, db: Session, user_id: Optional[str] = "dummy"):
        neo4j_config = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

        self.provider_service = ProviderService(db, user_id)
        self.embedding_model = get_embedding_model()  # Use singleton to avoid reloading
        self.search_service = SearchService(db)
        self.project_manager = ProjectService(db)
        self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", 50))

    def close(self):
        self.driver.close()

    def log_graph_stats(self, repo_id):
        query = """
        MATCH (n:NODE {repoId: $repo_id})
        OPTIONAL MATCH (n)-[r]-(m:NODE {repoId: $repo_id})
        RETURN
        COUNT(DISTINCT n) AS nodeCount,
        COUNT(DISTINCT r) AS relationshipCount
        """

        try:
            # Establish connection
            with self.driver.session() as session:
                # Execute the query
                result = session.run(query, repo_id=repo_id)
                record = result.single()

                if record:
                    node_count = record["nodeCount"]
                    relationship_count = record["relationshipCount"]

                    # Log the results
                    logger.info(
                        f"DEBUGNEO4J: Repo ID: {repo_id}, Nodes: {node_count}, Relationships: {relationship_count}"
                    )
                else:
                    logger.info(
                        f"DEBUGNEO4J: No data found for repository ID: {repo_id}"
                    )

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def num_tokens_from_string(self, string: str, model: str = "gpt-4") -> int:
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string, disallowed_special=set()))

    def fetch_graph(self, repo_id: str) -> List[Dict]:
        batch_size = 500
        all_nodes = []
        with self.driver.session() as session:
            offset = 0
            while True:
                result = session.run(
                    "MATCH (n:NODE {repoId: $repo_id}) "
                    "RETURN n.node_id AS node_id, n.text AS text, n.file_path AS file_path, n.start_line AS start_line, n.end_line AS end_line, n.name AS name "
                    "SKIP $offset LIMIT $limit",
                    repo_id=repo_id,
                    offset=offset,
                    limit=batch_size,
                )
                batch = [dict(record) for record in result]
                if not batch:
                    break
                all_nodes.extend(batch)
                offset += batch_size
        logger.info(f"DEBUGNEO4J: Fetched {len(all_nodes)} nodes for repo {repo_id}")
        return all_nodes

    def fetch_graph_by_directory(
        self, repo_id: str, directory_path: Optional[str] = None, is_root: bool = False
    ) -> List[Dict]:
        """
        Fetch nodes for a specific directory or root level.

        Args:
            repo_id: Repository ID
            directory_path: Directory path to filter nodes (e.g., "src/services")
            is_root: If True, fetch only root-level files (no subdirectories)

        Returns:
            List of node dictionaries with node_id, text, file_path, etc.

        Note:
            - When is_root=True, fetches files like "main.py", "setup.py" but NOT "src/main.py"
            - When directory_path is provided, fetches files within that directory
            - Uses proper Cypher filtering to avoid OOM issues
        """
        batch_size = 500
        all_nodes = []

        with self.driver.session() as session:
            offset = 0
            while True:
                # Build query based on parameters
                if is_root:
                    # Root directory: files without '/' in path
                    query = """
                        MATCH (n:NODE {repoId: $repo_id})
                        WHERE n.file_path IS NOT NULL
                          AND n.file_path <> ''
                          AND NOT n.file_path CONTAINS '/'
                        RETURN n.node_id AS node_id,
                               n.text AS text,
                               n.file_path AS file_path,
                               n.start_line AS start_line,
                               n.end_line AS end_line,
                               n.name AS name
                        SKIP $offset LIMIT $limit
                    """
                    params = {"repo_id": repo_id, "offset": offset, "limit": batch_size}

                elif directory_path:
                    # Specific directory: files that start with directory_path/
                    query = """
                        MATCH (n:NODE {repoId: $repo_id})
                        WHERE n.file_path IS NOT NULL
                          AND n.file_path STARTS WITH $directory_prefix
                        RETURN n.node_id AS node_id,
                               n.text AS text,
                               n.file_path AS file_path,
                               n.start_line AS start_line,
                               n.end_line AS end_line,
                               n.name AS name
                        SKIP $offset LIMIT $limit
                    """
                    # Ensure directory_path ends with / for proper prefix matching
                    directory_prefix = (
                        directory_path if directory_path.endswith("/")
                        else f"{directory_path}/"
                    )
                    params = {
                        "repo_id": repo_id,
                        "directory_prefix": directory_prefix,
                        "offset": offset,
                        "limit": batch_size,
                    }
                else:
                    # No filtering - fetch all (same as fetch_graph)
                    query = """
                        MATCH (n:NODE {repoId: $repo_id})
                        RETURN n.node_id AS node_id,
                               n.text AS text,
                               n.file_path AS file_path,
                               n.start_line AS start_line,
                               n.end_line AS end_line,
                               n.name AS name
                        SKIP $offset LIMIT $limit
                    """
                    params = {"repo_id": repo_id, "offset": offset, "limit": batch_size}

                result = session.run(query, **params)
                batch = [dict(record) for record in result]

                if not batch:
                    break

                all_nodes.extend(batch)
                offset += batch_size

        filter_desc = (
            "root level" if is_root
            else f"directory '{directory_path}'" if directory_path
            else "all"
        )
        logger.info(
            f"DEBUGNEO4J: Fetched {len(all_nodes)} nodes for repo {repo_id} "
            f"({filter_desc})"
        )
        return all_nodes

    async def process_nodes_streaming(
        self,
        repo_id: str,
        directory_path: Optional[str] = None,
        is_root: bool = False,
        chunk_size: int = 500,
        filter_uninferred: bool = False,
        use_inference_context: bool = True,
    ) -> Dict[str, DocstringResponse]:
        """
        Process nodes in streaming fashion to avoid OOM for large directories.

        This method:
        1. Fetches nodes in chunks from Neo4j
        2. Batches them for LLM processing (using inference_context for 85-90% token savings)
        3. Generates docstrings and embeddings
        4. Updates Neo4j immediately (no accumulation in memory)
        5. Creates search indices in batches (deferred commit)

        NEW: use_inference_context=True uses minimal context for 85-90% token savings.
        CRITICAL FIX #2: Conditionally fetches n.text only when context missing.

        Args:
            repo_id: Repository ID
            directory_path: Directory to process (None = all nodes)
            is_root: Whether this is root directory
            chunk_size: Number of nodes to fetch per DB query
            filter_uninferred: If True, skip nodes that already have docstrings
            use_inference_context: If True, use inference_context instead of full text

        Returns:
            Dict with summary statistics (not full docstrings to save memory)
        """
        logger.info(
            f"Starting streaming inference for repo {repo_id}, "
            f"directory={directory_path}, is_root={is_root}, "
            f"use_inference_context={use_inference_context}, filter_uninferred={filter_uninferred}"
        )

        # Auto-detect if project has inference_context (for backward compatibility)
        if use_inference_context:
            has_context = await self.check_has_inference_context(repo_id)
            if not has_context:
                logger.warning(
                    f"Project {repo_id} has no inference_context property on nodes. "
                    f"Falling back to full text mode (legacy project)."
                )
                use_inference_context = False

        total_nodes_processed = 0
        total_batches_processed = 0
        total_nodes_indexed = 0  # Track total for reporting
        failed_batches = []  # Track batch failures for reporting
        context_fallback_count = 0  # Track how many times we fell back to full text

        with self.driver.session() as session:
            offset = 0

            while True:
                # Build base query conditions
                base_conditions = ["n.file_path IS NOT NULL", "n.file_path <> ''"]

                if filter_uninferred:
                    base_conditions.append("n.docstring IS NULL")

                if is_root:
                    base_conditions.append("NOT n.file_path CONTAINS '/'")
                elif directory_path:
                    base_conditions.append("n.file_path STARTS WITH $directory_prefix")

                where_clause = " AND ".join(base_conditions)

                # CRITICAL FIX #2: Conditionally fetch text ONLY if context missing
                if use_inference_context:
                    # KEY FIX: Use CASE to fetch text only when context is NULL
                    query = f"""
                        MATCH (n:NODE {{repoId: $repo_id}})
                        WHERE {where_clause}
                        RETURN n.node_id AS node_id,
                               n.inference_context AS inference_context,
                               CASE
                                 WHEN n.inference_context IS NULL THEN n.text
                                 ELSE NULL
                               END AS text,
                               n.file_path AS file_path,
                               n.start_line AS start_line,
                               n.end_line AS end_line,
                               n.name AS name
                        SKIP $offset LIMIT $limit
                    """
                else:
                    # Legacy mode: fetch full text
                    query = f"""
                        MATCH (n:NODE {{repoId: $repo_id}})
                        WHERE {where_clause}
                        RETURN n.node_id AS node_id,
                               n.text AS text,
                               n.file_path AS file_path,
                               n.start_line AS start_line,
                               n.end_line AS end_line,
                               n.name AS name
                        SKIP $offset LIMIT $limit
                    """

                # Build params
                params = {"repo_id": repo_id, "offset": offset, "limit": chunk_size}
                if directory_path:
                    directory_prefix = (
                        directory_path if directory_path.endswith("/")
                        else f"{directory_path}/"
                    )
                    params["directory_prefix"] = directory_prefix

                result = session.run(query, **params)
                nodes_chunk = [dict(record) for record in result]

                if not nodes_chunk:
                    # No more nodes to process
                    break

                logger.info(
                    f"Processing chunk: offset={offset}, nodes_fetched={len(nodes_chunk)}"
                )

                # Batch nodes for LLM processing (with inference_context optimization)
                # Pass repo_id for fallback text fetching when context fails
                batches = self.batch_nodes(
                    nodes_chunk,
                    use_inference_context=use_inference_context,
                    repo_id=repo_id
                )

                # Process batches with semaphore for rate limiting
                semaphore = asyncio.Semaphore(self.parallel_requests)

                async def process_batch(batch, batch_index: int):
                    async with semaphore:
                        logger.info(
                            f"Processing batch {batch_index} for repo {repo_id}"
                        )
                        response = await self.generate_response(batch, repo_id)
                        if isinstance(response, DocstringResponse):
                            # Update Neo4j immediately (don't accumulate in memory)
                            self.update_neo4j_with_docstrings(repo_id, response)
                        else:
                            logger.warning(
                                f"Invalid response for batch {batch_index}, retrying..."
                            )
                            response = await self.generate_response(batch, repo_id)
                            if isinstance(response, DocstringResponse):
                                self.update_neo4j_with_docstrings(repo_id, response)
                        return response

                tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Track batch failures for reporting
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Batch {i} failed for repo {repo_id}: {result}",
                            exc_info=result
                        )
                        failed_batches.append({
                            'batch_index': i,
                            'error': str(result),
                            'error_type': type(result).__name__
                        })
                    elif not isinstance(result, DocstringResponse):
                        logger.warning(
                            f"Batch {i} returned invalid response for repo {repo_id}: {type(result)}"
                        )
                        failed_batches.append({
                            'batch_index': i,
                            'error': 'Invalid response type',
                            'error_type': 'InvalidResponseError'
                        })

                total_batches_processed += len(batches)

                # Create search indices for this chunk only (not accumulated)
                nodes_to_index_chunk = []
                for node in nodes_chunk:
                    if node.get("file_path") and node.get("name"):
                        nodes_to_index_chunk.append({
                            "project_id": repo_id,
                            "node_id": node["node_id"],
                            "name": node.get("name", ""),
                            "file_path": node.get("file_path", ""),
                            "content": f"{node.get('name', '')} {node.get('file_path', '')}",
                        })

                # Batch create search indices for this chunk (deferred commit)
                if nodes_to_index_chunk:
                    logger.debug(
                        f"Creating search indices for chunk: {len(nodes_to_index_chunk)} nodes "
                        f"(offset={offset}, auto_commit=False)"
                    )
                    await self.search_service.bulk_create_search_indices(
                        nodes_to_index_chunk, auto_commit=False
                    )
                    total_nodes_indexed += len(nodes_to_index_chunk)
                    del nodes_to_index_chunk  # Explicit cleanup

                total_nodes_processed += len(nodes_chunk)

                # Clear chunk from memory
                del nodes_chunk
                offset += chunk_size

        logger.info(
            f"Streaming inference complete: repo={repo_id}, "
            f"total_nodes={total_nodes_processed}, "
            f"total_batches={total_batches_processed}, "
            f"total_indexed={total_nodes_indexed}"
        )

        if failed_batches:
            logger.warning(
                f"Failed {len(failed_batches)}/{total_batches_processed} batches for repo {repo_id}: "
                f"{failed_batches}"
            )

        return {
            "total_nodes_processed": total_nodes_processed,
            "total_batches_processed": total_batches_processed,
            "nodes_indexed": total_nodes_indexed,
            "failed_batches": failed_batches,
            "batch_failure_rate": len(failed_batches) / total_batches_processed if total_batches_processed > 0 else 0.0,
            "used_inference_context": use_inference_context,
        }

    def get_entry_points(self, repo_id: str) -> List[str]:
        batch_size = 400  # Define the batch size
        all_entry_points = []
        with self.driver.session() as session:
            offset = 0
            while True:
                result = session.run(
                    f"""
                    MATCH (f:FUNCTION)
                    WHERE f.repoId = '{repo_id}'
                    AND NOT ()-[:CALLS]->(f)
                    AND (f)-[:CALLS]->()
                    RETURN f.node_id as node_id
                    SKIP $offset LIMIT $limit
                    """,
                    offset=offset,
                    limit=batch_size,
                )
                batch = result.data()
                if not batch:
                    break
                all_entry_points.extend([record["node_id"] for record in batch])
                offset += batch_size
        return all_entry_points

    def get_neighbours(self, node_id: str, repo_id: str):
        with self.driver.session() as session:
            batch_size = 400  # Define the batch size
            all_nodes_info = []
            offset = 0
            while True:
                result = session.run(
                    """
                    MATCH (start {node_id: $node_id, repoId: $repo_id})
                    OPTIONAL MATCH (start)-[:CALLS]->(direct_neighbour)
                    OPTIONAL MATCH (start)-[:CALLS]->()-[:CALLS*0..]->(indirect_neighbour)
                    WITH start, COLLECT(DISTINCT direct_neighbour) + COLLECT(DISTINCT indirect_neighbour) AS all_neighbours
                    UNWIND all_neighbours AS neighbour
                    WITH start, neighbour
                    WHERE neighbour IS NOT NULL AND neighbour <> start
                    RETURN DISTINCT neighbour.node_id AS node_id, neighbour.name AS function_name, labels(neighbour) AS labels
                    SKIP $offset LIMIT $limit
                    """,
                    node_id=node_id,
                    repo_id=repo_id,
                    offset=offset,
                    limit=batch_size,
                )
                batch = result.data()
                if not batch:
                    break
                all_nodes_info.extend(
                    [
                        record["node_id"]
                        for record in batch
                        if "FUNCTION" in record["labels"]
                    ]
                )
                offset += batch_size
            return all_nodes_info

    def get_entry_points_for_nodes(
        self, node_ids: List[str], repo_id: str
    ) -> Dict[str, List[str]]:
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $node_ids AS nodeId
                MATCH (n:FUNCTION)
                WHERE n.node_id = nodeId and n.repoId = $repo_id
                OPTIONAL MATCH path = (entryPoint)-[*]->(n)
                WHERE NOT (entryPoint)<--()
                RETURN n.node_id AS input_node_id, collect(DISTINCT entryPoint.node_id) AS entry_point_node_ids

                """,
                node_ids=node_ids,
                repo_id=repo_id,
            )
            return {
                record["input_node_id"]: (
                    record["entry_point_node_ids"]
                    if len(record["entry_point_node_ids"]) > 0
                    else [record["input_node_id"]]
                )
                for record in result
            }

    def batch_nodes(
        self, nodes: List[Dict], max_tokens: int = 16000, model: str = "gpt-4",
        use_inference_context: bool = True, repo_id: Optional[str] = None
    ) -> List[List[DocstringRequest]]:
        """
        Batch nodes for LLM processing.

        NEW: Supports inference_context for 85-90% token savings.
        When inference_context is available, formats it into a compact prompt.
        Falls back to full text when context is missing or invalid.

        Args:
            nodes: List of node dictionaries from Neo4j
            max_tokens: Maximum tokens per batch
            model: Model name for tokenization
            use_inference_context: Whether to use inference_context (default True)
            repo_id: Repository ID for fallback text fetching (required when use_inference_context=True)

        Returns:
            List of batches, each containing DocstringRequest objects
        """
        batches = []
        current_batch = []
        current_tokens = 0
        node_dict = {node["node_id"]: node for node in nodes}
        context_used_count = 0
        fallback_count = 0
        fetch_fallback_count = 0  # Track nodes that needed DB fetch
        skipped_count = 0

        def replace_referenced_text(
            text: str, node_dict: Dict[str, Dict[str, str]]
        ) -> str:
            pattern = r"Code replaced for brevity\. See node_id ([a-f0-9]+)"
            regex = re.compile(pattern)

            def replace_match(match):
                node_id = match.group(1)
                if node_id in node_dict and node_dict[node_id].get("text"):
                    return "\n" + node_dict[node_id]["text"].split("\n", 1)[-1]
                return match.group(0)

            previous_text = None
            current_text = text

            while previous_text != current_text:
                previous_text = current_text
                current_text = regex.sub(replace_match, current_text)
            return current_text

        # Collect nodes that need text fetched (context failed but no text in result)
        nodes_needing_fetch = []
        for node in nodes:
            if use_inference_context and node.get("inference_context"):
                try:
                    context = json.loads(node["inference_context"])
                    formatted = self._format_inference_context(context)
                    if not formatted and not node.get("text"):
                        # Context exists but formatting failed, and no text available
                        nodes_needing_fetch.append(node["node_id"])
                except Exception:
                    if not node.get("text"):
                        nodes_needing_fetch.append(node["node_id"])

        # Batch fetch text for nodes that need it (FIX #3 - actually use the fallback!)
        fetched_texts = {}
        if nodes_needing_fetch and repo_id:
            logger.info(f"Fetching full text for {len(nodes_needing_fetch)} nodes where context failed")
            fetched_texts = self._fetch_node_texts_batch(repo_id, nodes_needing_fetch)
            fetch_fallback_count = len(fetched_texts)

        for node in nodes:
            prompt_text = None
            node_id = node["node_id"]

            # Try to use inference_context first (85-90% token savings)
            if use_inference_context and node.get("inference_context"):
                try:
                    context = json.loads(node["inference_context"])
                    formatted = self._format_inference_context(context)
                    if formatted:
                        prompt_text = formatted
                        context_used_count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse inference_context for node {node_id}: {e}")

            # Fallback to full text if no valid context
            if not prompt_text:
                # First try text from query result
                text = node.get("text")

                # If no text in result, check batch-fetched texts
                if not text and node_id in fetched_texts:
                    text = fetched_texts[node_id]

                # If still no text, try individual fetch as last resort
                if not text and repo_id:
                    logger.debug(f"Individual fetch for node {node_id}")
                    text = self._fetch_node_text(repo_id, node_id)
                    if text:
                        fetch_fallback_count += 1

                if not text:
                    logger.warning(f"Node {node_id} has no text or context. Skipping...")
                    skipped_count += 1
                    continue

                prompt_text = replace_referenced_text(text, node_dict)
                fallback_count += 1

            node_tokens = self.num_tokens_from_string(prompt_text, model)

            if node_tokens > max_tokens:
                logger.warning(
                    f"Node {node_id} - {node_tokens} tokens, has exceeded the max_tokens limit. Skipping..."
                )
                skipped_count += 1
                continue

            if current_tokens + node_tokens > max_tokens:
                if current_batch:  # Only append if there are items
                    batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(
                DocstringRequest(node_id=node_id, text=prompt_text)
            )
            current_tokens += node_tokens

        if current_batch:
            batches.append(current_batch)

        # Log optimization stats
        total_processed = context_used_count + fallback_count
        if total_processed > 0:
            context_rate = context_used_count / total_processed * 100
            logger.info(
                f"Batch nodes: {context_used_count}/{total_processed} ({context_rate:.1f}%) "
                f"used inference_context, {fallback_count} used full text "
                f"({fetch_fallback_count} required DB fetch), {skipped_count} skipped"
            )

        total_nodes = sum(len(batch) for batch in batches)
        logger.info(f"Batched {total_nodes} nodes into {len(batches)} batches")
        logger.info(f"Batch sizes: {[len(batch) for batch in batches]}")

        return batches

    def _format_inference_context(self, context: Dict) -> Optional[str]:
        """
        Format inference context into compact LLM prompt (FIX #3).

        Converts extracted context into readable format for LLM.
        This is the magic that saves 85-90% of tokens!

        Args:
            context: Dict with signature, identifiers, operations, etc.

        Returns:
            Compact string prompt (~50 tokens vs ~500 for full code)
            OR None if context is insufficient (signals caller to fetch full text)
        """
        parts = []

        # Core signature (REQUIRED - FIX #3)
        if not context.get('signature'):
            logger.warning("Missing signature in inference_context")
            return None  # Signal caller to use fallback

        parts.append(context['signature'])

        # Class context (if method)
        if context.get('class_name'):
            parts.append(f"# Method of class: {context['class_name']}")

        # Existing docstring (style guide for LLM)
        if context.get('existing_docstring'):
            parts.append(f'"""{context["existing_docstring"]}"""')

        # First operations (what does it do?)
        if context.get('first_operations'):
            parts.append(f"# First operations: {context['first_operations']}")

        # Key identifiers (domain vocabulary)
        if context.get('key_identifiers') and len(context['key_identifiers']) > 0:
            ids = ', '.join(context['key_identifiers'][:5])
            parts.append(f"# Uses: {ids}")

        # Language-specific hints
        hints = []
        if context.get('is_async'):
            hints.append("async")
        if context.get('visibility') == 'private':
            hints.append("private")
        if context.get('decorators'):
            hints.append(f"decorators: {', '.join(context['decorators'][:2])}")

        if hints:
            parts.append(f"# Modifiers: {', '.join(hints)}")

        result = "\n".join(parts)

        # Validate we have meaningful content
        # Short signatures like "def x():" are valid - just check non-empty
        stripped = result.strip()
        if not stripped:
            logger.warning("Inference context is empty")
            return None

        return result

    async def check_has_inference_context(self, project_id: str) -> bool:
        """
        Check if project nodes have inference_context property (FIX #1).

        For backward compatibility with old parsed repos.
        Uses LIMIT 1 to short-circuit after finding first match.

        Improvement: O(1) instead of O(n) - returns after finding ANY node with context.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE {repoId: $repo_id})
                WHERE n.inference_context IS NOT NULL
                RETURN 1 LIMIT 1
                """,
                repo_id=project_id
            ).single()

            return result is not None

    def _fetch_node_text(self, repo_id: str, node_id: str) -> str:
        """
        Fetch n.text for a specific node (FIX #3 fallback).

        Used when inference_context exists but is invalid/empty.

        **Expected Hit Rate**: <5% of nodes (only when context is malformed)
        **Performance**: Single round-trip per node. For pathological batches with
                        many failures, consider using _fetch_node_texts_batch().

        Note: The repoId filter is kept for consistency, though node_id is globally
              unique. This ensures we only fetch nodes from the expected project.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE {node_id: $node_id})
                WHERE n.repoId = $repo_id
                RETURN n.text AS text
                """,
                repo_id=repo_id,
                node_id=node_id
            ).single()

            return result['text'] if result else ""

    def _fetch_node_texts_batch(self, repo_id: str, node_ids: List[str]) -> Dict[str, str]:
        """
        Batch fetch n.text for multiple nodes (FIX #8 - batching optimization).

        Use this when >10% of nodes in a chunk need fallback fetching.

        Args:
            repo_id: Project repository ID
            node_ids: List of node_ids to fetch

        Returns:
            Dict mapping node_id → text

        **When to Use**:
        - If >50 nodes per 500-node chunk need fallback → use this method
        - Otherwise, individual fetches are fine (expected <5% hit rate)
        """
        if not node_ids:
            return {}

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE)
                WHERE n.node_id IN $node_ids AND n.repoId = $repo_id
                RETURN n.node_id AS node_id, n.text AS text
                """,
                node_ids=node_ids,
                repo_id=repo_id
            )

            return {record['node_id']: record['text'] for record in result}

    async def generate_docstrings_for_entry_points(
        self,
        all_docstrings,
        entry_points_neighbors: Dict[str, List[str]],
    ) -> Dict[str, DocstringResponse]:
        docstring_lookup = {
            d.node_id: d.docstring for d in all_docstrings["docstrings"]
        }

        entry_point_batches = self.batch_entry_points(
            entry_points_neighbors, docstring_lookup
        )

        semaphore = asyncio.Semaphore(self.parallel_requests)

        async def process_batch(batch):
            async with semaphore:
                response = await self.generate_entry_point_response(batch)
                if isinstance(response, DocstringResponse):
                    return response
                else:
                    return await self.generate_docstrings_for_entry_points(
                        all_docstrings, entry_points_neighbors
                    )

        tasks = [process_batch(batch) for batch in entry_point_batches]
        results = await asyncio.gather(*tasks)

        updated_docstrings = DocstringResponse(docstrings=[])
        for result in results:
            updated_docstrings.docstrings.extend(result.docstrings)

        # Update all_docstrings with the new entry point docstrings
        for updated_docstring in updated_docstrings.docstrings:
            existing_index = next(
                (
                    i
                    for i, d in enumerate(all_docstrings["docstrings"])
                    if d.node_id == updated_docstring.node_id
                ),
                None,
            )
            if existing_index is not None:
                all_docstrings["docstrings"][existing_index] = updated_docstring
            else:
                all_docstrings["docstrings"].append(updated_docstring)

        return all_docstrings

    def batch_entry_points(
        self,
        entry_points_neighbors: Dict[str, List[str]],
        docstring_lookup: Dict[str, str],
        max_tokens: int = 16000,
        model: str = "gpt-4",
    ) -> List[List[Dict[str, str]]]:
        batches = []
        current_batch = []
        current_tokens = 0

        for entry_point, neighbors in entry_points_neighbors.items():
            entry_docstring = docstring_lookup.get(entry_point, "")
            neighbor_docstrings = [
                f"{neighbor}: {docstring_lookup.get(neighbor, '')}"
                for neighbor in neighbors
            ]
            flow_description = "\n".join(neighbor_docstrings)

            entry_point_data = {
                "node_id": entry_point,
                "entry_docstring": entry_docstring,
                "flow_description": entry_docstring + "\n" + flow_description,
            }

            entry_point_tokens = self.num_tokens_from_string(
                entry_docstring + flow_description, model
            )

            if entry_point_tokens > max_tokens:
                continue  # Skip entry points that exceed the max_tokens limit

            if current_tokens + entry_point_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(entry_point_data)
            current_tokens += entry_point_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def generate_entry_point_response(
        self, batch: List[Dict[str, str]]
    ) -> DocstringResponse:
        prompt = """
        You are an expert software architect with deep knowledge of distributed systems and cloud-native applications. Your task is to analyze entry points and their function flows in a codebase.

        For each of the following entry points and their function flows, perform the following task:

        1. **Flow Summary**: Generate a concise yet comprehensive summary of the overall intent and purpose of the entry point and its flow. Follow these guidelines:
           - Start with a high-level overview of the entry point's purpose.
           - Detail the main steps or processes involved in the flow.
           - Highlight key interactions with external systems or services.
           - Specify ALL API paths, HTTP methods, topic names, database interactions, and critical function calls.
           - Identify any error handling or edge cases.
           - Conclude with the expected output or result of the flow.

        Remember, the summary should be technical enough for a senior developer to understand the code's functionality via similarity search, but concise enough to be quickly parsed. Aim for a balance between detail and brevity.

        Your response must be a valid JSON object containing a list of docstrings, where each docstring object has:
        - node_id: The ID of the entry point being documented
        - docstring: A comprehensive flow summary following the guidelines above
        - tags: A list of relevant tags based on the functionality (e.g., ["API", "DATABASE"] for endpoints that interact with a database)

        Here are the entry points and their flows:

        {entry_points}
        """

        entry_points_text = "\n\n".join(
            [
                f"Entry point: {entry_point['node_id']}\n"
                f"Flow:\n{entry_point['flow_description']}"
                f"Entry docstring:\n{entry_point['entry_docstring']}"
                for entry_point in batch
            ]
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert software architecture documentation assistant. You will analyze code flows and provide structured documentation in JSON format.",
            },
            {"role": "user", "content": prompt.format(entry_points=entry_points_text)},
        ]

        try:
            result = await self.provider_service.call_llm_with_structured_output(
                messages=messages,
                output_schema=DocstringResponse,
                config_type="inference",
            )
            return result
        except Exception as e:
            logger.error(f"Entry point response generation failed: {e}")
            return DocstringResponse(docstrings=[])

    async def generate_docstrings(self, repo_id: str) -> Dict[str, DocstringResponse]:
        logger.info(
            f"DEBUGNEO4J: Function: {self.generate_docstrings.__name__}, Repo ID: {repo_id}"
        )
        self.log_graph_stats(repo_id)
        nodes = self.fetch_graph(repo_id)
        logger.info(
            f"DEBUGNEO4J: After fetch graph, Repo ID: {repo_id}, Nodes: {len(nodes)}"
        )
        self.log_graph_stats(repo_id)
        logger.info(
            f"Creating search indices for project {repo_id} with nodes count {len(nodes)}"
        )

        # Prepare a list of nodes for bulk insert
        nodes_to_index = [
            {
                "project_id": repo_id,
                "node_id": node["node_id"],
                "name": node.get("name", ""),
                "file_path": node.get("file_path", ""),
                "content": f"{node.get('name', '')} {node.get('file_path', '')}",
            }
            for node in nodes
            if node.get("file_path") not in {None, ""}
            and node.get("name") not in {None, ""}
        ]

        # Perform bulk insert
        await self.search_service.bulk_create_search_indices(nodes_to_index)

        logger.info(
            f"Project {repo_id}: Created search indices over {len(nodes_to_index)} nodes"
        )

        await self.search_service.commit_indices()
        # entry_points = self.get_entry_points(repo_id)
        # logger.info(
        #     f"DEBUGNEO4J: After get entry points, Repo ID: {repo_id}, Entry points: {len(entry_points)}"
        # )
        # self.log_graph_stats(repo_id)
        # entry_points_neighbors = {}
        # for entry_point in entry_points:
        #     neighbors = self.get_neighbours(entry_point, repo_id)
        #     entry_points_neighbors[entry_point] = neighbors

        # logger.info(
        #     f"DEBUGNEO4J: After get neighbours, Repo ID: {repo_id}, Entry points neighbors: {len(entry_points_neighbors)}"
        # )
        # self.log_graph_stats(repo_id)
        batches = self.batch_nodes(nodes)
        all_docstrings = {"docstrings": []}

        semaphore = asyncio.Semaphore(self.parallel_requests)

        async def process_batch(batch, batch_index: int):
            async with semaphore:
                logger.info(f"Processing batch {batch_index} for project {repo_id}")
                response = await self.generate_response(batch, repo_id)
                if not isinstance(response, DocstringResponse):
                    logger.warning(
                        f"Parsing project {repo_id}: Invalid response from LLM. Not an instance of DocstringResponse. Retrying..."
                    )
                    response = await self.generate_response(batch, repo_id)
                else:
                    self.update_neo4j_with_docstrings(repo_id, response)
                return response

        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        for result in results:
            if not isinstance(result, DocstringResponse):
                logger.error(
                    f"Project {repo_id}: Invalid response from during inference. Manually verify the project completion."
                )

        # updated_docstrings = await self.generate_docstrings_for_entry_points(
        #     all_docstrings, entry_points_neighbors
        # )
        updated_docstrings = all_docstrings
        return updated_docstrings

    async def generate_response(
        self, batch: List[DocstringRequest], repo_id: str
    ) -> DocstringResponse:
        base_prompt = """
        You are a senior software engineer with expertise in code analysis and documentation. Your task is to generate concise docstrings for each code snippet and tagging it based on its purpose. Approach this task methodically, following these steps:

        1. **Node Identification**:
        - Carefully parse the provided `code_snippets` to identify each `node_id` and its corresponding code block.
        - Ensure that every `node_id` present in the `code_snippets` is accounted for and processed individually.

        2. **For Each Node**:
        Perform the following tasks for every identified `node_id` and its associated code:

        You are a software engineer tasked with generating concise docstrings for each code snippet and tagging it based on its purpose.

        **Instructions**:
        2.1. **Identify Code Type**:
        - Determine whether each code snippet is primarily **backend** or **frontend**.
        - Use common indicators:
            - **Backend**: Handles database interactions, API endpoints, configuration, or server-side logic.
            - **Frontend**: Contains UI components, event handling, state management, or styling.

        2.2. **Summarize the Purpose**:
        - Based on the identified type, write a brief (1-2 sentences) summary of the code's main purpose and functionality.
        - Focus on what the code does, its role in the system, and any critical operations it performs.
        - If the code snippet is related to **specific roles** like authentication, database access, or UI component, state management, explicitly mention this role.

        2.3. **Assign Tags Based on Code Type**:
        - Use these specific tags based on whether the code is identified as backend or frontend:

        **Backend Tags**:
            - **AUTH**: Handles authentication or authorization.
            - **DATABASE**: Interacts with databases.
            - **API**: Defines API endpoints.
            - **UTILITY**: Provides helper or utility functions.
            - **PRODUCER**: Sends messages to a queue or topic.
            - **CONSUMER**: Processes messages from a queue or topic.
            - **EXTERNAL_SERVICE**: Integrates with external services.
            - **CONFIGURATION**: Manages configuration settings.

        **Frontend Tags**:
            - **UI_COMPONENT**: Renders a visual component in the UI.
            - **FORM_HANDLING**: Manages form data submission and validation.
            - **STATE_MANAGEMENT**: Manages application or component state.
            - **DATA_BINDING**: Binds data to UI elements.
            - **ROUTING**: Manages frontend navigation.
            - **EVENT_HANDLING**: Handles user interactions.
            - **STYLING**: Applies styling or theming.
            - **MEDIA**: Manages media, like images or video.
            - **ANIMATION**: Defines animations in the UI.
            - **ACCESSIBILITY**: Implements accessibility features.
            - **DATA_FETCHING**: Fetches data for frontend use.

        Your response must be a valid JSON object containing a list of docstrings, where each docstring object has:
        - node_id: The ID of the node being documented
        - docstring: A concise description of the code's purpose and functionality
        - tags: A list of relevant tags from the categories above

        Here are the code snippets:

        {code_snippets}
        """

        # Prepare the code snippets
        code_snippets = ""
        for request in batch:
            code_snippets += (
                f"node_id: {request.node_id} \n```\n{request.text}\n```\n\n "
            )

        messages = [
            {
                "role": "system",
                "content": "You are an expert software documentation assistant. You will analyze code and provide structured documentation in JSON format.",
            },
            {
                "role": "user",
                "content": base_prompt.format(code_snippets=code_snippets),
            },
        ]

        import time

        start_time = time.time()
        logger.info(f"Parsing project {repo_id}: Starting the inference process...")

        try:
            result = await self.provider_service.call_llm_with_structured_output(
                messages=messages,
                output_schema=DocstringResponse,
                config_type="inference",
            )
        except Exception as e:
            logger.error(
                f"Parsing project {repo_id}: Inference request failed. Error: {str(e)}"
            )
            result = DocstringResponse(docstrings=[])

        end_time = time.time()
        logger.info(
            f"Parsing project {repo_id}: Inference request completed. Time Taken: {end_time - start_time} seconds"
        )
        return result

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def update_neo4j_with_docstrings(self, repo_id: str, docstrings: DocstringResponse):
        with self.driver.session() as session:
            batch_size = 300
            docstring_list = [
                {
                    "node_id": n.node_id,
                    "docstring": n.docstring,
                    "tags": n.tags,
                    "embedding": self.generate_embedding(n.docstring),
                }
                for n in docstrings.docstrings
            ]
            for i in range(0, len(docstring_list), batch_size):
                batch = docstring_list[i : i + batch_size]
                # Note: We no longer remove n.text to enable:
                # - Inference retry without re-parsing
                # - Model upgrades and re-generation
                # - Debugging and quality validation
                # The inference_context provides 85-90% token savings anyway
                session.run(
                    """
                    UNWIND $batch AS item
                    MATCH (n:NODE {repoId: $repo_id, node_id: item.node_id})
                    SET n.docstring = item.docstring,
                        n.embedding = item.embedding,
                        n.tags = item.tags
                    """,
                    batch=batch,
                    repo_id=repo_id,
                )

    def create_vector_index(self):
        with self.driver.session() as session:
            session.run(
                """
                CREATE VECTOR INDEX docstring_embedding IF NOT EXISTS
                FOR (n:NODE)
                ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
                """
            )

    async def run_inference(self, repo_id: str):
        docstrings = await self.generate_docstrings(repo_id)
        logger.info(
            f"DEBUGNEO4J: After generate docstrings, Repo ID: {repo_id}, Docstrings: {len(docstrings)}"
        )
        self.log_graph_stats(repo_id)
        self.create_vector_index()

    def query_vector_index(
        self,
        project_id: str,
        query: str,
        node_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        embedding = self.generate_embedding(query)

        with self.driver.session() as session:
            if node_ids:
                # Fetch context node IDs
                result_neighbors = session.run(
                    """
                    MATCH (n:NODE)
                    WHERE n.repoId = $project_id AND n.node_id IN $node_ids
                    CALL {
                        WITH n
                        MATCH (n)-[*1..4]-(neighbor:NODE)
                        RETURN COLLECT(DISTINCT neighbor.node_id) AS neighbor_ids
                    }
                    RETURN COLLECT(DISTINCT n.node_id) + REDUCE(acc = [], neighbor_ids IN COLLECT(neighbor_ids) | acc + neighbor_ids) AS context_node_ids
                    """,
                    project_id=project_id,
                    node_ids=node_ids,
                )
                context_node_ids = result_neighbors.single()["context_node_ids"]

                # Use vector index and filter by context_node_ids
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('docstring_embedding', $initial_k, $embedding)
                    YIELD node, score
                    WHERE node.repoId = $project_id AND node.node_id IN $context_node_ids
                    RETURN node.node_id AS node_id,
                        node.docstring AS docstring,
                        node.file_path AS file_path,
                        node.start_line AS start_line,
                        node.end_line AS end_line,
                        score AS similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                    """,
                    project_id=project_id,
                    embedding=embedding,
                    context_node_ids=context_node_ids,
                    initial_k=top_k * 10,  # Adjust as needed
                    top_k=top_k,
                )
            else:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('docstring_embedding', $top_k, $embedding)
                    YIELD node, score
                    WHERE node.repoId = $project_id
                    RETURN node.node_id AS node_id,
                        node.docstring AS docstring,
                        node.file_path AS file_path,
                        node.start_line AS start_line,
                        node.end_line AS end_line,
                        score AS similarity
                    """,
                    project_id=project_id,
                    embedding=embedding,
                    top_k=top_k,
                )

            # Ensure all fields are included in the final output
            return [dict(record) for record in result]

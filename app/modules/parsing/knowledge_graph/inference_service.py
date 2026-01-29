import asyncio
import os
import re
import time
from typing import Any, Dict, List, Optional

import tiktoken
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.core.database import get_db
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.parsing.knowledge_graph.inference_schema import (
    DocstringRequest,
    DocstringResponse,
)
from app.modules.parsing.services.inference_cache_service import InferenceCacheService
from app.modules.parsing.utils.content_hash import (
    generate_content_hash,
    is_content_cacheable,
)
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

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

        self.db = db
        self.provider_service = ProviderService(db, user_id if user_id else "dummy")
        self.embedding_model = get_embedding_model()  # Use singleton to avoid reloading
        self.search_service = SearchService(db)
        self.project_manager = ProjectService(db)
        self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", 50))

    def close(self):
        self.driver.close()

    def _get_cache_service(self) -> Optional[InferenceCacheService]:
        """Get cache service using the instance's DB session."""
        try:
            return InferenceCacheService(self.db)
        except Exception as e:
            logger.warning(f"Failed to initialize cache service: {e}")
            return None

    def _normalize_node_text(
        self, text: str, node_dict: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Normalize node text by resolving references for consistent hashing.

        This ensures the same code content produces the same hash across parses,
        even if referenced nodes have different node_ids.
        """
        if text is None:
            return ""

        pattern = r"Code replaced for brevity\. See node_id ([a-f0-9]+)"
        regex = re.compile(pattern)

        def replace_match(match):
            node_id = match.group(1)
            if node_id in node_dict and node_dict[node_id].get("text"):
                # Return full text of referenced node for consistent cache hashing
                return node_dict[node_id]["text"]
            else:
                # Normalize unresolved references to remove node_id dependency
                return "Code replaced for brevity. See node_id <REFERENCE>"

        previous_text = None
        current_text = text

        # Recursively resolve nested references
        while previous_text != current_text:
            previous_text = current_text
            current_text = regex.sub(replace_match, current_text)

        return current_text

    def _lookup_cache_for_nodes(
        self,
        nodes: List[Dict],
        node_dict: Dict[str, Dict[str, Any]],
        cache_service: InferenceCacheService,
        project_id: str,
    ) -> Dict[str, Any]:
        """
        Look up cache for all nodes and mark them with cache hit/miss status.

        Mutates nodes in place, adding:
        - cached_inference: The cached inference data (if hit)
        - content_hash: The hash of normalized content
        - should_cache: Whether to cache the inference result
        - normalized_text: The normalized text for LLM processing

        Returns cache statistics.
        """
        cache_hits = 0
        cache_misses = 0
        uncacheable_nodes = 0
        cache_lookup_time = 0.0

        lookup_start = time.time()

        for node in nodes:
            text = node.get("text")
            if not text:
                continue

            # Normalize text for consistent hashing
            normalized_text = self._normalize_node_text(text, node_dict)
            node["normalized_text"] = normalized_text

            # Check if content is cacheable
            if not is_content_cacheable(normalized_text):
                uncacheable_nodes += 1
                continue

            # Generate content hash
            content_hash = generate_content_hash(normalized_text, node.get("node_type"))
            node["content_hash"] = content_hash

            # Look up in cache
            node_lookup_start = time.time()
            cached_inference = cache_service.get_cached_inference(content_hash)
            cache_lookup_time += time.time() - node_lookup_start

            if cached_inference:
                # Cache hit - store the inference data on the node
                node["cached_inference"] = cached_inference
                cache_hits += 1
                # Verify the assignment worked
                if not node.get("cached_inference"):
                    logger.error(
                        f"CACHE BUG: cached_inference not set after assignment! node={node['node_id'][:8]}"
                    )
                logger.debug(
                    f"✅ CACHE HIT | node={node['node_id'][:8]} | "
                    f"hash={content_hash[:12]} | type={node.get('node_type', 'UNKNOWN')}"
                )
            else:
                # Cache miss - mark for caching after LLM inference
                node["should_cache"] = True
                cache_misses += 1
                logger.debug(
                    f"❌ CACHE MISS | node={node['node_id'][:8]} | "
                    f"hash={content_hash[:12]} | type={node.get('node_type', 'UNKNOWN')}"
                )

        total_lookup_time = time.time() - lookup_start
        total_cacheable = cache_hits + cache_misses
        hit_rate = (cache_hits / total_cacheable * 100) if total_cacheable > 0 else 0

        logger.info(
            f"[CACHE LOOKUP] Completed cache lookup for {len(nodes)} nodes: "
            f"Hits: {cache_hits} ({hit_rate:.1f}%), Misses: {cache_misses}, "
            f"Uncacheable: {uncacheable_nodes}, Total time: {total_lookup_time:.2f}s",
            project_id=project_id,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            uncacheable_nodes=uncacheable_nodes,
            cache_hit_rate=hit_rate,
        )

        return {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "uncacheable_nodes": uncacheable_nodes,
            "total_nodes": len(nodes),
            "cache_lookup_time": cache_lookup_time,
            "cache_hit_rate": hit_rate,
        }

    def _store_inference_in_cache(
        self,
        cache_service: InferenceCacheService,
        content_hash: str,
        docstring: str,
        tags: List[str],
        embedding: List[float],
        project_id: str,
        node_type: Optional[str] = None,
        content_length: Optional[int] = None,
    ) -> bool:
        """Store inference result in cache. Returns True if successful."""
        try:
            inference_data = {
                "docstring": docstring,
                "tags": tags,
            }
            cache_service.store_inference(
                content_hash=content_hash,
                inference_data=inference_data,
                project_id=project_id,
                node_type=node_type,
                content_length=content_length,
                embedding_vector=embedding,
                tags=tags,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to store inference in cache: {e}")
            return False

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

    # Class-level cache for tiktoken encodings
    _encoding_cache: Dict[str, Any] = {}

    def num_tokens_from_string(self, string: str, model: str = "gpt-4") -> int:
        """Returns the number of tokens in a text string."""
        # Handle None or empty strings gracefully
        if string is None:
            return 0
        if not isinstance(string, str):
            logger.warning(
                f"Expected string, got {type(string)}. Converting to string."
            )
            string = str(string)

        # Cache the encoding to avoid repeated model lookups
        if model not in InferenceService._encoding_cache:
            try:
                InferenceService._encoding_cache[model] = tiktoken.encoding_for_model(
                    model
                )
            except KeyError:
                logger.warning("Warning: model not found. Using cl100k_base encoding.")
                InferenceService._encoding_cache[model] = tiktoken.get_encoding(
                    "cl100k_base"
                )

        encoding = InferenceService._encoding_cache[model]
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

    def split_large_node(
        self, node_text: str, node_id: str, max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Split large nodes into processable chunks with context preservation.

        Uses incremental token counting for O(n) performance instead of O(n²).
        """
        model = "gpt-4"
        max_chunk_tokens = max_tokens // 2  # Reserve space for prompt

        lines = node_text.split("\n")
        chunks = []
        current_chunk_lines = []
        current_tokens = 0

        # Token overhead for newlines (approximately 1 token per newline)
        NEWLINE_OVERHEAD = 1

        for line in lines:
            # Count tokens for just this line (O(1) per line instead of O(n))
            line_tokens = self.num_tokens_from_string(line, model)

            # Estimate total: current + new line + newline overhead
            estimated_total = current_tokens + line_tokens + NEWLINE_OVERHEAD

            if estimated_total > max_chunk_tokens and current_chunk_lines:
                # Save current chunk and start new one
                chunks.append(
                    {
                        "text": "\n".join(current_chunk_lines),
                        "node_id": f"{node_id}_chunk_{len(chunks)}",
                        "is_chunk": True,
                        "parent_node_id": node_id,
                        "chunk_index": len(chunks),
                    }
                )
                current_chunk_lines = [line]
                current_tokens = line_tokens
            else:
                current_chunk_lines.append(line)
                current_tokens = estimated_total

        # Add final chunk
        if current_chunk_lines:
            chunks.append(
                {
                    "text": "\n".join(current_chunk_lines),
                    "node_id": f"{node_id}_chunk_{len(chunks)}",
                    "is_chunk": True,
                    "parent_node_id": node_id,
                    "chunk_index": len(chunks),
                }
            )

        return chunks

    def consolidate_chunk_responses(
        self, chunk_responses: List[DocstringResponse], parent_node_id: str
    ) -> DocstringResponse:
        """Consolidate multiple chunk docstring responses into a single parent node response"""
        if not chunk_responses:
            return DocstringResponse(docstrings=[])

        # Collect all chunk docstrings
        all_docstrings = []
        all_tags = set()

        for response in chunk_responses:
            for docstring in response.docstrings:
                all_docstrings.append(docstring.docstring)
                all_tags.update(docstring.tags or [])

        # Create consolidated docstring
        if len(all_docstrings) == 1:
            consolidated_text = all_docstrings[0]
        else:
            # Combine multiple chunk descriptions intelligently
            consolidated_text = f"This is a large code component split across {len(all_docstrings)} sections: "
            consolidated_text += " | ".join(
                [f"Section {i + 1}: {doc}" for i, doc in enumerate(all_docstrings)]
            )

        # Create single consolidated docstring for parent node
        from app.modules.parsing.knowledge_graph.inference_schema import DocstringNode

        consolidated_docstring = DocstringNode(
            node_id=parent_node_id, docstring=consolidated_text, tags=list(all_tags)
        )

        return DocstringResponse(docstrings=[consolidated_docstring])

    def process_chunk_responses(
        self, response: DocstringResponse, batch: List[DocstringRequest]
    ) -> Optional[DocstringResponse]:
        """Process chunk responses and consolidate them by parent node"""
        # Separate chunk responses from regular responses
        chunk_responses = {}
        regular_responses = []

        for docstring in response.docstrings:
            # Find the corresponding request to get metadata
            request = next(
                (req for req in batch if req.node_id == docstring.node_id), None
            )
            if request and request.metadata and request.metadata.get("is_chunk"):
                parent_id = request.metadata.get("parent_node_id")
                if parent_id:
                    if parent_id not in chunk_responses:
                        chunk_responses[parent_id] = []
                    chunk_responses[parent_id].append(docstring)
            else:
                regular_responses.append(docstring)

        # If no chunks, return original response
        if not chunk_responses:
            return response

        # Consolidate chunk responses
        consolidated_responses = []

        for parent_id, chunk_docstrings in chunk_responses.items():
            # Create a mock response list for consolidation
            mock_responses = []
            for chunk_doc in chunk_docstrings:
                from app.modules.parsing.knowledge_graph.inference_schema import (
                    DocstringResponse,
                )

                mock_responses.append(DocstringResponse(docstrings=[chunk_doc]))

            consolidated = self.consolidate_chunk_responses(mock_responses, parent_id)
            consolidated_responses.extend(consolidated.docstrings)

        # Add regular (non-chunk) responses
        consolidated_responses.extend(regular_responses)

        return DocstringResponse(docstrings=consolidated_responses)

    def _create_batches_from_nodes(
        self,
        nodes: List[Dict],
        max_tokens: int = 200000,  # Increased but still conservative (context window is 272k, prompt overhead ~2-3k)
        model: str = "gpt-4",
        project_id: Optional[str] = None,
    ) -> List[List[DocstringRequest]]:
        """
        Create LLM batches from nodes that need inference (cache misses).

        Only processes nodes that don't have cached_inference set.
        Uses normalized_text if available, otherwise normalizes on the fly.
        """
        batch_start_time = time.time()
        node_dict = {node["node_id"]: node for node in nodes}

        # Filter to nodes that need LLM inference:
        # - No cache hit (cached_inference not set)
        # - Has text content
        # - Is cacheable (skip small/trivial nodes that aren't worth processing)
        nodes_with_cached = sum(1 for n in nodes if n.get("cached_inference"))
        nodes_with_text = sum(1 for n in nodes if n.get("text"))

        # Skip uncacheable nodes - they're typically too small or have unresolved references
        # Only process nodes that have should_cache=True (cacheable but cache miss)
        # OR have content_hash set (cacheable)
        nodes_needing_inference = [
            n
            for n in nodes
            if not n.get("cached_inference")
            and n.get("text")
            and (n.get("should_cache") or n.get("content_hash"))  # Only cacheable nodes
        ]

        nodes_skipped_uncacheable = sum(
            1
            for n in nodes
            if not n.get("cached_inference")
            and n.get("text")
            and not n.get("should_cache")
            and not n.get("content_hash")
        )

        logger.info(
            f"[BATCHING] Creating batches for {len(nodes_needing_inference)} nodes "
            f"(total: {len(nodes)}, cached: {nodes_with_cached}, with text: {nodes_with_text}, "
            f"skipped uncacheable: {nodes_skipped_uncacheable})",
            project_id=project_id,
            nodes_total=len(nodes),
            nodes_with_cached=nodes_with_cached,
            nodes_with_text=nodes_with_text,
            nodes_needing_inference=len(nodes_needing_inference),
            nodes_skipped_uncacheable=nodes_skipped_uncacheable,
        )

        batches = []
        current_batch = []
        current_tokens = 0

        for node in nodes_needing_inference:
            # Use normalized text if available, otherwise normalize now
            text = node.get("normalized_text")
            if not text:
                text = self._normalize_node_text(node.get("text", ""), node_dict)
                node["normalized_text"] = text

            node_tokens = self.num_tokens_from_string(text, model)

            # Handle large nodes by splitting
            if node_tokens > max_tokens:
                logger.debug(
                    f"Node {node['node_id'][:8]} exceeds token limit ({node_tokens}). Splitting..."
                )
                node_chunks = self.split_large_node(text, node["node_id"], max_tokens)

                for chunk in node_chunks:
                    chunk_tokens = self.num_tokens_from_string(chunk["text"], model)

                    if current_tokens + chunk_tokens > max_tokens:
                        if current_batch:
                            batches.append(current_batch)
                        current_batch = []
                        current_tokens = 0

                    current_batch.append(
                        DocstringRequest(
                            node_id=chunk["node_id"],
                            text=chunk["text"],
                            metadata={
                                "is_chunk": True,
                                "parent_node_id": chunk["parent_node_id"],
                                "chunk_index": chunk.get("chunk_index", 0),
                                "should_cache": node.get("should_cache", False),
                                "content_hash": node.get("content_hash"),
                                "node_type": node.get("node_type"),
                            },
                        )
                    )
                    current_tokens += chunk_tokens
                continue

            # Add to batch
            if current_tokens + node_tokens > max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(
                DocstringRequest(
                    node_id=node["node_id"],
                    text=text,
                    metadata={
                        "should_cache": node.get("should_cache", False),
                        "content_hash": node.get("content_hash"),
                        "node_type": node.get("node_type"),
                    },
                )
            )
            current_tokens += node_tokens

        if current_batch:
            batches.append(current_batch)

        batch_time = time.time() - batch_start_time
        total_batched = sum(len(b) for b in batches)

        logger.info(
            f"[BATCHING] Created {len(batches)} batches with {total_batched} nodes in {batch_time:.2f}s",
            project_id=project_id,
            batch_count=len(batches),
            total_batched=total_batched,
        )

        return batches

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
                # Safety check: only append if current_batch has items
                if current_batch:
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

    async def generate_docstrings(
        self, repo_id: str
    ) -> tuple[Dict[str, DocstringResponse], Dict[str, Any]]:
        inference_start_time = time.time()
        logger.info(
            f"[INFERENCE] Starting docstring generation for project {repo_id}",
            project_id=repo_id,
        )
        self.log_graph_stats(repo_id)

        # Initialize cache service once for the entire inference process
        cache_service = self._get_cache_service()
        if cache_service:
            logger.info(
                "[INFERENCE] Cache service initialized successfully",
                project_id=repo_id,
            )
        else:
            logger.warning(
                "[INFERENCE] Cache service unavailable, proceeding without caching",
                project_id=repo_id,
            )

        # Step 1: Fetch graph nodes
        fetch_start = time.time()
        logger.info(
            f"[INFERENCE] Step 1/6: Fetching graph nodes from Neo4j",
            project_id=repo_id,
        )
        nodes = self.fetch_graph(repo_id)
        fetch_time = time.time() - fetch_start
        logger.info(
            f"[INFERENCE] Fetched {len(nodes)} nodes from graph in {fetch_time:.2f}s",
            project_id=repo_id,
            node_count=len(nodes),
            fetch_time_seconds=fetch_time,
        )
        self.log_graph_stats(repo_id)

        # Step 2: Create search indices
        search_index_start = time.time()
        logger.info(
            f"[INFERENCE] Step 2/6: Creating search indices for {len(nodes)} nodes",
            project_id=repo_id,
            node_count=len(nodes),
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
        search_index_time = time.time() - search_index_start
        logger.info(
            f"[INFERENCE] Created search indices over {len(nodes_to_index)} nodes in {search_index_time:.2f}s",
            project_id=repo_id,
            indexed_nodes=len(nodes_to_index),
            search_index_time_seconds=search_index_time,
        )

        commit_start = time.time()
        await self.search_service.commit_indices()
        commit_time = time.time() - commit_start
        logger.info(
            f"[INFERENCE] Committed search indices in {commit_time:.2f}s",
            project_id=repo_id,
            commit_time_seconds=commit_time,
        )

        # Step 3: Cache lookup - check which nodes have cached inference
        cache_lookup_start = time.time()
        logger.info(
            f"[INFERENCE] Step 3/6: Looking up cache for {len(nodes)} nodes",
            project_id=repo_id,
        )

        node_dict = {node["node_id"]: node for node in nodes}
        cache_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "uncacheable_nodes": 0,
            "total_nodes": len(nodes),
        }

        if cache_service:
            cache_stats = self._lookup_cache_for_nodes(
                nodes, node_dict, cache_service, repo_id
            )
        else:
            # Mark all cacheable nodes for caching when service is available
            for node in nodes:
                if node.get("text"):
                    normalized_text = self._normalize_node_text(
                        node.get("text", ""), node_dict
                    )
                    node["normalized_text"] = normalized_text
                    if is_content_cacheable(normalized_text):
                        node["content_hash"] = generate_content_hash(
                            normalized_text, node.get("node_type")
                        )
                        node["should_cache"] = True
                        cache_stats["cache_misses"] += 1
                    else:
                        cache_stats["uncacheable_nodes"] += 1

        cache_lookup_time = time.time() - cache_lookup_start

        # Verify cache state after lookup
        nodes_with_cached_inference = sum(1 for n in nodes if n.get("cached_inference"))
        nodes_with_should_cache = sum(1 for n in nodes if n.get("should_cache"))

        logger.info(
            f"[INFERENCE] Cache lookup completed in {cache_lookup_time:.2f}s | "
            f"Stats: hits={cache_stats.get('cache_hits', 0)}, misses={cache_stats.get('cache_misses', 0)} | "
            f"Verification: nodes_with_cached_inference={nodes_with_cached_inference}, nodes_with_should_cache={nodes_with_should_cache}",
            project_id=repo_id,
            cache_lookup_time_seconds=cache_lookup_time,
            nodes_with_cached_inference=nodes_with_cached_inference,
            nodes_with_should_cache=nodes_with_should_cache,
        )

        # Step 4: Create batches for nodes that need LLM inference
        batch_start = time.time()
        logger.info(
            f"[INFERENCE] Step 4/6: Creating batches for LLM inference",
            project_id=repo_id,
        )
        batches = self._create_batches_from_nodes(nodes, project_id=repo_id)
        batch_time = time.time() - batch_start
        logger.info(
            f"[INFERENCE] Created {len(batches)} batches in {batch_time:.2f}s",
            project_id=repo_id,
            batch_count=len(batches),
            batch_time_seconds=batch_time,
        )

        # Step 5: Process cached nodes (batch update Neo4j with cached inference)
        cached_process_start = time.time()
        cached_nodes = [node for node in nodes if node.get("cached_inference")]
        logger.info(
            f"[INFERENCE] Step 5/6: Batch processing {len(cached_nodes)} cached nodes",
            project_id=repo_id,
        )

        # Use batch update for much better performance (single DB call instead of N calls)
        cached_updated = self.batch_update_neo4j_with_cached_inference(
            cached_nodes, repo_id
        )

        cached_process_time = time.time() - cached_process_start
        logger.info(
            f"[INFERENCE] Batch processed {cached_updated} cached nodes in {cached_process_time:.2f}s",
            project_id=repo_id,
            cached_nodes_count=cached_updated,
            cached_process_time_seconds=cached_process_time,
        )

        all_docstrings = {"docstrings": []}
        total_cache_stored_count = 0

        # Step 6: Process LLM batches and store results in cache
        llm_batch_start = time.time()
        logger.info(
            f"[INFERENCE] Step 6/6: Processing {len(batches)} LLM batches",
            project_id=repo_id,
            batch_count=len(batches),
        )

        semaphore = asyncio.Semaphore(self.parallel_requests)

        batch_timings = []
        cache_store_times = []
        total_cache_stored_count = 0

        async def process_batch(batch, batch_index: int):
            nonlocal total_cache_stored_count
            async with semaphore:
                batch_process_start = time.time()
                logger.info(
                    f"[INFERENCE] Processing LLM batch {batch_index + 1}/{len(batches)} ({len(batch)} nodes)",
                    project_id=repo_id,
                    batch_index=batch_index + 1,
                    total_batches=len(batches),
                    batch_size=len(batch),
                )
                try:
                    # Generate inference for batch
                    llm_start = time.time()
                    response = await self.generate_response(batch, repo_id)
                    llm_time = time.time() - llm_start

                    if not isinstance(response, DocstringResponse):
                        logger.warning(
                            f"[INFERENCE] Invalid response from LLM for batch {batch_index + 1}. Retrying...",
                            project_id=repo_id,
                            batch_index=batch_index + 1,
                        )
                        llm_retry_start = time.time()
                        response = await self.generate_response(batch, repo_id)
                        llm_time += time.time() - llm_retry_start

                    if isinstance(response, DocstringResponse):
                        # Store results in cache and Neo4j
                        cache_store_start = time.time()
                        batch_cache_stored = 0

                        # Pre-generate embeddings for all docstrings in batch
                        # This allows us to cache embeddings and reuse them in Neo4j update
                        docstring_embeddings = {}
                        for docstring_result in response.docstrings:
                            embedding = self.generate_embedding(
                                docstring_result.docstring
                            )
                            docstring_embeddings[docstring_result.node_id] = embedding

                        for request, docstring_result in zip(
                            batch, response.docstrings
                        ):
                            metadata = request.metadata or {}
                            embedding = docstring_embeddings.get(
                                docstring_result.node_id
                            )

                            # Store in cache if eligible (includes embedding for future reuse)
                            if (
                                cache_service
                                and metadata.get("should_cache")
                                and metadata.get("content_hash")
                            ):
                                try:
                                    store_start = time.time()
                                    # Convert DocstringResult to dictionary for caching
                                    inference_data = {
                                        "node_id": docstring_result.node_id,
                                        "docstring": docstring_result.docstring,
                                        "tags": docstring_result.tags,
                                    }

                                    # Store with embedding for future cache hits
                                    cache_service.store_inference(
                                        content_hash=metadata["content_hash"],
                                        inference_data=inference_data,
                                        project_id=repo_id,
                                        node_type=metadata.get("node_type"),
                                        content_length=len(request.text),
                                        embedding_vector=embedding,
                                        tags=docstring_result.tags,
                                    )
                                    cache_store_times.append(time.time() - store_start)
                                    batch_cache_stored += 1
                                    total_cache_stored_count += 1
                                except Exception as cache_error:
                                    logger.warning(
                                        f"[INFERENCE] Failed to cache inference for node {request.node_id}: {cache_error}",
                                        project_id=repo_id,
                                        node_id=request.node_id,
                                    )

                        cache_store_time = time.time() - cache_store_start

                        # Handle chunk consolidation before Neo4j update
                        neo4j_update_start = time.time()
                        processed_response = self.process_chunk_responses(
                            response, batch
                        )
                        if processed_response:
                            # Pass pre-generated embeddings to avoid regenerating
                            self.update_neo4j_with_docstrings(
                                repo_id, processed_response, docstring_embeddings
                            )
                        neo4j_update_time = time.time() - neo4j_update_start

                        batch_process_time = time.time() - batch_process_start
                        batch_timings.append(batch_process_time)

                        logger.info(
                            f"[INFERENCE] Completed batch {batch_index + 1}/{len(batches)}: "
                            f"LLM: {llm_time:.2f}s, "
                            f"Cache store: {cache_store_time:.3f}s ({batch_cache_stored} stored in this batch), "
                            f"Neo4j update: {neo4j_update_time:.2f}s, "
                            f"Total: {batch_process_time:.2f}s",
                            project_id=repo_id,
                            batch_index=batch_index + 1,
                            llm_time_seconds=llm_time,
                            cache_store_time_seconds=cache_store_time,
                            batch_cache_stored=batch_cache_stored,
                            neo4j_update_time_seconds=neo4j_update_time,
                            batch_process_time_seconds=batch_process_time,
                        )

                    return response

                except Exception as e:
                    batch_process_time = time.time() - batch_process_start
                    logger.error(
                        f"[INFERENCE] Failed to process batch {batch_index + 1}: {e} (took {batch_process_time:.2f}s)",
                        project_id=repo_id,
                        batch_index=batch_index + 1,
                        batch_process_time_seconds=batch_process_time,
                    )
                    # Continue with next batch instead of failing entire operation
                    return DocstringResponse(docstrings=[])

        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)
        llm_batch_time = time.time() - llm_batch_start

        avg_batch_time = sum(batch_timings) / len(batch_timings) if batch_timings else 0
        avg_cache_store_time = (
            sum(cache_store_times) / len(cache_store_times) if cache_store_times else 0
        )

        logger.info(
            f"[INFERENCE] Completed LLM batch processing: "
            f"{len(batches)} batches in {llm_batch_time:.2f}s, "
            f"avg batch time: {avg_batch_time:.2f}s, "
            f"cached {total_cache_stored_count} results (avg store: {avg_cache_store_time*1000:.1f}ms)",
            project_id=repo_id,
            batch_count=len(batches),
            llm_batch_time_seconds=llm_batch_time,
            avg_batch_time_seconds=avg_batch_time,
            total_cache_stored_count=total_cache_stored_count,
            avg_cache_store_time_ms=avg_cache_store_time * 1000,
        )

        # Validate results
        validation_start = time.time()
        invalid_results = 0
        for result in results:
            if not isinstance(result, DocstringResponse):
                invalid_results += 1
                logger.error(
                    f"[INFERENCE] Invalid response during inference. Manually verify project completion.",
                    project_id=repo_id,
                )
        validation_time = time.time() - validation_start

        updated_docstrings = all_docstrings

        # Update cache stats with storage info
        cache_stats["cache_stored"] = total_cache_stored_count
        cache_stats["cached_nodes_processed"] = len(cached_nodes)

        total_inference_time = time.time() - inference_start_time

        logger.info(
            f"[INFERENCE] Docstring generation completed in {total_inference_time:.2f}s: "
            f"Fetch: {fetch_time:.2f}s, "
            f"Search index: {search_index_time:.2f}s, "
            f"Cache lookup: {cache_lookup_time:.2f}s, "
            f"Batching: {batch_time:.2f}s, "
            f"Cached processing: {cached_process_time:.2f}s, "
            f"LLM batches: {llm_batch_time:.2f}s, "
            f"Validation: {validation_time:.2f}s",
            project_id=repo_id,
            total_inference_time_seconds=total_inference_time,
            fetch_time_seconds=fetch_time,
            search_index_time_seconds=search_index_time,
            cache_lookup_time_seconds=cache_lookup_time,
            batch_time_seconds=batch_time,
            cached_process_time_seconds=cached_process_time,
            llm_batch_time_seconds=llm_batch_time,
            validation_time_seconds=validation_time,
            invalid_results=invalid_results,
        )

        return updated_docstrings, cache_stats

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

        # Build full prompt to check token count
        system_message = "You are an expert software documentation assistant. You will analyze code and provide structured documentation in JSON format."
        user_content = base_prompt.format(code_snippets=code_snippets)

        # Check total token count before sending (context window is typically 272k, but we should be conservative)
        # Account for response tokens and overhead
        MAX_CONTEXT_TOKENS = 250000  # Conservative limit, leaving room for response
        model = "gpt-4"  # Default model for token counting

        system_tokens = self.num_tokens_from_string(system_message, model)
        user_tokens = self.num_tokens_from_string(user_content, model)
        total_tokens = system_tokens + user_tokens

        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(
                f"Batch exceeds token limit: {total_tokens} > {MAX_CONTEXT_TOKENS}. "
                f"Batch size: {len(batch)} nodes. Splitting batch..."
            )
            # Split batch in half and process recursively
            mid = len(batch) // 2
            batch1 = batch[:mid]
            batch2 = batch[mid:]

            result1 = await self.generate_response(batch1, repo_id)
            result2 = await self.generate_response(batch2, repo_id)

            # Combine results
            combined_docstrings = result1.docstrings + result2.docstrings
            return DocstringResponse(docstrings=combined_docstrings)

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        logger.info(
            f"Parsing project {repo_id}: Starting the inference process... "
            f"(batch size: {len(batch)} nodes, total tokens: {total_tokens})"
        )

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

        logger.info(f"Parsing project {repo_id}: Inference request completed.")
        return result

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def batch_update_neo4j_with_cached_inference(
        self, nodes: List[Dict[str, Any]], repo_id: str
    ) -> int:
        """
        Batch update Neo4j with cached inference data for multiple nodes.
        Much faster than updating one node at a time.

        Returns the number of nodes updated.
        """
        if not nodes:
            return 0

        # Get project info ONCE for all nodes
        try:
            project = self.project_manager.get_project_from_db_by_id_sync(repo_id)
            repo_path = (
                project.get("repo_path")
                if project and isinstance(project, dict)
                else None
            )
        except Exception:
            repo_path = None
        is_local_repo = True if repo_path else False

        # Prepare batch data - reuse cached embeddings where available
        batch_data = []
        embeddings_generated = 0
        embeddings_reused = 0

        for node in nodes:
            cached_inference = node.get("cached_inference", {})
            if not cached_inference:
                continue

            docstring = cached_inference.get("docstring", "")
            tags = cached_inference.get("tags", [])

            # Reuse cached embedding if available, otherwise generate new one
            embedding = cached_inference.get("embedding_vector")
            if embedding is None:
                embedding = self.generate_embedding(docstring)
                embeddings_generated += 1
            else:
                embeddings_reused += 1

            batch_data.append(
                {
                    "node_id": node["node_id"],
                    "docstring": docstring,
                    "embedding": embedding,
                    "tags": tags,
                }
            )

        if not batch_data:
            return 0

        logger.debug(
            f"Batch updating {len(batch_data)} cached nodes "
            f"(embeddings: {embeddings_reused} reused, {embeddings_generated} generated)"
        )

        # Single Neo4j session for all updates
        with self.driver.session() as session:
            # Process in batches of 300 for optimal performance
            batch_size = 300
            for i in range(0, len(batch_data), batch_size):
                batch = batch_data[i : i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS item
                    MATCH (n:NODE {repoId: $repo_id, node_id: item.node_id})
                    SET n.docstring = item.docstring,
                        n.embedding = item.embedding,
                        n.tags = item.tags
                    """
                    + ("" if is_local_repo else ", n.text = null, n.signature = null"),
                    batch=batch,
                    repo_id=repo_id,
                )

        logger.info(
            f"Batch updated {len(batch_data)} cached nodes in Neo4j "
            f"(embeddings: {embeddings_reused} reused, {embeddings_generated} generated)"
        )
        return len(batch_data)

    async def update_neo4j_with_cached_inference(self, node: Dict[str, Any]) -> None:
        """Update Neo4j with cached inference data for a single node (legacy, use batch version)"""
        cached_inference = node.get("cached_inference", {})
        if not cached_inference:
            return

        # Extract inference data
        docstring = cached_inference.get("docstring", "")
        tags = cached_inference.get("tags", [])

        # Reuse cached embedding if available, otherwise generate new one
        embedding = cached_inference.get("embedding_vector")
        if embedding is None:
            logger.debug(
                f"Generating new embedding for cached inference node {node.get('node_id', 'unknown')}"
            )
            embedding = self.generate_embedding(docstring)
        else:
            logger.debug(
                f"Reusing cached embedding for node {node.get('node_id', 'unknown')}"
            )

        with self.driver.session() as session:
            project = self.project_manager.get_project_from_db_by_id_sync(
                node.get("project_id", "")
            )
            repo_path = project.get("repo_path") if project else None
            is_local_repo = True if repo_path else False

            session.run(
                """
                MATCH (n:NODE {repoId: $repo_id, node_id: $node_id})
                SET n.docstring = $docstring,
                    n.embedding = $embedding,
                    n.tags = $tags
                """
                + ("" if is_local_repo else ", n.text = null, n.signature = null"),
                repo_id=node.get("project_id", ""),
                node_id=node["node_id"],
                docstring=docstring,
                embedding=embedding,
                tags=tags,
            )

        logger.debug(f"Updated Neo4j with cached inference for node {node['node_id']}")

    def update_neo4j_with_docstrings(
        self,
        repo_id: str,
        docstrings: DocstringResponse,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Update Neo4j with docstrings and embeddings.

        Args:
            repo_id: Project/repo ID
            docstrings: DocstringResponse with results
            precomputed_embeddings: Optional dict of node_id -> embedding to avoid regenerating
        """
        with self.driver.session() as session:
            batch_size = 300
            precomputed = precomputed_embeddings or {}
            docstring_list = [
                {
                    "node_id": n.node_id,
                    "docstring": n.docstring,
                    "tags": n.tags,
                    # Reuse precomputed embedding if available, otherwise generate
                    "embedding": precomputed.get(n.node_id)
                    or self.generate_embedding(n.docstring),
                }
                for n in docstrings.docstrings
            ]
            project = self.project_manager.get_project_from_db_by_id_sync(repo_id)
            repo_path = project.get("repo_path")
            is_local_repo = True if repo_path else False
            for i in range(0, len(docstring_list), batch_size):
                batch = docstring_list[i : i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS item
                    MATCH (n:NODE {repoId: $repo_id, node_id: item.node_id})
                    SET n.docstring = item.docstring,
                        n.embedding = item.embedding,
                        n.tags = item.tags
                    """
                    + ("" if is_local_repo else "REMOVE n.text, n.signature"),
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
        run_inference_start = time.time()
        logger.info(
            f"[INFERENCE RUN] Starting inference pipeline for project {repo_id}",
            project_id=repo_id,
        )

        try:
            # Set status to INFERRING at the beginning (repo_id may be str or int)
            try:
                project_id_for_status = (
                    int(repo_id) if isinstance(repo_id, str) else repo_id
                )
            except (TypeError, ValueError):
                project_id_for_status = repo_id
            await self.project_manager.update_project_status(
                project_id_for_status, ProjectStatusEnum.INFERRING
            )

            # Generate docstrings
            docstrings, cache_stats = await self.generate_docstrings(repo_id)
            docstring_count = (
                len(docstrings.get("docstrings", []))
                if isinstance(docstrings, dict)
                else 0
            )
            logger.info(
                f"[INFERENCE RUN] Generated {docstring_count} docstrings",
                project_id=repo_id,
                docstring_count=docstring_count,
            )
            self.log_graph_stats(repo_id)

            # Create vector index
            vector_index_start = time.time()
            logger.info(
                f"[INFERENCE RUN] Creating vector index",
                project_id=repo_id,
            )
            self.create_vector_index()
            vector_index_time = time.time() - vector_index_start
            logger.info(
                f"[INFERENCE RUN] Created vector index in {vector_index_time:.2f}s",
                project_id=repo_id,
                vector_index_time_seconds=vector_index_time,
            )

            # Set status to READY after successful completion
            await self.project_manager.update_project_status(
                project_id_for_status, ProjectStatusEnum.READY
            )

            total_run_time = time.time() - run_inference_start
            logger.info(
                f"[INFERENCE RUN] Inference pipeline completed in {total_run_time:.2f}s",
                project_id=repo_id,
                total_run_time_seconds=total_run_time,
            )

            return cache_stats
        except Exception as e:
            logger.error(f"Inference failed for project {repo_id}: {e}")
            # Set status to ERROR on failure
            try:
                pid = int(repo_id) if isinstance(repo_id, str) else repo_id
                await self.project_manager.update_project_status(
                    pid, ProjectStatusEnum.ERROR
                )
            except Exception:
                pass
            raise

    def query_vector_index(
        self,
        project_id: str,
        query: str,
        node_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Query the vector index for similar nodes.

        Note: This may fail if called during INFERRING status when embeddings/index
        are not yet ready. The calling tool (ask_knowledge_graph_queries) handles
        these errors gracefully by returning empty results.
        """
        embedding = self.generate_embedding(query)

        with self.driver.session() as session:
            try:
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
                            node.name AS name,
                            node.type AS type,
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
                            node.name AS name,
                            node.type AS type,
                            score AS similarity
                        """,
                        project_id=project_id,
                        embedding=embedding,
                        top_k=top_k,
                    )

                # Ensure all fields are included in the final output
                return [dict(record) for record in result]
            except Exception as e:
                logger.warning(
                    f"Error querying vector index for project {project_id}: {e}"
                )
                return []

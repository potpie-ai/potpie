import asyncio
import os
import re
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
        # Handle None or empty strings gracefully
        if string is None:
            return 0
        if not isinstance(string, str):
            logger.warning(
                f"Expected string, got {type(string)}. Converting to string."
            )
            string = str(string)

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
        """Split large nodes into processable chunks with context preservation"""
        model = "gpt-4"  # Should be configurable
        max_chunk_tokens = max_tokens // 2  # Reserve space for prompt

        # Try to split by logical boundaries (functions, classes, etc.)
        lines = node_text.split("\n")
        chunks = []
        current_chunk_lines = []

        for line in lines:
            test_chunk = "\n".join(current_chunk_lines + [line])
            test_tokens = self.num_tokens_from_string(test_chunk, model)

            if test_tokens > max_chunk_tokens and current_chunk_lines:
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
                self.num_tokens_from_string(line, model)
            else:
                current_chunk_lines.append(line)

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

    def batch_nodes(
        self,
        nodes: List[Dict],
        max_tokens: int = 16000,
        model: str = "gpt-4",
        project_id: Optional[str] = None,
    ) -> List[List[DocstringRequest]]:
        """Enhanced batching with cache-aware processing"""
        batches = []
        current_batch = []
        current_tokens = 0
        node_dict = {node["node_id"]: node for node in nodes}

        # Track cache hits/misses for metrics
        cache_hits = 0
        cache_misses = 0
        uncacheable_nodes = 0

        # Get database session for cache operations
        cache_service = None
        db = None
        try:
            db = next(get_db())
            cache_service = InferenceCacheService(db)
        except Exception as e:
            logger.warning(
                f"Failed to initialize cache service: {e}. Continuing without cache."
            )
            if db:
                db.close()
            cache_service = None

        def replace_referenced_text(
            text: str, node_dict: Dict[str, Dict[str, str]]
        ) -> str:
            # Handle None input gracefully
            if text is None:
                return ""

            pattern = r"Code replaced for brevity\. See node_id ([a-f0-9]+)"
            regex = re.compile(pattern)

            resolved_refs = 0
            failed_refs = 0

            def replace_match(match):
                nonlocal resolved_refs, failed_refs
                node_id = match.group(1)
                if node_id in node_dict and node_dict[node_id].get("text"):
                    resolved_refs += 1
                    # Return full text of referenced node for consistent cache hashing
                    return node_dict[node_id]["text"]
                else:
                    failed_refs += 1
                    logger.debug(f"Failed to resolve reference to node_id: {node_id}")
                    return match.group(0)

            previous_text = None
            current_text = text

            while previous_text != current_text:
                previous_text = current_text
                current_text = regex.sub(replace_match, current_text)

            # Log reference resolution stats if any references were found
            if resolved_refs > 0 or failed_refs > 0:
                logger.debug(
                    f"Reference resolution: {resolved_refs} resolved, {failed_refs} failed"
                )

            return current_text

        for node in nodes:
            if not node.get("text"):
                logger.warning(f"Node {node['node_id']} has no text. Skipping...")
                continue

            updated_text = replace_referenced_text(node.get("text"), node_dict)
            node_tokens = self.num_tokens_from_string(updated_text, model)

            # Check if content is cacheable and look for cached inference
            if cache_service and is_content_cacheable(updated_text):
                content_hash = generate_content_hash(
                    updated_text, node.get("node_type")
                )

                # Check cache for existing inference
                # Simplified - project_id parameter ignored by cache service
                cached_inference = cache_service.get_cached_inference(content_hash)

                if cached_inference:
                    # Cache hit - store inference directly in node
                    node["cached_inference"] = cached_inference
                    node["content_hash"] = content_hash
                    cache_hits += 1

                    # Detailed logging for cache hits
                    logger.debug(
                        f"✅ CACHE HIT | "
                        f"node={node['node_id'][:8]} | "
                        f"hash={content_hash[:12]} | "
                        f"type={node.get('node_type', 'MISSING')}"
                    )
                    continue  # Skip adding to LLM batch
                else:
                    cache_misses += 1

                    # Detailed logging for cache misses
                    logger.debug(
                        f"❌ CACHE MISS | "
                        f"node={node['node_id'][:8]} | "
                        f"hash={content_hash[:12]} | "
                        f"type={node.get('node_type', 'MISSING')} | "
                        f"preview={updated_text[:80].replace(chr(10), ' ')}"
                    )

                    # Check for unresolved references
                    if "Code replaced for brevity" in updated_text:
                        logger.warning(
                            f"⚠️  UNRESOLVED REFERENCE | "
                            f"node={node['node_id'][:8]} | "
                            f"text contains unreplaced placeholder - SKIPPING CACHE"
                        )
                        # DON'T mark for caching - unresolved references have low reuse value
                        # Continue to LLM processing but without cache storage
                    else:
                        # Only mark resolved nodes for caching
                        node["content_hash"] = content_hash
                        node["should_cache"] = True
            else:
                uncacheable_nodes += 1

            # Handle large nodes (existing logic from Phase 1)
            if node_tokens > max_tokens:
                logger.info(
                    f"Node {node['node_id']} exceeds token limit ({node_tokens} tokens). Splitting into chunks..."
                )
                node_chunks = self.split_large_node(
                    updated_text, node["node_id"], max_tokens
                )

                # Process each chunk as a separate node
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
                                "should_cache": True,
                                "content_hash": generate_content_hash(
                                    chunk["text"], "chunk"
                                ),
                            },
                        )
                    )
                    current_tokens += chunk_tokens
                continue  # Skip normal processing for large nodes

            if current_tokens + node_tokens > max_tokens:
                # Finalize current batch if it has nodes
                if current_batch:
                    batches.append(current_batch)

                # Start new batch with current node
                current_batch = [
                    DocstringRequest(
                        node_id=node["node_id"], text=updated_text, metadata=node
                    )
                ]
                current_tokens = node_tokens
            else:
                # Add to current batch
                current_batch.append(
                    DocstringRequest(
                        node_id=node["node_id"], text=updated_text, metadata=node
                    )
                )
                current_tokens += node_tokens

        if current_batch:
            batches.append(current_batch)

        # Enhanced logging with cache metrics
        total_nodes = len(nodes)
        batched_nodes = sum(len(batch) for batch in batches)
        large_nodes_split = len(
            [
                n
                for n in nodes
                if n.get("text")
                and self.num_tokens_from_string(
                    replace_referenced_text(n.get("text", ""), node_dict) or "", model
                )
                > max_tokens
            ]
        )

        if cache_service:
            logger.info(
                f"Cache stats - Hits: {cache_hits}, Misses: {cache_misses}, Uncacheable: {uncacheable_nodes}"
            )
            cache_hit_rate = cache_hits / total_nodes * 100 if total_nodes > 0 else 0
            logger.info(f"Cache hit rate: {cache_hit_rate:.1f}%")

            # Run diagnostics on nodes if DEBUG logging is enabled
            try:
                from app.modules.parsing.utils.cache_diagnostics import (
                    analyze_cache_misses,
                    log_diagnostics_summary,
                )

                # Run diagnostics on the nodes we just processed
                diagnostics = analyze_cache_misses(nodes, cache_service.db)
                log_diagnostics_summary(diagnostics)
            except Exception as e:
                logger.warning(f"Failed to run cache diagnostics: {e}")

        logger.info(f"Batched {batched_nodes} nodes into {len(batches)} batches")
        logger.info(f"Large nodes split: {large_nodes_split}")
        logger.info(f"Batch sizes: {[len(batch) for batch in batches]}")

        if cache_service:
            cache_service.db.close()

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
        # Batch nodes (this mutates each node dict with cache-hit metadata)
        batches = self.batch_nodes(nodes, project_id=repo_id)

        # Process cached nodes after batching so hits are populated
        cached_nodes = [node for node in nodes if node.get("cached_inference")]
        for node in cached_nodes:
            node["project_id"] = repo_id
            await self.update_neo4j_with_cached_inference(node)
        logger.info(f"Processed {len(cached_nodes)} cached nodes for project {repo_id}")
        all_docstrings = {"docstrings": []}

        # Process LLM batches and cache results
        cache_service = None
        result_db = None
        try:
            result_db = next(get_db())
            cache_service = InferenceCacheService(result_db)
        except Exception as e:
            logger.warning(
                f"Failed to initialize cache service for result storage: {e}"
            )
            if result_db:
                result_db.close()
            cache_service = None

        semaphore = asyncio.Semaphore(self.parallel_requests)

        async def process_batch(batch, batch_index: int):
            async with semaphore:
                logger.info(f"Processing batch {batch_index} for project {repo_id}")
                try:
                    # Generate inference for batch
                    response = await self.generate_response(batch, repo_id)
                    if not isinstance(response, DocstringResponse):
                        logger.warning(
                            f"Parsing project {repo_id}: Invalid response from LLM. Not an instance of DocstringResponse. Retrying..."
                        )
                        response = await self.generate_response(batch, repo_id)

                    if isinstance(response, DocstringResponse):
                        # Store results in cache and Neo4j
                        for request, docstring_result in zip(
                            batch, response.docstrings
                        ):
                            metadata = request.metadata or {}

                            # Store in cache if eligible
                            if (
                                cache_service
                                and metadata.get("should_cache")
                                and metadata.get("content_hash")
                            ):
                                try:
                                    # Convert DocstringResult to dictionary for caching
                                    inference_data = {
                                        "node_id": docstring_result.node_id,
                                        "docstring": docstring_result.docstring,
                                        "tags": docstring_result.tags,
                                    }

                                    # project_id stored for metadata/tracing only
                                    cache_service.store_inference(
                                        content_hash=metadata["content_hash"],
                                        inference_data=inference_data,
                                        project_id=repo_id,  # Metadata only
                                        node_type=metadata.get("node_type"),
                                        content_length=len(request.text),
                                        tags=docstring_result.tags,
                                    )
                                except Exception as cache_error:
                                    logger.warning(
                                        f"Failed to cache inference for node {request.node_id}: {cache_error}"
                                    )

                        # Handle chunk consolidation before Neo4j update
                        processed_response = self.process_chunk_responses(
                            response, batch
                        )
                        if processed_response:
                            self.update_neo4j_with_docstrings(
                                repo_id, processed_response
                            )

                    return response

                except Exception as e:
                    logger.error(f"Failed to process batch {batch_index}: {e}")
                    # Continue with next batch instead of failing entire operation
                    return DocstringResponse(docstrings=[])

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

        if cache_service:
            cache_service.db.close()

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

        logger.info(f"Parsing project {repo_id}: Inference request completed.")
        return result

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    async def update_neo4j_with_cached_inference(self, node: Dict[str, Any]) -> None:
        """Update Neo4j with cached inference data for a single node"""
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

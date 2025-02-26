import asyncio
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
    DocstringNode,
)
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService
from app.modules.utils.posthog_helper import PostHogClient
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, db: Session, user_id: Optional[str] = "dummy"):
        neo4j_config = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

        self.provider_service = ProviderService(db, user_id)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.search_service = SearchService(db)
        self.project_manager = ProjectService(db)
        self.posthog_client = PostHogClient()
        self.user_id = user_id
        self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", 50))
        self.cache_enabled = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"

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
            
            # First, check total nodes and how many have content_hash
            counts_check = session.run(
                """
                MATCH (n:NODE {repoId: $repo_id})
                RETURN COUNT(n) as total_count,
                       COUNT(n.content_hash) as hash_count
                """,
                repo_id=repo_id
            )
            counts = counts_check.single()
            total_count = counts["total_count"]
            hash_count = counts["hash_count"]
            logger.info(f"Found {hash_count} nodes with content_hash out of {total_count} total nodes in repo {repo_id}")
            
            # If no hashes found, try to recalculate them
            if hash_count == 0:
                logger.warning(f"No content_hash values found in repo {repo_id}, attempting to calculate them")
                # Calculate each node's content hash and store it directly in our Python code
                # Retrieve nodes without content hash that have both name and text
                nodes_to_update = []
                nodes_query = session.run(
                    """
                    MATCH (n:NODE {repoId: $repo_id})
                    WHERE n.name IS NOT NULL AND n.text IS NOT NULL
                    AND (n.content_hash IS NULL)
                    RETURN n.node_id AS node_id, n.name AS name, n.text AS text
                    """,
                    repo_id=repo_id
                )
                
                # Calculate SHA-256 hash in Python and prepare batch updates
                import hashlib
                for record in nodes_query:
                    node_id = record["node_id"]
                    name = record["name"] or ""
                    text = record["text"] or ""
                    
                    # Normalize whitespace and combine name and text (same as our generate_content_hash method)
                    normalized_name = " ".join(name.split())
                    normalized_text = " ".join(text.split())
                    combined_string = f"{normalized_name}:{normalized_text}"
                    
                    # Create SHA-256 hash
                    hash_object = hashlib.sha256()
                    hash_object.update(combined_string.encode("utf-8"))
                    content_hash = hash_object.hexdigest()
                    
                    nodes_to_update.append({
                        "node_id": node_id,
                        "content_hash": content_hash
                    })
                
                # Apply updates in batches
                batch_size = 500
                total_updated = 0
                for i in range(0, len(nodes_to_update), batch_size):
                    batch = nodes_to_update[i:i+batch_size]
                    update_result = session.run(
                        """
                        UNWIND $batch AS item
                        MATCH (n:NODE {repoId: $repo_id, node_id: item.node_id})
                        SET n.content_hash = item.content_hash
                        """,
                        batch=batch,
                        repo_id=repo_id
                    )
                    total_updated += len(batch)
                
                logger.info(f"Updated content_hash for {total_updated} nodes in repo {repo_id}")
            
            # Now fetch all nodes including content_hash
            while True:
                result = session.run(
                    "MATCH (n:NODE {repoId: $repo_id}) "
                    "RETURN n.node_id AS node_id, n.text AS text, n.file_path AS file_path, n.start_line AS start_line, "
                    "n.end_line AS end_line, n.name AS name, n.content_hash AS content_hash "
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
        nodes_with_hash = sum(1 for node in all_nodes if node.get("content_hash"))
        logger.info(f"CONTENT_HASH: {nodes_with_hash} out of {len(all_nodes)} nodes have content hash ({(nodes_with_hash/len(all_nodes))*100:.2f}%)")
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
                MATCH (n:FUNCTION:FILE)
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
        self, nodes: List[Dict], max_tokens: int = 16000, model: str = "gpt-4"
    ) -> List[List[DocstringRequest]]:
        batches = []
        current_batch = []
        current_tokens = 0
        node_dict = {node["node_id"]: node for node in nodes}

        def replace_referenced_text(
            text: str, node_dict: Dict[str, Dict[str, str]]
        ) -> str:
            pattern = r"Code replaced for brevity\. See node_id ([a-f0-9]+)"
            regex = re.compile(pattern)

            def replace_match(match):
                node_id = match.group(1)
                if node_id in node_dict:
                    return "\n" + node_dict[node_id]["text"].split("\n", 1)[-1]
                return match.group(0)

            previous_text = None
            current_text = text

            while previous_text != current_text:
                previous_text = current_text
                current_text = regex.sub(replace_match, current_text)
            return current_text

        for node in nodes:
            if not node.get("text"):
                logger.warning(f"Node {node['node_id']} has no text. Skipping...")
                continue

            updated_text = replace_referenced_text(node["text"], node_dict)
            node_tokens = self.num_tokens_from_string(updated_text, model)

            if node_tokens > max_tokens:
                logger.warning(
                    f"Node {node['node_id']} - {node_tokens} tokens, has exceeded the max_tokens limit. Skipping..."
                )
                continue

            if current_tokens + node_tokens > max_tokens:
                if current_batch:  # Only append if there are items
                    batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(
                DocstringRequest(
                    node_id=node["node_id"], 
                    text=updated_text,
                    content_hash=node.get("content_hash"),
                    name=node.get("name", "")
                )
            )
            current_tokens += node_tokens

        if current_batch:
            batches.append(current_batch)

        total_nodes = sum(len(batch) for batch in batches)
        logger.info(f"Batched {total_nodes} nodes into {len(batches)} batches")
        logger.info(f"Batch sizes: {[len(batch) for batch in batches]}")

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
                messages=messages, output_schema=DocstringResponse, size="small"
            )
            return result
        except Exception as e:
            logger.error(f"Entry point response generation failed: {e}")
            return DocstringResponse(docstrings=[])

    def find_cached_nodes(self, content_hashes: List[str]) -> Dict[str, Dict]:
        """
        Finds nodes with matching content hashes from the global cache.
        
        Args:
            content_hashes: List of content hashes to look up
            
        Returns:
            Dictionary mapping content_hash to cached node data (docstring, tags, embedding)
        """
        if not content_hashes or not self.cache_enabled:
            return {}
            
        cached_nodes = {}
        with self.driver.session() as session:
            
            # Create the index regardless - Neo4j will handle if it already exists
            try:
                # Create content_hash index without checking if it exists first
                logger.info("Creating content_hash index for better cache lookup performance")
                session.run(
                    """
                    CREATE INDEX content_hash_idx IF NOT EXISTS
                    FOR (n:NODE) ON (n.content_hash)
                    """
                )
            except Exception as e:
                logger.warning(f"Error creating content_hash index: {str(e)}")
                # Fall back to simpler index creation if the above doesn't work
                try:
                    session.run("CREATE INDEX ON :NODE(content_hash)")
                except Exception as e2:
                    logger.warning(f"Error creating fallback index: {str(e2)}")
                    # Continue without index - will be slower but should still work
            
            # Count nodes with content_hash and docstring for debugging
            count_query = session.run(
                """
                MATCH (n:NODE) 
                WHERE n.content_hash IS NOT NULL AND n.docstring IS NOT NULL
                RETURN COUNT(n) AS nodeCount
                """
            )
            count_record = count_query.single()
            if count_record:
                logger.info(f"Total nodes with content_hash and docstring in database: {count_record['nodeCount']}")
            
            # Process in batches to avoid query size limitations
            batch_size = 500
            for i in range(0, len(content_hashes), batch_size):
                batch_hashes = content_hashes[i:i+batch_size]
                # Add debugging to see what's happening
                logger.info(f"Looking up {len(batch_hashes)} content hashes in the cache")
                
                # Get a sample of hashes for debugging
                sample = batch_hashes[:5] if len(batch_hashes) > 5 else batch_hashes
                logger.info(f"Sample hashes being looked up: {sample}")
                
                # Try to directly find one hash for debugging
                if batch_hashes:
                    test_hash = batch_hashes[0]
                    test_result = session.run(
                        """
                        MATCH (n:NODE)
                        WHERE n.content_hash = $test_hash
                        RETURN COUNT(n) as count
                        """,
                        test_hash=test_hash
                    )
                    test_count = test_result.single()["count"]
                    logger.info(f"Direct test lookup for hash {test_hash}: found {test_count} nodes")
                
                # Find all matching nodes with the requested content hashes
                result = session.run(
                    """
                    // For each content hash, find the best match with a docstring and embedding
                    UNWIND $content_hashes as hash
                    MATCH (n:NODE {content_hash: hash})
                    WHERE n.docstring IS NOT NULL AND n.embedding IS NOT NULL
                    // Add more detailed logging to debug what's happening
                    WITH hash, collect(n) as nodes
                    WITH hash, nodes, size(nodes) as count
                    
                    // If we found any nodes, return one of them
                    WHERE count > 0
                    WITH hash, nodes[0] as bestNode
                    
                    RETURN hash as content_hash, 
                           bestNode.docstring AS docstring, 
                           bestNode.tags AS tags, 
                           bestNode.embedding AS embedding
                    """,
                    content_hashes=batch_hashes
                )
                
                # Get all records and log the count
                records = list(result)
                logger.info(f"Found {len(records)} matching nodes with content_hash in the cache")
                
                # Get detailed info about which hashes matched and which didn't
                found_hashes = set(record["content_hash"] for record in records)
                missed_hashes = set(batch_hashes) - found_hashes
                
                if missed_hashes:
                    logger.info(f"MISSED CACHE: {len(missed_hashes)}/{len(batch_hashes)} content hashes not found in cache")
                    
                    # Get all nodes in the database with their docstrings and embeddings status
                    node_stats = session.run(
                        """
                        MATCH (n:NODE)
                        WHERE n.content_hash IS NOT NULL
                        RETURN 
                            COUNT(n) AS total_nodes,
                            COUNT(CASE WHEN n.docstring IS NOT NULL THEN 1 END) AS nodes_with_docstring,
                            COUNT(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS nodes_with_embedding,
                            COUNT(CASE WHEN n.docstring IS NOT NULL AND n.embedding IS NOT NULL THEN 1 END) AS nodes_with_both
                        """
                    )
                    
                    stats = node_stats.single()
                    if stats:
                        logger.info(f"DATABASE STATS: Total nodes with content_hash: {stats['total_nodes']}, " +
                                  f"with docstring: {stats['nodes_with_docstring']}, " +
                                  f"with embedding: {stats['nodes_with_embedding']}, " +
                                  f"with both: {stats['nodes_with_both']}")
                    
                    # Get a sample of missed hashes
                    sample_missed = list(missed_hashes)[:3] if len(missed_hashes) > 3 else list(missed_hashes)
                    if sample_missed:
                        logger.info(f"MISSED HASH EXAMPLES: {sample_missed}")
                        
                        # Look for these hashes in the database
                        for missed_hash in sample_missed:
                            verification = session.run(
                                """
                                MATCH (n:NODE)
                                WHERE n.content_hash = $hash
                                RETURN n.node_id AS node_id, n.name AS name, n.file_path AS file_path,
                                      n.docstring IS NOT NULL as has_docstring,
                                      n.embedding IS NOT NULL as has_embedding
                                """,
                                hash=missed_hash
                            )
                            
                            verification_records = list(verification)
                            if verification_records:
                                for v_record in verification_records:
                                    logger.info(f"  Found missed hash {missed_hash} in node {v_record['node_id']}, name: {v_record['name']}, file: {v_record['file_path']}, has_docstring: {v_record['has_docstring']}, has_embedding: {v_record['has_embedding']}")
                            else:
                                logger.info(f"  Missed hash {missed_hash} not found in ANY node in the database")
                            
                    # Also check for file changes between the two branches
                    missed_files_check = session.run(
                        """
                        MATCH (n:NODE)
                        WHERE n.content_hash IN $missed_hashes
                        RETURN DISTINCT n.file_path AS file_path
                        """,
                        missed_hashes=list(missed_hashes)
                    )
                    
                    missed_files = [record["file_path"] for record in missed_files_check]
                    if missed_files:
                        logger.info(f"CACHE MISS FILES: {missed_files}")
                
                for record in records:
                    cached_nodes[record["content_hash"]] = {
                        "docstring": record["docstring"],
                        "tags": record["tags"],
                        "embedding": record["embedding"]
                    }
                    
        return cached_nodes

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
        
        # Apply semantic caching to skip LLM calls for nodes with matching content
        nodes_to_process = []
        cached_nodes_count = 0
        cache_hits = []
        
        if self.cache_enabled:
            # Extract content hashes from nodes
            content_hashes = [node.get("content_hash") for node in nodes if node.get("content_hash")]
            
            if content_hashes:
                # Log this for debugging purposes
                logger.info(f"Found {len(content_hashes)} nodes with content hash out of {len(nodes)} total nodes")
                # Show a sample of the hashes
                sample_hashes = content_hashes[:5] if len(content_hashes) > 5 else content_hashes
                logger.info(f"Sample content hashes: {sample_hashes}")
                
                # Show a sample node with its properties
                sample_node = next((n for n in nodes if n.get("content_hash")), None)
                if sample_node:
                    logger.info(f"Sample node with content_hash: {sample_node}")
                else:
                    logger.warning("No nodes with content_hash field found to display")
                
                # Find matching nodes in the cache
                cached_nodes_map = self.find_cached_nodes(content_hashes)
                logger.info(f"Cache lookup returned {len(cached_nodes_map)} hits")
                
                # Process each node - either use cache or mark for LLM processing
                for node in nodes:
                    content_hash = node.get("content_hash")
                    if content_hash and content_hash in cached_nodes_map:
                        # Use the cached result
                        cache_hits.append({
                            "node_id": node["node_id"],
                            "docstring": cached_nodes_map[content_hash]["docstring"],
                            "tags": cached_nodes_map[content_hash]["tags"],
                            "embedding": cached_nodes_map[content_hash]["embedding"]
                        })
                        cached_nodes_count += 1
                    else:
                        # Node needs LLM processing
                        nodes_to_process.append(node)
            else:
                logger.warning("No nodes with content_hash found - all nodes will be processed")
                nodes_to_process = nodes
        else:
            # If caching is disabled, process all nodes
            logger.info("Semantic caching is disabled")
            nodes_to_process = nodes
        
        # Log cache metrics
        cache_hit_rate = 0 if not nodes else (cached_nodes_count / len(nodes)) * 100
        logger.info(f"Semantic cache hit rate: {cache_hit_rate:.2f}% ({cached_nodes_count}/{len(nodes)} nodes)")
        
        # Send cache metrics to PostHog
        self.posthog_client.send_event(
            self.user_id,
            "semantic_cache_metrics",
            {
                "project_id": repo_id,
                "total_nodes": len(nodes),
                "cache_hits": cached_nodes_count,
                "cache_hit_rate": cache_hit_rate
            }
        )
        
        # Create a DocstringResponse for cached nodes
        cached_docstrings = DocstringResponse(docstrings=[])
        for hit in cache_hits:
            cached_docstrings.docstrings.append(
                DocstringNode(
                    node_id=hit["node_id"],
                    docstring=hit["docstring"],
                    tags=hit["tags"]
                )
            )
        
        # Update Neo4j with cached results immediately
        if cached_nodes_count > 0:
            self.update_neo4j_with_docstrings(repo_id, cached_docstrings)
        
        # If we have nodes that need processing, proceed with LLM inference
        all_docstrings = {"docstrings": cached_docstrings.docstrings}
        
        if nodes_to_process:
            batches = self.batch_nodes(nodes_to_process)
            
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
                        f"Project {repo_id}: Invalid response during inference. Manually verify the project completion."
                    )
                else:
                    all_docstrings["docstrings"].extend(result.docstrings)

        return all_docstrings

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
                messages=messages, output_schema=DocstringResponse, size="small"
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
            project = self.project_manager.get_project_from_db_by_id_sync(repo_id)
            repo_path = project.get("repo_path")
            is_local_repo = True if repo_path else False
            
            # First, fetch all relevant properties for each node
            node_properties = {}
            for i in range(0, len(docstring_list), batch_size):
                batch = docstring_list[i : i + batch_size]
                node_ids = [item["node_id"] for item in batch]
                
                # Get content_hash and other properties
                hash_result = session.run(
                    """
                    MATCH (n:NODE {repoId: $repo_id})
                    WHERE n.node_id IN $node_ids
                    RETURN 
                      n.node_id AS node_id, 
                      n.content_hash AS content_hash,
                      n.name AS name,
                      n.text AS text
                    """,
                    repo_id=repo_id,
                    node_ids=node_ids
                )
                
                for record in hash_result:
                    node_id = record["node_id"]
                    node_properties[node_id] = {
                        "content_hash": record["content_hash"],
                        "name": record["name"],
                        "text": record["text"]
                    }
            
            # Update nodes with docstrings and embeddings
            for i in range(0, len(docstring_list), batch_size):
                batch = docstring_list[i : i + batch_size]
                
                # Add properties to each item in the batch
                for item in batch:
                    node_id = item["node_id"]
                    if node_id in node_properties:
                        props = node_properties[node_id]
                        
                        # If content_hash is missing but we have name and text, calculate it
                        if props["content_hash"] is None and props["name"] and props["text"]:
                            # Use the same algorithm as in CodeGraphService
                            import hashlib
                            name = props["name"] or ""
                            text = props["text"] or ""
                            
                            # Normalize whitespace and combine name and text
                            normalized_name = " ".join(name.split())
                            normalized_text = " ".join(text.split())
                            combined_string = f"{normalized_name}:{normalized_text}"
                            
                            # Create a SHA-256 hash of the combined string
                            hash_object = hashlib.sha256()
                            hash_object.update(combined_string.encode("utf-8"))
                            props["content_hash"] = hash_object.hexdigest()
                            
                        # Add properties to item
                        item["content_hash"] = props["content_hash"]
                
                # Log what we're updating
                nodes_with_hash = sum(1 for item in batch if item.get("content_hash"))
                logger.info(f"Updating {len(batch)} nodes, {nodes_with_hash} have content_hash")
                
                # Execute the update
                session.run(
                    """
                    UNWIND $batch AS item
                    MATCH (n:NODE {repoId: $repo_id, node_id: item.node_id})
                    SET n.docstring = item.docstring,
                        n.embedding = item.embedding,
                        n.tags = item.tags,
                        n.content_hash = item.content_hash
                    """
                    + ("" if is_local_repo else "REMOVE n.text, n.signature"),
                    batch=batch,
                    repo_id=repo_id,
                )

    def create_vector_index(self):
        with self.driver.session() as session:
            # Create vector index for embeddings
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
            
            # Create index on content_hash for faster cache lookups
            session.run(
                """
                CREATE INDEX content_hash_idx IF NOT EXISTS
                FOR (n:NODE) ON (n.content_hash)
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

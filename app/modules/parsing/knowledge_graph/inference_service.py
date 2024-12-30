import asyncio
import logging
import os
import re
from typing import Dict, List, Optional

import tiktoken
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.intelligence.provider.provider_service import (
    AgentType,
    ProviderService,
)
from app.modules.parsing.knowledge_graph.inference_schema import (
    DocstringRequest,
    DocstringResponse,
)
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, db: Session, user_id: Optional[str] = "dummy"):
        neo4j_config = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )
        self._llm = None
        self._llm_provider = ProviderService(db, user_id)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.search_service = SearchService(db)
        self.project_manager = ProjectService(db)
        self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", 50))

    async def _get_llm(self):
        if self._llm is None:
            self._llm = await self._llm_provider.get_small_llm(agent_type=AgentType.LANGCHAIN)
        return self._llm

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
                DocstringRequest(node_id=node["node_id"], text=updated_text)
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

        # Process batches sequentially instead of using semaphore
        updated_docstrings = DocstringResponse(docstrings=[])
        
        for batch_index, batch in enumerate(entry_point_batches):
            try:
                logger.info(f"Processing entry point batch {batch_index + 1}/{len(entry_point_batches)}")
                
                # First attempt
                response = await self.generate_entry_point_response(batch)
                
                if not isinstance(response, DocstringResponse):
                    logger.warning("Invalid response from LLM, retrying after delay...")
                    await asyncio.sleep(12)  # 5 requests/minute = 12 second delay
                    response = await self.generate_entry_point_response(batch)
                
                if isinstance(response, DocstringResponse):
                    updated_docstrings.docstrings.extend(response.docstrings)
                    logger.info(f"Successfully processed batch {batch_index + 1}")
                else:
                    logger.error(f"Invalid response after retry for batch {batch_index + 1}, skipping")
                
                # Add delay between batches
                await asyncio.sleep(12)
                
            except Exception as e:
                logger.error(f"Error processing entry point batch {batch_index + 1}: {str(e)}")
                await asyncio.sleep(12)
                continue
            
            # Log progress every 5 batches
            if (batch_index + 1) % 5 == 0:
                logger.info(
                    f"Entry point processing progress: {batch_index + 1}/{len(entry_point_batches)} batches"
                )

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

        logger.info(
            f"Completed entry point processing. "
            f"Updated {len(updated_docstrings.docstrings)} docstrings"
        )
        
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

        # Prepare nodes for indexing
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

        # Create batches and initialize response dictionary
        batches = self.batch_nodes(nodes)
        logger.info(f"Batched {len(nodes)} nodes into {len(batches)} batches")
        logger.info(f"Batch sizes: {[len(batch) for batch in batches]}")
        
        all_docstrings = {"docstrings": []}
        
        # Process batches sequentially
        for batch_index, batch in enumerate(batches):
            try:
                logger.info(f"Processing batch {batch_index}/{len(batches)} for project {repo_id}")
                
                # First attempt
                logger.info(f"Parsing project {repo_id}: Starting the inference process...")
                response = await self.generate_response(batch, repo_id)
                
                # Handle invalid response with one retry
                if not isinstance(response, DocstringResponse):
                    logger.warning(
                        f"Parsing project {repo_id}: Invalid response from LLM. Not an instance of DocstringResponse. Retrying..."
                    )
                    # Wait before retry
                    await asyncio.sleep(12)  # 5 requests/minute = 12 seconds between requests
                    response = await self.generate_response(batch, repo_id)
                
                # Process valid response
                if isinstance(response, DocstringResponse):
                    self.update_neo4j_with_docstrings(repo_id, response)
                    all_docstrings["docstrings"].extend(response.docstrings)
                    logger.info(
                        f"Successfully processed batch {batch_index} with {len(response.docstrings)} docstrings"
                    )
                else:
                    logger.error(
                        f"Project {repo_id}: Invalid response for batch {batch_index} after retry. Skipping batch."
                    )
                
                # Add delay between batches to respect rate limits
                await asyncio.sleep(12)  # Ensure we stay within rate limits
                
            except Exception as e:
                logger.error(
                    f"Error processing batch {batch_index} for project {repo_id}: {str(e)}"
                )
                # Wait after error before continuing to next batch
                await asyncio.sleep(12)
                continue
            
            # Log progress
            if (batch_index + 1) % 5 == 0:
                logger.info(
                    f"Progress update: Completed {batch_index + 1}/{len(batches)} batches for project {repo_id}"
                )

        logger.info(
            f"Completed docstring generation for project {repo_id}. "
            f"Generated {len(all_docstrings['docstrings'])} docstrings"
        )
        
        return all_docstrings

    async def generate_response(
        self, batch: List[DocstringRequest], repo_id: str
    ) -> str:
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
        - Based on the identified type, write a brief (1-2 sentences) summary of the code’s main purpose and functionality.
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


        3. **Output Compilation**:
        - Collect the generated docstrings and classifications for each `node_id`.
        - Ensure that the output includes an entry for every `node_id` provided in the `code_snippets`.

        4. **Review and Verification**:
        Before finalizing your response:
        - Verify that every `node_id` from the input is present in the output.
        - Ensure each docstring is clear, comprehensive, and technically accurate.
        - Confirm that the assigned tags are justified by the code's functionality.
        - Make sure all crucial technical details are captured without unnecessary verbosity.

        Refine your output as needed to ensure high-quality, precise documentation that accurately represents the code's structure and functionality.

        {format_instructions}
        Ensure that the response is a valid DocstringResponse object. Every entry in the response must contain the key "docstring".
        Even if the docstring is empty, you must still include the node_id and an empty docstring in your response.
        Here are the code snippets:

        {code_snippets}
        """

        # Prepare the code snippets
        code_snippets = ""
        for request in batch:
            code_snippets += (
                f"node_id: {request.node_id} \n```\n{request.text}\n```\n\n "
            )

        output_parser = PydanticOutputParser(pydantic_object=DocstringResponse)

        chat_prompt = ChatPromptTemplate.from_template(
            template=base_prompt,
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )

        import time

        start_time = time.time()
        logger.info(f"Parsing project {repo_id}: Starting the inference process...")

        chain = chat_prompt | await self._get_llm() | output_parser
        try:
            result = await chain.ainvoke({"code_snippets": code_snippets})
        except Exception as e:
            logger.error(
                f"Parsing project {repo_id}: Inference request failed. Error: {str(e)}"
            )
            result = ""

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

import asyncio
import logging
from typing import Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.parsing.knowledge_graph.inference_schema import (
    DocstringRequest,
    DocstringResponse,
)
from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, db: Session):
        neo4j_config = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.search_service = SearchService(db)

    def close(self):
        self.driver.close()

    def fetch_graph(self, repo_id: str) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:NODE {repoId: $repo_id}) "
                "RETURN n.node_id AS node_id, n.text AS text, n.file_path AS file_path, n.start_line AS start_line, n.end_line AS end_line, n.name AS name",
                repo_id=repo_id,
            )
            return [dict(record) for record in result]

    def get_entry_points(self, repo_id: str) -> List[str]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (f:FUNCTION)
                WHERE f.repoId = '{repo_id}'
                AND NOT ()-[:CALLS]->(f)
                AND (f)-[:CALLS]->()
                RETURN f.node_id as node_id
                """,
            )
            data = result.data()
            return [record["node_id"] for record in data]

    def get_neighbours(self, node_id: str, repo_id: str):
        with self.driver.session() as session:
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
                """,
                node_id=node_id,
                repo_id=repo_id,
            )
            data = result.data()

            nodes_info = [
                record["node_id"] for record in data if "FUNCTION" in record["labels"]
            ]
            return nodes_info

    def get_entry_points_for_nodes(
        self, node_ids: List[str], repo_id: str
    ) -> Dict[str, List[str]]:
        with self.driver.session() as session:
            result = session.run(
                """
UNWIND $node_ids AS nodeId
                MATCH (n)
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
        self, nodes: List[Dict], max_tokens: int = 32000
    ) -> List[List[DocstringRequest]]:
        batches = []
        current_batch = []
        current_tokens = 0

        for node in nodes:
            # Skip nodes with None or empty text
            if not node.get("text"):
                continue

            node_tokens = len(node["text"].split())
            if node_tokens > max_tokens:
                continue  # Skip nodes that exceed the max_tokens limit

            if current_tokens + node_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(
                DocstringRequest(node_id=node["node_id"], text=node["text"])
            )
            current_tokens += node_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def generate_docstrings_for_entry_points(
        self,
        all_docstrings: DocstringResponse,
        entry_points_neighbors: Dict[str, List[str]],
    ) -> Dict[str, DocstringResponse]:
        docstring_lookup = {
            d.node_id: d.docstring for d in all_docstrings["docstrings"]
        }

        entry_point_batches = self.batch_entry_points(
            entry_points_neighbors, docstring_lookup
        )

        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

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
        max_tokens: int = 32000,
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

            entry_point_tokens = len(entry_docstring) + len(flow_description)

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
        For each of the following entry points and their function flows, perform the following tasks:

        1. **Flow Summary**: Generate a concise summary of the overall intent and purpose of the entry point and its flow. The summary should be sufficiently verbose and technical to understand the code via similarity search but concise enough to not take up too much space. ALWAYS Include technical details such as ALL API paths, HTTP methods, topic names, database interactions, and key function calls.

        2. **Classification**: Classify the entry point into one or more of the following broader level tags: **[API, WEBSOCKET, PRODUCER, CONSUMER, DATABASE, HTTP]**. Only use these tags and select the ones that are most relevant to the entry point.

        {entry_points}

        Respond with a list of "node_id":"updated docstring","tags" pairs, where the updated docstring includes the original docstring followed by the flow summary.

        {format_instructions}
        """

        entry_points_text = "\n\n".join(
            [
                f"Entry point: {entry_point['node_id']}\n"
                f"Flow:\n{entry_point['flow_description']}"
                f"Entry docstring:\n{entry_point['entry_docstring']}"
                for entry_point in batch
            ]
        )

        formatted_prompt = prompt
        return await self.generate_llm_response(
            formatted_prompt, {"entry_points": entry_points_text}
        )

    async def generate_llm_response(self, prompt: str, inputs: Dict) -> str:
        output_parser = PydanticOutputParser(pydantic_object=DocstringResponse)

        chat_prompt = ChatPromptTemplate.from_template(
            template=prompt,
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
        )
        chain = chat_prompt | self.llm | output_parser
        result = await chain.ainvoke(input=inputs)
        return result

    async def generate_docstrings(self, repo_id: str) -> Dict[str, DocstringResponse]:
        nodes = self.fetch_graph(repo_id)
        for node in nodes:
            if node.get("file_path") not in {None, ""} and node.get("name") not in {
                None,
                "",
            }:
                await self.search_service.create_search_index(repo_id, node)
        await self.search_service.commit_indices()
        entry_points = self.get_entry_points(repo_id)
        entry_points_neighbors = {}
        for entry_point in entry_points:
            neighbors = self.get_neighbours(entry_point, repo_id)
            entry_points_neighbors[entry_point] = neighbors

        batches = self.batch_nodes(nodes)
        all_docstrings = {}

        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

        async def process_batch(batch):
            async with semaphore:
                response = await self.generate_response(batch)
                if isinstance(response, DocstringResponse):
                    return response
                else:
                    return await self.generate_docstrings(repo_id)

        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        for result in results:
            all_docstrings.update(result)

        updated_docstrings = await self.generate_docstrings_for_entry_points(
            all_docstrings, entry_points_neighbors
        )

        return updated_docstrings

    async def generate_response(self, batch: List[DocstringRequest]) -> str:
        base_prompt = """
        For each of the following code snippets, perform the following task:

        1. **Docstring Generation**: Generate a detailed technical docstring that encapsulates the technical and functional purpose of the code. The docstring should be sufficiently verbose and technical to understand the code via similarity search but concise enough to not take up too much space. ALWAYS include relevant technical details like API paths, HTTP methods, topic names, function calls, database operations, etc.
        2. **Classification**: Classify the entry point into one or more of the following broader level tags: **[API, WEBSOCKET, PRODUCER, CONSUMER, DATABASE, HTTP]**. Only use these tags and select the ones that are most relevant to the entry point.

        Here are the code snippets:
        {code_snippets}

        {format_instructions}
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
        print("Starting the inference process...")
        total_word_count = len(base_prompt.split()) + sum(
            len(request.text.split()) for request in batch
        )
        print(f"Request contains {total_word_count} words.")

        chain = chat_prompt | self.llm | output_parser
        result = await chain.ainvoke({"code_snippets": code_snippets})
        end_time = time.time()

        print(
            f"Start Time: {start_time}, End Time: {end_time}, Total Time Taken: {end_time - start_time} seconds"
        )
        return result

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    async def update_neo4j_with_docstrings(
        self, repo_id: str, docstrings: DocstringResponse
    ):
        with self.driver.session() as session:
            batch_size = 300
            docstring_list = [
                {
                    "node_id": n.node_id,
                    "docstring": n.docstring,
                    "tags": n.tags ,
                    "embedding": self.generate_embedding(n.docstring),
                }
                for n in docstrings["docstrings"]
            ]

            for i in range(0, len(docstring_list), batch_size):
                batch = docstring_list[i : i + batch_size]
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
        await self.update_neo4j_with_docstrings(repo_id, docstrings)
        self.create_vector_index()

    async def query_vector_index(
        self,
        project_id: str,
        query: str,
        node_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        embedding = self.generate_embedding(query)

        with self.driver.session() as session:
            if node_ids:
                # Part 1: Fetch neighboring nodes
                result_neighbors = session.run(
                    """
                    MATCH (n:NODE)
                    WHERE n.repoId = $project_id AND n.node_id IN $node_ids
                    CALL {
                        WITH n
                        MATCH (n)-[*1..4]-(neighbor:NODE)
                        RETURN COLLECT(DISTINCT neighbor) AS neighbors
                    }
                    RETURN COLLECT(DISTINCT n) + REDUCE(acc = [], neighbors IN COLLECT(neighbors) | acc + neighbors) AS context_nodes
                    """,
                    project_id=project_id,
                    node_ids=node_ids,
                )
                context_nodes = result_neighbors.single()["context_nodes"]

                context_node_data = [
                    {
                        "node_id": node["node_id"],
                        "embedding": node["embedding"],
                        "docstring": node.get("docstring", ""),
                        "file_path": node.get("file_path", ""),
                        "start_line": node.get("start_line", -1),
                        "end_line": node.get("end_line", -1),
                    }
                    for node in context_nodes
                ]

                result = session.run(
                    """
                    UNWIND $context_node_data AS context_node
                    WITH context_node,
                         vector.similarity.cosine(context_node.embedding, $embedding) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                    RETURN context_node.node_id AS node_id,
                           context_node.docstring AS docstring,
                           context_node.file_path AS file_path,
                           context_node.start_line AS start_line,
                           context_node.end_line AS end_line,
                           similarity
                    """,
                    context_node_data=context_node_data,
                    embedding=embedding,
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

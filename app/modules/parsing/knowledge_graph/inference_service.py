import asyncio
import logging
from typing import Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from app.core.config_provider import config_provider
from app.modules.parsing.knowledge_graph.inference_schema import (
    DocstringRequest,
    DocstringResponse,
)
from app.modules.search.search_service import SearchService
from sqlalchemy.orm import Session
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

    def get_neighbours(self, node_id: str):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p {node_id: $node_id})
                CALL apoc.neighbors.byhop(p, ">", 10)
                YIELD nodes
                UNWIND nodes AS all_nodes
                RETURN all_nodes.node_id AS node_id, all_nodes.name AS function_name, labels(all_nodes) AS labels
                """,
                node_id=node_id,
            )
            data = result.data()

            nodes_info = [
                record["node_id"] for record in data if record["labels"] == ["function"]
            ]
            return nodes_info

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
                "flow_description": flow_description,
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
        Analyze the following entry points and their function flows to generate concise summaries of the overall intent and purpose for each:

        {entry_points}

        For each entry point, provide a brief, high-level description of what the flow accomplishes, such as "API to do XYZ" or "Kafka consumer for consuming topic ABC", but with more technical detail.
        ALWAYS INCLUDE TECHICAL details about the API Path if relevant e.g. "GET document API at /api/v1/document/id", Topic name e.g. "Kafka consumer with 5 replicas consuming from the 'input' topic", flow of the code, the entry point and the function calls between them.
        Respond with a list of "node_id":"updated docstring" pairs, where the updated docstring includes the original docstring followed by the flow summary.

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

        # formatted_prompt = prompt.format(entry_points=entry_points_text)
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
            if node.get("file_path") not in {None, ""} and node.get("name") not in {None, ""}:
                await self.search_service.create_search_index(
                    repo_id, node
            )
        await self.search_service.commit_indices()
        entry_points = self.get_entry_points(repo_id)
        entry_points_neighbors = {}
        for entry_point in entry_points:
            neighbors = self.get_neighbours(entry_point)
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
        Generate a detailed technical docstring for each of the following code snippets.
        The docstring should encapsulate the technical and functional purpose of the code.
        Include details about inputs, outputs, function calls, logical flow, and any other relevant information.
        If the code snippet serves a special purpose like defining an API or a Kafka consumer or Producer, make note of that in the docstring with details like API path, topic name etc.
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
                        n.embedding = item.embedding
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

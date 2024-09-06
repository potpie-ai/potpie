import asyncio
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from app.core.config_provider import config_provider
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class DocstringRequest(BaseModel):
    node_id: str
    text: str

class DocstringNode(BaseModel):
    node_id: str
    docstring: str 

class DocstringResponse(BaseModel):
    docstrings: List[DocstringNode]

class InferenceService:
    def __init__(self):
        neo4j_config = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"])
        )
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def close(self):
        self.driver.close()

    def fetch_graph(self, repo_id: str) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:NODE {repoId: $repo_id}) "
                "RETURN n.node_id AS node_id, n.type AS type, n.text AS text",
                repo_id=repo_id
            )
            return [dict(record) for record in result]

    def batch_nodes(self, nodes: List[Dict], max_tokens: int = 32000) -> List[List[DocstringRequest]]:
        batches = []
        current_batch = []
        current_tokens = 0

        for node in nodes:
            node_tokens = len(node['text'].split())
            if node_tokens > max_tokens:
                continue  # Skip nodes that exceed the max_tokens limit

            if current_tokens + node_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(DocstringRequest(node_id=node['node_id'], text=node['text']))
            current_tokens += node_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def generate_docstrings(self, repo_id: str) -> Dict[str, DocstringResponse]:
        nodes = self.fetch_graph(repo_id)
        batches = self.batch_nodes(nodes)
        all_docstrings = {}

        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

        async def process_batch(batch, retry=1):
            async with semaphore:
                response =  await self.generate_response(batch)
                if isinstance(response, DocstringResponse):
                    return response
                else:
                    retry=retry-1
                    if retry > 0:
                        await process_batch(batch, retry=retry)
                    else:
                        raise Exception("Failed to generate docstrings for batch")


        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        for result in results:
            all_docstrings.update(result)

        return all_docstrings


    async def generate_response(self, batch: List[DocstringRequest]) -> str:
        base_prompt = """
        Generate a detailed technical docstring for each of the following code snippets. 
        The docstring should encapsulate the technical and functional purpose of the code. 
        Include details about inputs, outputs, function calls, logical flow, and any other relevant information.

        Here are the code snippets:
        {code_snippets}

        {format_instructions}
        """

        # Prepare the code snippets
        code_snippets = ""
        for request in batch:
            code_snippets += f"node_id: {request.node_id} \n```\n{request.text}\n```\n\n "

        output_parser = PydanticOutputParser(pydantic_object=DocstringResponse)
        
        chat_prompt = ChatPromptTemplate.from_template(
            template=base_prompt,
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            }
        )

    
        import time

        start_time = time.time()
        print("Starting the inference process...")
        total_word_count = len(base_prompt.split()) + sum(len(request.text.split()) for request in batch)
        print(f"Request contains {total_word_count} words.")

        chain = chat_prompt | self.llm | output_parser
        result = await chain.ainvoke({"code_snippets": code_snippets})
        end_time = time.time()

        print(f"Start Time: {start_time}, End Time: {end_time}, Total Time Taken: {end_time - start_time} seconds")
        return str(result)


    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    async def update_neo4j_with_docstrings(self, repo_id: str, docstrings: DocstringResponse):
        with self.driver.session() as session:
            batch_size = 300
            docstring_list = [{"node_id": n.node_id, "docstring": n.docstring, "embedding": self.generate_embedding(n.docstring)} for n in docstrings["docstrings"]]
            
            for i in range(0, len(docstring_list), batch_size):
                batch = docstring_list[i:i+batch_size]
                session.run(
                    """
                    UNWIND $batch AS item
                    MATCH (n:NODE {repoId: $repo_id, node_id: item.node_id})
                    SET n.docstring = item.docstring,
                        n.embedding = item.embedding
                    """,
                    batch=batch, repo_id=repo_id
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

    async def query_vector_index(self, query: str, node_ids: Optional[List[str]] = None, top_k: int = 5) -> List[Dict]:
        embedding = self.generate_embedding(query)
        
        with self.driver.session() as session:
            if node_ids:
                # Fetch neighboring nodes and perform context-aware search
                result = session.run(
                    """
                    MATCH (n:NODE)
                    WHERE n.node_id IN $node_ids
                    CALL {
                        WITH n
                        MATCH (n)-[*1..4]-(neighbor:NODE)
                        RETURN COLLECT(DISTINCT neighbor) AS neighbors
                    }
                    WITH COLLECT(DISTINCT n) + REDUCE(acc = [], neighbors IN COLLECT(neighbors) | acc + neighbors) AS context_nodes
                    UNWIND context_nodes AS context_node
                    WITH context_node, gds.similarity.cosine(context_node.embedding, $embedding) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                    RETURN context_node.node_id AS node_id, context_node.docstring AS docstring, similarity
                    """,
                    node_ids=node_ids, embedding=embedding, top_k=top_k
                )
            else:
                # Perform simple vector search
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('docstring_embedding', $top_k, $embedding)
                    YIELD node, score
                    RETURN node.node_id AS node_id, node.docstring AS docstring, score AS similarity
                    """,
                    embedding=embedding, top_k=top_k
                )
            
            return [dict(record) for record in result]

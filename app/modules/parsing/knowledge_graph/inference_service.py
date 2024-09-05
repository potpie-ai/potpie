import asyncio
from typing import Dict, List
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from app.core.config_provider import config_provider
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

class DocstringRequest(BaseModel):
    node_id: str
    text: str

class DocstringResponse(BaseModel):
    node_id: str
    docstring: str = Field(..., max_length=1000)

class InferenceService:
    def __init__(self):
        neo4j_config = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"])
        )
        self.llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)

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

    async def generate_docstrings(self, repo_id: str) -> Dict[str, str]:
        nodes = self.fetch_graph(repo_id)
        batches = self.batch_nodes(nodes)
        all_docstrings = {}

        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

        async def process_batch(batch):
            async with semaphore:
                prompt = self.create_prompt(batch)
                response = await self.generate_response(prompt)
                return self.parse_response(response)

        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        for result in results:
            all_docstrings.update(result)

        return all_docstrings

    def create_prompt(self, batch: List[DocstringRequest]) -> str:
        prompt = """
        Generate a detailed technical docstring for each of the following code snippets. 
        The docstring should encapsulate the technical and functional purpose of the code. 
        Provide the output in the following format:

        node_id1:
        ```
        docstring1
        ```

        node_id2:
        ```
        docstring2
        ```

        Here are the code snippets:

        """
        for request in batch:
            prompt += f"{request.node_id}:\n```\n{request.text}\n```\n\n"

        return prompt

    async def generate_response(self, prompt: str) -> str:
        output_parser = PydanticOutputParser(pydantic_object=DocstringResponse)
        chat_prompt = ChatPromptTemplate.from_template(template=prompt)
        chain = chat_prompt | self.llm | output_parser
        result = await chain.ainvoke({"content": prompt})
        return str(result)

    def parse_response(self, response: str) -> Dict[str, str]:
        docstrings = {}
        current_node_id = None
        current_docstring = []

        for line in response.split('\n'):
            if line.endswith(':'):
                if current_node_id and current_docstring:
                    docstrings[current_node_id] = '\n'.join(current_docstring).strip()
                current_node_id = line[:-1]
                current_docstring = []
            elif line.strip() == '```':
                continue
            else:
                current_docstring.append(line)

        if current_node_id and current_docstring:
            docstrings[current_node_id] = '\n'.join(current_docstring).strip()

        return docstrings

    async def update_neo4j_with_docstrings(self, repo_id: str, docstrings: Dict[str, str]):
        async with self.driver.session() as session:
            for node_id, docstring in docstrings.items():
                await session.run(
                    "MATCH (n:NODE {repoId: $repo_id, node_id: $node_id}) "
                    "SET n.docstring = $docstring",
                    repo_id=repo_id, node_id=node_id, docstring=docstring
                )

    async def run_inference(self, repo_id: str):
        docstrings = await self.generate_docstrings(repo_id)
        await self.update_neo4j_with_docstrings(repo_id, docstrings)

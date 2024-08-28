import asyncio
import hashlib
import logging

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.search.search_service import SearchService


class CodeGraphService:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, db: Session):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.db = db

    @staticmethod
    def generate_node_id(path: str, user_id: str):
        # Concatenate path and signature
        combined_string = f"{user_id}:{path}"

        # Create a SHA-1 hash of the combined string
        hash_object = hashlib.md5()
        hash_object.update(combined_string.encode("utf-8"))

        # Get the hexadecimal representation of the hash
        node_id = hash_object.hexdigest()

        return node_id

    def close(self):
        self.driver.close()

    def create_and_store_graph(self, repo_dir, project_id, user_id):
        # Create the graph using RepoMap
        self.repo_map = RepoMap(
            root=repo_dir,
            verbose=True,
            main_model=SimpleTokenCounter(),
            io=SimpleIO(),
        )

        nx_graph = self.repo_map.create_graph(repo_dir)

        with self.driver.session() as session:
            # Create nodes
            import time

            start_time = time.time()  # Start timing
            node_count = nx_graph.number_of_nodes()
            logging.info(f"Creating {node_count} nodes")

            # Initialize SearchService
            search_service = SearchService(self.db)

            # Batch insert nodes
            batch_size = 300
            for i in range(0, node_count, batch_size):
                batch_nodes = list(nx_graph.nodes(data=True))[i : i + batch_size]
                nodes_to_create = []
                for node in batch_nodes:
                    node_data = {
                        "name": node[0],
                        "file": node[1].get("file", ""),
                        "start_line": node[1].get("line", -1),
                        "repoId": project_id,
                        "node_id": CodeGraphService.generate_node_id(
                            node[1].get("file", ""), user_id
                        ),
                        "entityId": user_id,
                    }
                    nodes_to_create.append(node_data)
                    # Create search index for each node
                    asyncio.run(
                        search_service.create_search_index(project_id, node_data)
                    )

                session.run(
                    "UNWIND $nodes AS node "
                    "CREATE (d:Definition {name: node.name, file: node.file, start_line: node.start_line, repoId: node.repoId, node_id: node.node_id, entityId: node.entityId})",
                    nodes=nodes_to_create,
                )

            # Commit the search indices
            asyncio.run(search_service.commit_indices())

            relationship_count = nx_graph.number_of_edges()
            logging.info(f"Creating {relationship_count} relationships")

            # Create relationships in batches
            for i in range(0, relationship_count, batch_size):
                batch_edges = list(nx_graph.edges(data=True))[i : i + batch_size]
                session.run(
                    """
                    UNWIND $edges AS edge
                    MATCH (s:Definition {name: edge.source}), (t:Definition {name: edge.target})
                    CREATE (s)-[:REFERENCES {type: edge.type}]->(t)
                    """,
                    edges=[
                        {"source": edge[0], "target": edge[1], "type": edge[2]["type"]}
                        for edge in batch_edges
                    ],
                )

            end_time = time.time()  # End timing
            logging.info(
                f"Time taken to create graph and search index: {end_time - start_time:.2f} seconds"
            )  # Log time taken

    def query_graph(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]


class SimpleIO:
    def read_text(self, fname):
        with open(fname, "r") as f:
            return f.read()

    def tool_error(self, message):
        logging.error(f"Error: {message}")

    def tool_output(self, message):
        logging.info(message)


class SimpleTokenCounter:
    def token_count(self, text):
        return len(text.split())

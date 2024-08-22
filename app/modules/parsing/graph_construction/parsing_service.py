from neo4j import GraphDatabase
from app.modules.parsing.graph_construction.parsing_helper import RepoMap
import os 
import traceback
import uuid
from blar_graph.db_managers import Neo4jManager
from blar_graph.graph_construction.core.graph_builder import GraphConstructor

class SimpleIO:
    def read_text(self, fname):
        with open(fname, 'r') as f:
            return f.read()

    def tool_error(self, message):
        print(f"Error: {message}")

    def tool_output(self, message):
        print(message)
        
class SimpleTokenCounter:
    def token_count(self, text):
        return len(text.split())
    
class CodeGraphService:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
    def close(self):
        self.driver.close()

    def create_and_store_graph(self, repo_dir):
        # Create the graph using RepoMap
        self.repo_map = RepoMap(
            root=repo_dir,
            verbose=True,
            main_model=SimpleTokenCounter(),
            io=SimpleIO(),
        )

        nx_graph = self.repo_map.create_graph(repo_dir)

        with self.driver.session() as session:
            # Clear existing data in Neo4j
            session.run("MATCH (n) DETACH DELETE n")

            # Create nodes
            import time

            start_time = time.time()  # Start timing
            node_count = nx_graph.number_of_nodes()
            print(f"Creating {node_count} nodes")

            # Batch insert nodes
            batch_size = 300
            for i in range(0, node_count, batch_size):
                batch_nodes = list(nx_graph.nodes(data=True))[i:i + batch_size]
                session.run(
                    "UNWIND $nodes AS node "
                    "CREATE (d:Definition {name: node.name, file: node.file, line: node.line})",
                    nodes=[{'name': node[0], 'file': node[1].get('file', ''), 'line': node[1].get('line', -1)} for node in batch_nodes]
                )

            relationship_count = nx_graph.number_of_edges()
            print(f"Creating {relationship_count} relationships")

            # Create relationships in batches
            for i in range(0, relationship_count, batch_size):
                batch_edges = list(nx_graph.edges(data=True))[i:i + batch_size]
                session.run(
                    """
                    UNWIND $edges AS edge
                    MATCH (s:Definition {name: edge.source}), (t:Definition {name: edge.target})
                    CREATE (s)-[:REFERENCES {type: edge.type}]->(t)
                    """,
                    edges=[{'source': edge[0], 'target': edge[1], 'type': edge[2]['type']} for edge in batch_edges]
                )

            end_time = time.time()  # End timing
            print(f"Time taken to create graph: {end_time - start_time:.2f} seconds")  # Log time taken


    def query_graph(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

# Example usage
async def analyze_directory(repo_dir):
    service = CodeGraphService("bolt://localhost:7687", "neo4j", "mysecretpassword")
    
    try:

        # repo_dir = "/Users/dhirenmathur/Downloads/dispatch-master"
        # repo_dir = "/Users/dhirenmathur/Downloads/litellm-main"
        # repo_dir = "/Users/dhirenmathur/Downloads/simplQ-backend-master"
        # Basic language detection based on file extensions and character count
        def detect_repo_language(repo_dir):
            lang_count = {"python": 0, "javascript": 0, "typescript": 0, "other": 0}
            total_chars = 0

            for root, _, files in os.walk(repo_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            total_chars += len(content)
                            if ext == '.py':
                                lang_count["python"] += 1
                            elif ext in ['.js', '.jsx']:
                                lang_count["javascript"] += 1
                            elif ext in ['.ts', '.tsx']:
                                lang_count["typescript"] += 1
                            else:
                                lang_count["other"] += 1
                    except (UnicodeDecodeError, FileNotFoundError):
                        continue

            # Determine the predominant language based on counts
            if lang_count["python"] > lang_count["javascript"] and lang_count["python"] > lang_count["typescript"]:
                return "python"
            elif lang_count["javascript"] > lang_count["python"] and lang_count["javascript"] > lang_count["typescript"]:
                return "javascript"
            elif lang_count["typescript"] > lang_count["python"] and lang_count["typescript"] > lang_count["javascript"]:
                return "typescript"
            else:
                return "other"

        repo_lang = detect_repo_language(repo_dir)
        
        if repo_lang not in ["python", "javascript", "typescript"]:
            service.create_and_store_graph(repo_dir)
        
            # Example query
            result = service.query_graph("MATCH (n:Definition) RETURN n.name, n.file, n.line LIMIT 5")
            print(result)
        else: 
        
   

            repoId = str(uuid.uuid4())
            entityId = str(uuid.uuid4())
            graph_manager = Neo4jManager(repoId, entityId)

            try:
                graph_constructor = GraphConstructor(graph_manager, entityId)
                n,r = graph_constructor.build_graph(repo_dir)
                graph_manager.save_graph(n,r)
                graph_manager.close()
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                graph_manager.close()
    finally:
        service.close()
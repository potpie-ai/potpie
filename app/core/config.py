from dotenv import load_dotenv
import os

load_dotenv()

neo4j_config = {
    "uri": os.getenv("NEO4J_URI"),
    "username": os.getenv("NEO4J_USERNAME"),
    "password": os.getenv("NEO4J_PASSWORD"),
}
github_key = os.getenv("GITHUB_PRIVATE_KEY")
import os

from dotenv import load_dotenv

load_dotenv()


class ConfigProvider:
    def __init__(self):
        self.neo4j_config = {
            "uri": os.getenv("NEO4J_URI"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD"),
        }
        self.github_key = os.getenv("GITHUB_PRIVATE_KEY")

    def get_neo4j_config(self):
        return self.neo4j_config

    def get_github_key(self):
        return self.github_key


config_provider = ConfigProvider()

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

    def get_demo_repo_list(self):
        return [
            {
                "id": "demo4",
                "name": "mem0",
                "full_name": "mem0ai/mem0",
                "private": False,
                "url": "https://github.com/mem0ai/mem0",
                "owner": "mem0ai",
            },
            {
                "id": "demo3",
                "name": "gateway",
                "full_name": "Portkey-AI/gateway",
                "private": False,
                "url": "https://github.com/Portkey-AI/gateway",
                "owner": "Portkey-AI",
            },
            {
                "id": "demo2",
                "name": "crewAI",
                "full_name": "crewAIInc/crewAI",
                "private": False,
                "url": "https://github.com/crewAIInc/crewAI",
                "owner": "crewAIInc",
            },
            {
                "id": "demo1",
                "name": "langchain",
                "full_name": "langchain-ai/langchain",
                "private": False,
                "url": "https://github.com/langchain-ai/langchain",
                "owner": "langchain-ai",
            },
        ]

    def get_redis_url(self):
        redishost = os.getenv("REDISHOST", "localhost")
        redisport = int(os.getenv("REDISPORT", 6379))
        redisuser = os.getenv("REDISUSER", "")
        redispassword = os.getenv("REDISPASSWORD", "")
        # Construct the Redis URL
        if redisuser and redispassword:
            redis_url = f"redis://{redisuser}:{redispassword}@{redishost}:{redisport}/0"
        else:
            redis_url = f"redis://{redishost}:{redisport}/0"
        return redis_url



config_provider = ConfigProvider()

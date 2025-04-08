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
        self.is_development_mode = os.getenv("isDevelopmentMode", "disabled")

    def get_neo4j_config(self):
        return self.neo4j_config

    def get_github_key(self):
        return self.github_key

    def get_demo_repo_list(self):
        return [
            {
                "id": "demo8",
                "name": "langchain",
                "full_name": "langchain-ai/langchain",
                "private": False,
                "url": "https://github.com/langchain-ai/langchain",
                "owner": "langchain-ai",
            },
            {
                "id": "demo6",
                "name": "cal.com",
                "full_name": "calcom/cal.com",
                "private": False,
                "url": "https://github.com/calcom/cal.com",
                "owner": "calcom",
            },
            {
                "id": "demo5",
                "name": "formbricks",
                "full_name": "formbricks/formbricks",
                "private": False,
                "url": "https://github.com/formbricks/formbricks",
                "owner": "formbricks",
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
                "name": "agentops",
                "full_name": "AgentOps-AI/agentops",
                "private": False,
                "url": "https://github.com/AgentOps-AI/agentops",
                "owner": "AgentOps-AI",
            },
            {
                "id": "demo0",
                "name": "agentstack",
                "full_name": "AgentOps-AI/AgentStack",
                "private": False,
                "url": "https://github.com/AgentOps-AI/AgentStack",
                "owner": "AgentOps-AI",
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

    def get_is_development_mode(self):
        return self.is_development_mode == "enabled"


config_provider = ConfigProvider()

from .langchain_agents import LangChainAgent

AGENT_REGISTRY = {}

def register_agent(agent_name: str, agent_class):
    AGENT_REGISTRY[agent_name] = agent_class

def get_agent(agent_name: str):
    agent_class = AGENT_REGISTRY.get(agent_name)
    if not agent_class:
        raise ValueError(f"Agent '{agent_name}' not found.")
    return agent_class()

# Register the LangChain DuckDuckGo agent
register_agent("langchain_duckduckgo", LangChainAgent)
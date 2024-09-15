from typing import List

from app.modules.intelligence.agents.agents_schema import AgentInfo


class AgentsService:
    def __init__(self, db):
        self.db = db

    @classmethod
    def create(cls, db):
        return cls(db)

    async def list_available_agents(self) -> List[AgentInfo]:
        return [
            AgentInfo(
                id="debugging_agent",
                name="Debugging with Knowledge Graph Agent",
                description="An agent specialized in debugging using knowledge graphs.",
            ),
            AgentInfo(
                id="codebase_qna_agent",
                name="Codebase Q&A Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
            ),
            AgentInfo(
                id="code_retrieval_agent",
                name="Code Retrieval Agent",
                description="An agent specialized in retrieving and analyzing code snippets from the knowledge graph based on node names or IDs.",
            ),
            AgentInfo(
                id="code_graph_retrieval_agent",
                name="Code Graph Retrieval Agent",
                description="An agent specialized in retrieving and analyzing code snippets from the knowledge graph based on node names or IDs.",
            ),
            AgentInfo(
                id="unit_test_agent",
                name="Unit Test Agent",
                description="An agent specialized in generating unit tests for code snippets from the knowledge graph based on funtion names",
            ),
            AgentInfo(
                id="code_changes_agent",
                name="Code Changes Agent",
                description="An agent specialized in generating detailed analysis of code changes in your current branch compared to default branch.",
            ),
        ]

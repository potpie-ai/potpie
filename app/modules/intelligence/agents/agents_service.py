import os
from typing import List

from app.modules.intelligence.agents.agents_schema import AgentInfo


class AgentsService:
    def __init__(self, db):
        self.project_path = os.getenv("PROJECT_PATH", "projects/")
        self.db = db

    @classmethod
    def create(cls, db):
        return cls(db)

    async def list_available_agents(self) -> List[AgentInfo]:
        return [
            AgentInfo(
                id="codebase_qna_agent",
                name="Codebase Q&A Agent",
                description="An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
            ),
            AgentInfo(
                id="debugging_agent",
                name="Debugging with Knowledge Graph Agent",
                description="An agent specialized in debugging using knowledge graphs.",
            ),
            AgentInfo(
                id="unit_test_agent",
                name="Unit Test Agent",
                description="An agent specialized in generating unit tests for code snippets for given function names",
            ),
            AgentInfo(
                id="integration_test_agent",
                name="Integration Test Agent",
                description="An agent specialized in generating integration tests for code snippets from the knowledge graph based on given function names of entry points. Works best with Py, JS, TS",
            ),
            AgentInfo(
                id="LLD_agent",
                name="Low-Level Design Agent",
                description="An agent specialized in generating a low-level design plan for implementing a new feature.",
            ),
            AgentInfo(
                id="code_changes_agent",
                name="Code Changes Agent",
                description="An agent specialized in generating detailed analysis of code changes in your current branch compared to default branch. Works best with Py, JS, TS",
            ),
        ]

    def format_citations(self, citations: List[str]) -> List[str]:
        cleaned_citations = []
        for citation in citations:
            cleaned_citations.append(
                citation.split(self.project_path, 1)[-1].split("/", 2)[-1]
                if self.project_path in citation
                else citation
            )
        return cleaned_citations

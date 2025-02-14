import os
from typing import List

from app.modules.intelligence.prompts.prompt_service import PromptService


class BaseAgentService:
    def __init__(self, db):
        self.project_path = os.getenv("PROJECT_PATH", "projects/")
        self.db = db
        self.prompt_service = PromptService(db)

    def format_citations(self, citations: List[str]) -> List[str]:
        cleaned_citations = []
        for citation in citations:
            cleaned_citations.append(
                citation.split(self.project_path, 1)[-1].split("/", 2)[-1]
                if self.project_path in citation
                else citation
            )
        return cleaned_citations

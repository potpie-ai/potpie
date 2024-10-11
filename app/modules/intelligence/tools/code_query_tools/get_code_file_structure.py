from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.github.github_service import GithubService


class RepoStructureService:
    def __init__(self, db: Session):
        self.github_service = GithubService(db)

    def fetch_repo_structure(self, repo_id: str) -> str:
        return self.github_service.get_project_structure(repo_id)

    async def run(self, repo_id: str) -> str:
        return self.fetch_repo_structure(repo_id)

    def run_tool(self, repo_id: str) -> str:
        return self.fetch_repo_structure(repo_id)


def get_code_file_structure_tool(db: Session) -> StructuredTool:
    return StructuredTool(
        name="get_code_file_structure",
        description="Get the file structure of a project",
        coroutine=RepoStructureService(db).run,
        func=RepoStructureService(db).run_tool,
    )

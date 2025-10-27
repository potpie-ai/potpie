import logging
import os
import random
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from github import Github
from github.GithubException import GithubException
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool

from app.modules.code_provider.provider_factory import CodeProviderFactory


class CodeProviderCreateBranchInput(BaseModel):
    """Input for creating a new branch in a repository."""

    repo_name: str = Field(
        ..., description="The full name of the repository (e.g., 'username/repo_name')"
    )
    base_branch: str = Field(
        ..., description="The name of the branch to branch from (e.g., 'main')"
    )
    new_branch_name: str = Field(
        ..., description="The name of the new branch to be created"
    )


class CodeProviderCreateBranchTool:
    """Tool for creating a new branch in a repository."""

    name: str = "Create a new branch in code repository"
    description: str = """
    Create a new branch in a code repository.
    Useful for starting a new feature, bugfix, or any work that requires a separate branch.
    The tool will create the branch from the specified base branch.
    """
    args_schema: Type[BaseModel] = CodeProviderCreateBranchInput
    gh_token_list: List[str] = []

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id

    @classmethod
    def initialize_tokens(cls):
        token_string = os.getenv("GH_TOKEN_LIST", "")
        cls.gh_token_list = [
            token.strip() for token in token_string.split(",") if token.strip()
        ]
        if not cls.gh_token_list:
            raise ValueError(
                "GitHub token list is empty or not set in environment variables"
            )
        logging.info(f"Initialized {len(cls.gh_token_list)} GitHub tokens")

    @classmethod
    def get_public_github_instance(cls):
        if not cls.gh_token_list:
            cls.initialize_tokens()
        token = random.choice(cls.gh_token_list)
        return Github(token)

    def _get_github_client(self, repo_name: str) -> Github:
        """Get code provider client using provider factory."""
        try:
            logging.info(f"[CREATE_BRANCH] Creating provider for repo: {repo_name}")
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
            logging.info(
                f"[CREATE_BRANCH] Provider created successfully, type: {type(provider).__name__}"
            )
            logging.info(
                f"[CREATE_BRANCH] Client object: {type(provider.client).__name__}"
            )
            return provider.client
        except Exception as e:
            logging.error(
                f"[CREATE_BRANCH] Failed to get client: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise Exception(
                f"Repository {repo_name} not found or inaccessible: {str(e)}"
            )

    def _run(
        self,
        repo_name: str,
        base_branch: str,
        new_branch_name: str,
    ) -> Dict[str, Any]:
        """
        Create a new branch in a repository.

        Args:
            repo_name: The full name of the repository (e.g., 'username/repo_name')
            base_branch: The name of the branch to branch from (e.g., 'main')
            new_branch_name: The name of the new branch to be created

        Returns:
            Dict containing the result of the branch creation operation
        """
        logging.info(
            f"[CREATE_BRANCH] Starting branch creation: repo={repo_name}, base={base_branch}, new={new_branch_name}"
        )
        try:
            # Initialize GitHub client
            logging.info(f"[CREATE_BRANCH] Getting client for repo: {repo_name}")
            g = self._get_github_client(repo_name)

            # Get the actual repo name for API calls (handles GitBucket conversion)
            from app.modules.parsing.utils.repo_name_normalizer import (
                get_actual_repo_name_for_lookup,
            )
            import os

            provider_type = os.getenv("CODE_PROVIDER", "github").lower()
            actual_repo_name = get_actual_repo_name_for_lookup(repo_name, provider_type)
            logging.info(
                f"[CREATE_BRANCH] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
            )

            repo = g.get_repo(actual_repo_name)
            logging.info(f"[CREATE_BRANCH] Successfully got repo object: {repo.name}")

            # Get the base branch reference
            try:
                logging.info(
                    f"[CREATE_BRANCH] Attempting to get ref for base branch: heads/{base_branch}"
                )
                base_ref = repo.get_git_ref(f"heads/{base_branch}")
                logging.info(
                    f"[CREATE_BRANCH] Successfully got base branch ref: {base_ref.ref}, sha: {base_ref.object.sha}"
                )
            except GithubException as e:
                logging.error(
                    f"[CREATE_BRANCH] Failed to get base branch '{base_branch}': status={e.status}, data={e.data}, message={str(e)}"
                )
                return {
                    "success": False,
                    "error": f"Base branch '{base_branch}' not found: {str(e)}",
                    "status_code": e.status if hasattr(e, "status") else None,
                    "details": e.data if hasattr(e, "data") else None,
                }

            # Check if the new branch already exists
            try:
                logging.info(
                    f"[CREATE_BRANCH] Checking if new branch already exists: heads/{new_branch_name}"
                )
                repo.get_git_ref(f"heads/{new_branch_name}")
                logging.warning(
                    f"[CREATE_BRANCH] Branch '{new_branch_name}' already exists"
                )
                return {
                    "success": False,
                    "error": f"Branch '{new_branch_name}' already exists",
                }
            except GithubException as e:
                if e.status != 404:
                    # If error is not "Not Found", it's an unexpected error
                    logging.error(
                        f"[CREATE_BRANCH] Unexpected error checking branch existence: status={e.status}, data={e.data}"
                    )
                    return {
                        "success": False,
                        "error": f"Error checking branch existence: {str(e)}",
                        "status_code": e.status,
                    }
                # 404 means the branch doesn't exist, which is what we want
                logging.info(
                    f"[CREATE_BRANCH] Branch '{new_branch_name}' does not exist (404), proceeding with creation"
                )

            # Create the new branch
            logging.info(
                f"[CREATE_BRANCH] Creating new branch: refs/heads/{new_branch_name} from sha: {base_ref.object.sha}"
            )
            new_ref = repo.create_git_ref(
                ref=f"refs/heads/{new_branch_name}", sha=base_ref.object.sha
            )
            logging.info(
                f"[CREATE_BRANCH] Successfully created branch: {new_ref.ref}, sha: {new_ref.object.sha}"
            )

            result = {
                "success": True,
                "operation": "create_branch",
                "base_branch": base_branch,
                "new_branch": new_branch_name,
                "sha": new_ref.object.sha,
                "url": f"https://github.com/{repo_name}/tree/{new_branch_name}",
            }
            logging.info(f"[CREATE_BRANCH] Returning success result: {result}")
            return result

        except GithubException as e:
            logging.error(
                f"[CREATE_BRANCH] GithubException caught: status={e.status}, data={e.data}, message={str(e)}"
            )
            return {
                "success": False,
                "error": f"GitHub API error: {str(e)}",
                "status_code": e.status if hasattr(e, "status") else None,
                "data": e.data if hasattr(e, "data") else None,
            }
        except Exception as e:
            logging.error(
                f"[CREATE_BRANCH] Unexpected exception: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            return {"success": False, "error": f"Error creating branch: {str(e)}"}

    async def _arun(
        self,
        repo_name: str,
        base_branch: str,
        new_branch_name: str,
    ) -> Dict[str, Any]:
        """Async implementation of the tool."""
        return self._run(
            repo_name=repo_name,
            base_branch=base_branch,
            new_branch_name=new_branch_name,
        )


def code_provider_create_branch_tool(
    sql_db: Session, user_id: str
) -> Optional[StructuredTool]:
    from app.modules.code_provider.provider_factory import has_code_provider_credentials

    if not has_code_provider_credentials():
        logging.warning(
            "No code provider credentials configured. Please set CODE_PROVIDER_TOKEN, "
            "GH_TOKEN_LIST, GITHUB_APP_ID, or CODE_PROVIDER_USERNAME/PASSWORD."
        )
        return None

    tool_instance = CodeProviderCreateBranchTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="Create a new branch",
        description="""
        Create a new branch in a repository.
        Useful for starting a new feature, bugfix, or any work that requires a separate branch.
        The tool will create the branch from the specified base branch.
        """,
        args_schema=CodeProviderCreateBranchInput,
    )

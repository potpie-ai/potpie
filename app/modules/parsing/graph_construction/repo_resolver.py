import os
import logging
from typing import Optional
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails

logger = logging.getLogger(__name__)

class RepoResolver:
    @staticmethod
    def resolve(repository_identifier: str, branch_name: Optional[str] = None, commit_id: Optional[str] = None) -> RepoDetails:
        is_local = False
        repo_path = None
        repo_name = repository_identifier

        # Auto-detect if repository_identifier is a filesystem path
        # We check for common path indicators or if it's an existing directory
        if (
            os.path.isabs(repository_identifier)
            or repository_identifier.startswith(("~", "./", "../"))
            or os.path.isdir(os.path.expanduser(repository_identifier))
        ):
            is_local = True
            repo_path = repository_identifier
            # Normalize path separators and get the last component as name
            repo_name = os.path.basename(os.path.normpath(repo_path))
            logger.info(
                f"Auto-detected filesystem path: repo_path={repo_path}, repo_name={repo_name}"
            )
        
        return RepoDetails(
            repository_identifier=repository_identifier,
            repo_name=repo_name,
            repo_path=repo_path,
            branch_name=branch_name,
            commit_id=commit_id,
            is_local=is_local
        )

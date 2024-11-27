import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import git
from fastapi import HTTPException
from redis import Redis
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class LocalRepoService:
    def __init__(self, db: Session):
        self.db = db
        self.project_manager = ProjectService(db)
        self.projects_dir = os.path.join(
            os.getcwd(), "projects"
        )  # Define the projects directory
        self.redis = Redis.from_url(config_provider.get_redis_url())
        self.max_workers = 10
        self.max_depth = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def get_repo(self, repo_name: str) -> git.Repo:
        repo_path = os.path.join(self.projects_dir, repo_name)
        if not os.path.exists(repo_path):
            raise HTTPException(
                status_code=404, detail=f"Local repository {repo_name} not found"
            )
        return git.Repo(repo_path)

    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        start_line: int,
        end_line: int,
        branch_name: str,
    ) -> str:
        logger.info(f"Attempting to access file: {file_path} in repo: {repo_name}")
        try:
            repo = self.get_repo(repo_name)
            repo.git.checkout(branch_name)
            file_full_path = os.path.join(self.projects_dir, repo_name, file_path)
            with open(file_full_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                if (start_line == end_line == 0) or (start_line == end_line == None):
                    return "".join(lines)
                start = start_line - 2 if start_line - 2 > 0 else 0
                selected_lines = lines[start:end_line]
                return "".join(selected_lines)
        except Exception as e:
            logger.error(
                f"Error processing file content for {repo_name}/{file_path}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file content: {str(e)}",
            )

    async def get_branch_list(self, repo_name: str):
        try:
            repo = self.get_repo(repo_name)
            branches = [head.name for head in repo.heads]
            return {"branches": branches}
        except Exception as e:
            logger.error(
                f"Error fetching branches for repo {repo_name}: {str(e)}", exc_info=True
            )
            raise HTTPException(
                status_code=404,
                detail=f"Repository not found or error fetching branches: {str(e)}",
            )

    async def get_project_structure_async(
        self, project_id: str, path: Optional[str] = None
    ) -> str:
        project = await self.project_manager.get_project_from_db_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        repo_path = project["repo_path"]
        print(911, project)
        if not repo_path:
            raise HTTPException(
                status_code=400, detail="Project has no associated local repository"
            )

        try:
            repo = self.get_repo(repo_path)
            structure = await self._fetch_repo_structure_async(
                repo, repo_path or "", current_depth=0, base_path=path
            )
            formatted_structure = self._format_tree_structure(structure)
            return formatted_structure
        except Exception as e:
            logger.error(
                f"Error fetching project structure for {repo_path}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch project structure: {str(e)}"
            )

    async def _fetch_repo_structure_async(
        self,
        repo: Any,
        path: str = "",
        current_depth: int = 0,
        base_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        exclude_extensions = [
            "png",
            "jpg",
            "jpeg",
            "gif",
            "bmp",
            "tiff",
            "webp",
            "ico",
            "svg",
            "mp4",
            "avi",
            "mov",
            "wmv",
            "flv",
            "ipynb",
            "zlib",
        ]

        # Calculate current depth relative to base_path
        if base_path:
            # If we have a base_path, calculate depth relative to it
            relative_path = path[len(base_path) :].strip("/")
            current_depth = len(relative_path.split("/")) if relative_path else 0
        else:
            # If no base_path, calculate depth from root
            current_depth = len(path.split("/")) if path else 0

        # If we've reached max depth, return truncated indicator
        if current_depth >= self.max_depth:
            return {
                "type": "directory",
                "name": path.split("/")[-1] or repo.name,
                "children": [{"type": "file", "name": "...", "path": "truncated"}],
            }

        structure = {
            "type": "directory",
            "name": path.split("/")[-1] or repo.name,
            "children": [],
        }

        try:
            contents = await asyncio.get_event_loop().run_in_executor(
                self.executor, repo.get_contents, path
            )

            if not isinstance(contents, list):
                contents = [contents]

            # Filter out files with excluded extensions
            contents = [
                item
                for item in contents
                if item.type == "dir"
                or not any(item.name.endswith(ext) for ext in exclude_extensions)
            ]

            tasks = []
            for item in contents:
                # Only process items within the base_path if it's specified
                if base_path and not item.path.startswith(base_path):
                    continue

                if item.type == "dir":
                    task = self._fetch_repo_structure_async(
                        repo,
                        item.path,
                        current_depth=current_depth,
                        base_path=base_path,
                    )
                    tasks.append(task)
                else:
                    structure["children"].append(
                        {
                            "type": "file",
                            "name": item.name,
                            "path": item.path,
                        }
                    )

            if tasks:
                children = await asyncio.gather(*tasks)
                structure["children"].extend(children)

        except Exception as e:
            logger.error(f"Error fetching contents for path {path}: {str(e)}")

        return structure

    def _format_tree_structure(
        self, structure: Dict[str, Any], root_path: str = ""
    ) -> str:
        """
        Creates a clear hierarchical structure using simple nested dictionaries.

        Args:
            self: The instance object
            structure: Dictionary containing name and children
            root_path: Optional root path string (unused but kept for signature compatibility)
        """

        def _format_node(node: Dict[str, Any], depth: int = 0) -> List[str]:
            output = []

            indent = "  " * depth
            if depth > 0:  # Skip root name
                output.append(f"{indent}{node['name']}")

            if "children" in node:
                children = sorted(node.get("children", []), key=lambda x: x["name"])
                for child in children:
                    output.extend(_format_node(child, depth + 1))

            return output

        return "\n".join(_format_node(structure))

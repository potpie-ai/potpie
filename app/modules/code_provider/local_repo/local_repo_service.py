import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import git
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class LocalRepoService:
    def __init__(self, db: Session):
        self.db = db
        self.project_manager = ProjectService(db)
        self.projects_dir = os.path.join(os.getcwd(), "projects")
        self.max_workers = 10
        self.max_depth = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def get_repo(self, repo_path: str) -> git.Repo:
        if not os.path.exists(repo_path):
            raise HTTPException(
                status_code=404, detail=f"Local repository at {repo_path} not found"
            )
        return git.Repo(repo_path)

    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        start_line: int,
        end_line: int,
        branch_name: str,
        project_id: str,
    ) -> str:
        logger.info(
            f"Attempting to access file: {file_path} for project ID: {project_id}"
        )
        try:
            project = self.project_manager.get_project_from_db_by_id_sync(project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            repo_path = project["repo_path"]
            if not repo_path:
                raise HTTPException(
                    status_code=400, detail="Project has no associated local repository"
                )

            repo = self.get_repo(repo_path)
            repo.git.checkout(branch_name)
            file_full_path = os.path.join(repo_path, file_path)
            with open(file_full_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                if (start_line == end_line == 0) or (start_line == end_line == None):
                    return "".join(lines)
                start = start_line - 2 if start_line - 2 > 0 else 0
                selected_lines = lines[start:end_line]
                return "".join(selected_lines)
        except Exception as e:
            logger.error(
                f"Error processing file content for project ID {project_id}, file {file_path}: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file content: {str(e)}",
            )

    async def get_project_structure_async(
        self, project_id: str, path: Optional[str] = None
    ) -> str:
        project = await self.project_manager.get_project_from_db_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        repo_path = project["repo_path"]
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
                self.executor, self._get_contents, path
            )

            if not isinstance(contents, list):
                contents = [contents]

            # Filter out files with excluded extensions
            contents = [
                        item
                        for item in contents
                        if item['type'] == "dir"
                        or not any(item['name'].endswith(ext) for ext in exclude_extensions)
                    ]

            tasks = []
            for item in contents:
                # Only process items within the base_path if it's specified
                if base_path and not item['path'].startswith(base_path):
                    continue

                if item['type'] == "dir":
                    task = self._fetch_repo_structure_async(
                        repo,
                        item['path'],
                        current_depth=current_depth,
                        base_path=base_path,
                    )
                    tasks.append(task)
                else:
                    structure["children"].append(
                        {
                            "type": "file",
                            "name": item['name'],
                            "path": item['path'],
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

    def get_local_repo_diff(self, repo_path: str, branch_name: str) -> Dict[str, str]:
        try:
            repo = self.get_repo(repo_path)
            repo.git.checkout(branch_name)

            # Determine the default branch name
            default_branch_name = repo.git.symbolic_ref(
                "refs/remotes/origin/HEAD"
            ).split("/")[-1]

            # Get the diff between the current branch and the default branch
            diff = repo.git.diff(f"{default_branch_name}..{branch_name}", unified=0)
            patches_dict = self._parse_diff(diff)
            return patches_dict
        except Exception as e:
            logger.error(
                f"Error computing diff for local repo: {str(e)}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail=f"Error computing diff for local repo: {str(e)}"
            )

    def _parse_diff(self, diff: str) -> Dict[str, str]:
        """
        Parses the git diff output and returns a dictionary of file patches.
        """
        patches_dict = {}
        current_file = None
        patch_lines = []

        for line in diff.splitlines():
            if line.startswith("diff --git"):
                if current_file and patch_lines:
                    patches_dict[current_file] = "\n".join(patch_lines)
                match = re.search(r"b/(.+)", line)
                current_file = match.group(1) if match else None
                patch_lines = []
            elif current_file:
                patch_lines.append(line)

        if current_file and patch_lines:
            patches_dict[current_file] = "\n".join(patch_lines)

        return patches_dict

    def _get_contents(self, path: str) -> Union[List[dict], dict]:
        """
        If the path is a directory, it returns a list of dictionaries,
        each representing a file or subdirectory. If the path is a file, its content is read and returned.
        
        :param path: Relative or absolute path within the local repository.
        :return: A dict if the path is a file (with file content loaded), or a list of dicts if the path is a directory.
        """
        if not isinstance(path, str):
            raise TypeError(f"Expected path to be a string, got {type(path).__name__}")

        if path == "/":
            path = ""
        
        abs_path = os.path.abspath(path)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path '{abs_path}' does not exist.")
        
        if os.path.isdir(abs_path):
            contents = []
            for item in os.listdir(abs_path):
                item_path = os.path.join(abs_path, item)
                if os.path.isdir(item_path):
                    contents.append({
                        "path": item_path,
                        "name": item,
                        "type": "dir",
                        "content": None, #path is a dir, content is not loaded
                        "completed": True
                    })
                elif os.path.isfile(item_path):
                    contents.append({
                        "path": item_path,
                        "name": item,
                        "type": "file",
                        "content": None,  
                        "completed": False
                    })
                else:
                    contents.append({
                        "path": item_path,
                        "name": item,
                        "type": "other",
                        "content": None,
                        "completed": True
                    })
            return contents
        
        elif os.path.isfile(abs_path):
            with open(abs_path, "r", encoding="utf-8") as file:
                file_content = file.read()
            return {
                "path": abs_path,
                "name": os.path.basename(abs_path),
                "type": "file",
                "content": file_content, #path is a file, content is loaded
                "completed": True
            }

        
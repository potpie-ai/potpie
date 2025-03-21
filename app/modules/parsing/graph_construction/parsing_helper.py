import json
import logging
import os
import shutil
import tarfile
from typing import Any, Tuple

import requests
from fastapi import HTTPException
from git import GitCommandError, Repo
from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class ParsingServiceError(Exception):
    """Base exception class for ParsingService errors."""


class ParsingFailedError(ParsingServiceError):
    """Raised when a parsing fails."""


class ParseHelper:
    def __init__(self, db_session: Session):
        self.project_manager = ProjectService(db_session)
        self.db = db_session
        self.github_service = CodeProviderService(db_session)

    @staticmethod
    def get_directory_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size

    async def clone_or_copy_repository(
        self, repo_details: RepoDetails, user_id: str
    ) -> Tuple[Any, str, Any]:
        owner = None
        auth = None
        repo = None

        if repo_details.repo_path:
            if not os.path.exists(repo_details.repo_path):
                raise HTTPException(
                    status_code=400,
                    detail="Local repository does not exist on the given path",
                )
            repo = Repo(repo_details.repo_path)
        else:
            try:
                github, repo = self.github_service.get_repo(repo_details.repo_name)
                owner = repo.owner.login
                if hasattr(github, "get_app_auth"):
                    auth = github.get_app_auth()
            except HTTPException as he:
                raise he
            except Exception as e:
                logger.error(f"Failed to fetch repository: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail="Repository not found or inaccessible on GitHub",
                )

        return repo, owner, auth

    def is_text_file(self, file_path):
        def open_text_file(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)
                return True
            except UnicodeDecodeError:
                return False

        ext = file_path.split(".")[-1]
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
        ]
        include_extensions = [
            "py",
            "js",
            "ts",
            "c",
            "cs",
            "cpp",
            "el",
            "ex",
            "exs",
            "elm",
            "go",
            "java",
            "ml",
            "mli",
            "php",
            "ql",
            "rb",
            "rs",
            "md",
            "txt",
            "json",
            "yaml",
            "yml",
            "toml",
            "ini",
            "cfg",
            "conf",
            "xml",
            "html",
            "css",
            "sh",
            "md",
            "mdx",
            "xsq",
            "proto",
        ]
        if ext in exclude_extensions:
            return False
        elif ext in include_extensions or open_text_file(file_path):
            return True
        else:
            return False

    async def download_and_extract_tarball(
        self, repo, branch, target_dir, auth, repo_details, user_id
    ):
        try:
            tarball_url = repo.get_archive_link("tarball", branch)
            headers = {"Authorization": f"Bearer {auth.token}"} if auth else {}
            response = requests.get(tarball_url, stream=True, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching tarball: {e}")
            return e
        tarball_path = os.path.join(
            target_dir,
            f"{repo.full_name.replace('/', '-').replace('.', '-')}-{branch.replace('/', '-').replace('.', '-')}.tar.gz",
        )

        final_dir = os.path.join(
            target_dir,
            f"{repo.full_name.replace('/', '-').replace('.', '-')}-{branch.replace('/', '-').replace('.', '-')}-{user_id}",
        )

        try:
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with tarfile.open(tarball_path, "r:gz") as tar:
                temp_dir = os.path.join(final_dir, "temp_extract")
                tar.extractall(path=temp_dir)
                extracted_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
                for root, dirs, files in os.walk(extracted_dir):
                    for file in files:
                        if file.startswith("."):
                            continue
                        file_path = os.path.join(root, file)
                        if self.is_text_file(file_path):
                            try:
                                relative_path = os.path.relpath(
                                    file_path, extracted_dir
                                )
                                dest_path = os.path.join(final_dir, relative_path)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy2(file_path, dest_path)
                            except (shutil.Error, OSError) as e:
                                logger.error(f"Error copying file {file_path}: {e}")
                # Remove the temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except OSError as e:
                    logger.error(f"Error removing temporary directory: {e}")
                    pass

        except (IOError, tarfile.TarError, shutil.Error) as e:
            logger.error(f"Error handling tarball: {e}")
            return e
        finally:
            if os.path.exists(tarball_path):
                os.remove(tarball_path)

        return final_dir

    @staticmethod
    def detect_repo_language(repo_dir):
        lang_count = {
            "c_sharp": 0,
            "c": 0,
            "cpp": 0,
            "elisp": 0,
            "elixir": 0,
            "elm": 0,
            "go": 0,
            "java": 0,
            "javascript": 0,
            "ocaml": 0,
            "php": 0,
            "python": 0,
            "ql": 0,
            "ruby": 0,
            "rust": 0,
            "typescript": 0,
            "markdown": 0,
            "xml": 0,
            "other": 0,
        }
        total_chars = 0

        try:
            for root, _, files in os.walk(repo_dir):
                if any(part.startswith(".") for part in root.split(os.sep)):
                    continue

                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            total_chars += len(content)
                            if ext == ".cs":
                                lang_count["c_sharp"] += 1
                            elif ext == ".c":
                                lang_count["c"] += 1
                            elif ext in [".cpp", ".cxx", ".cc"]:
                                lang_count["cpp"] += 1
                            elif ext == ".el":
                                lang_count["elisp"] += 1
                            elif ext == ".ex" or ext == ".exs":
                                lang_count["elixir"] += 1
                            elif ext == ".elm":
                                lang_count["elm"] += 1
                            elif ext == ".go":
                                lang_count["go"] += 1
                            elif ext == ".java":
                                lang_count["java"] += 1
                            elif ext in [".js", ".jsx"]:
                                lang_count["javascript"] += 1
                            elif ext == ".ml" or ext == ".mli":
                                lang_count["ocaml"] += 1
                            elif ext == ".php":
                                lang_count["php"] += 1
                            elif ext == ".py":
                                lang_count["python"] += 1
                            elif ext == ".ql":
                                lang_count["ql"] += 1
                            elif ext == ".rb":
                                lang_count["ruby"] += 1
                            elif ext == ".rs":
                                lang_count["rust"] += 1
                            elif ext in [".ts", ".tsx"]:
                                lang_count["typescript"] += 1
                            elif ext in [".md", ".mdx"]:
                                lang_count["markdown"] += 1
                            elif ext in [".xml", ".xsq"]:
                                lang_count["xml"] += 1
                            else:
                                lang_count["other"] += 1
                    except (
                        UnicodeDecodeError,
                        FileNotFoundError,
                        PermissionError,
                    ) as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
                        continue
        except (TypeError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Error accessing directory '{repo_dir}': {e}")

        # Determine the predominant language based on counts
        predominant_language = max(lang_count, key=lang_count.get)
        return predominant_language if lang_count[predominant_language] > 0 else "other"

    async def setup_project_directory(
        self,
        repo,
        branch,
        auth,
        repo_details,
        user_id,
        project_id=None,  # Change type to str
    ):
        full_name = (
            repo.working_tree_dir.split("/")[-1]
            if isinstance(repo_details, Repo)
            else (
                repo.full_name if hasattr(repo, "full_name") else repo_details.repo_name
            )
        )
        repo_path = getattr(repo_details, "repo_path", None)
        if full_name is None:
            full_name = repo_path.split("/")[-1]
        project = await self.project_manager.get_project_from_db(
            full_name, branch, user_id, repo_path
        )
        if not project:
            project_id = await self.project_manager.register_project(
                full_name,
                branch,
                user_id,
                project_id,
            )
        if repo_path is not None:
            if os.getenv("isDevelopmentMode", "false").lower() == "false":
                raise HTTPException(
                    status_code=400,
                    detail="Passing remote repositories is not allowed in development mode",
                )
            return repo_details.repo_path, project_id
        if isinstance(repo_details, Repo):
            extracted_dir = repo_details.working_tree_dir
            try:
                current_dir = os.getcwd()
                os.chdir(extracted_dir)  # Change to the cloned repo directory
                repo_details.git.checkout(branch)
            except GitCommandError as e:
                logger.error(f"Error checking out branch: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Failed to checkout branch {branch}"
                )
            finally:
                os.chdir(current_dir)  # Restore the original working directory
            branch_details = repo_details.head.commit
            latest_commit_sha = branch_details.hexsha
        else:
            extracted_dir = await self.download_and_extract_tarball(
                repo, branch, os.getenv("PROJECT_PATH"), auth, repo_details, user_id
            )
            branch_details = repo_details.get_branch(branch)
            latest_commit_sha = branch_details.commit.sha

        repo_metadata = ParseHelper.extract_repository_metadata(repo_details)
        repo_metadata["error_message"] = None
        project_metadata = json.dumps(repo_metadata).encode("utf-8")
        ProjectService.update_project(
            self.db,
            project_id,
            properties=project_metadata,
            commit_id=latest_commit_sha,
            status=ProjectStatusEnum.CLONED.value,
        )

        return extracted_dir, project_id

    def extract_repository_metadata(repo):
        if isinstance(repo, Repo):
            metadata = ParseHelper.extract_local_repo_metadata(repo)
        else:
            metadata = ParseHelper.extract_remote_repo_metadata(repo)
        return metadata

    def extract_local_repo_metadata(repo):
        languages = ParseHelper.get_local_repo_languages(repo.working_tree_dir)
        total_bytes = sum(languages.values())

        metadata = {
            "basic_info": {
                "full_name": os.path.basename(repo.working_tree_dir),
                "description": None,
                "created_at": None,
                "updated_at": None,
                "default_branch": repo.head.ref.name,
            },
            "metrics": {
                "size": ParseHelper.get_directory_size(repo.working_tree_dir),
                "stars": None,
                "forks": None,
                "watchers": None,
                "open_issues": None,
            },
            "languages": {
                "breakdown": languages,
                "total_bytes": total_bytes,
            },
            "commit_info": {"total_commits": len(list(repo.iter_commits()))},
            "contributors": {
                "count": len(list(repo.iter_commits("--all"))),
            },
            "topics": [],
        }

        return metadata

    def get_local_repo_languages(path):
        total_bytes = 0
        python_bytes = 0

        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                file_extension = os.path.splitext(filename)[1]
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path)
                total_bytes += file_size
                if file_extension == ".py":
                    python_bytes += file_size

        languages = {}
        if total_bytes > 0:
            languages["Python"] = python_bytes
            languages["Other"] = total_bytes - python_bytes

        return languages

    def extract_remote_repo_metadata(repo):
        languages = repo.get_languages()
        total_bytes = sum(languages.values())

        metadata = {
            "basic_info": {
                "full_name": repo.full_name,
                "description": repo.description,
                "created_at": repo.created_at.isoformat(),
                "updated_at": repo.updated_at.isoformat(),
                "default_branch": repo.default_branch,
            },
            "metrics": {
                "size": repo.size,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "watchers": repo.watchers_count,
                "open_issues": repo.open_issues_count,
            },
            "languages": {
                "breakdown": languages,
                "total_bytes": total_bytes,
            },
            "commit_info": {"total_commits": repo.get_commits().totalCount},
            "contributors": {
                "count": repo.get_contributors().totalCount,
            },
            "topics": repo.get_topics(),
        }

        return metadata

    async def check_commit_status(self, project_id: str) -> bool:
        """
        Check if the current commit ID of the project matches the latest commit ID from the repository.

        Args:
            project_id (str): The ID of the project to check.
        Returns:
            bool: True if the commit IDs match, False otherwise.
        """

        project = await self.project_manager.get_project_from_db_by_id(project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found")
            return False

        current_commit_id = project.get("commit_id")
        repo_name = project.get("project_name")
        branch_name = project.get("branch_name")

        if not repo_name or not branch_name:
            logger.error(
                f"Repository name or branch name not found for project ID {project_id}"
            )
            return False

        if len(repo_name.split("/")) < 2:
            # Local repo, always parse local repos
            return False

        try:
            github, repo = self.github_service.get_repo(repo_name)
            branch = repo.get_branch(branch_name)
            latest_commit_id = branch.commit.sha

            is_up_to_date = current_commit_id == latest_commit_id
            logger.info(
                f"""Project {project_id} commit status for branch {branch_name}: {'Up to date' if is_up_to_date else 'Outdated'}"
            Current commit ID: {current_commit_id}
            Latest commit ID: {latest_commit_id}"""
            )

            return is_up_to_date
        except Exception as e:
            logger.error(
                f"Error fetching latest commit for {repo_name}/{branch_name}: {e}"
            )
            return False

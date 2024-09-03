import json
import logging
import os
import tarfile
from typing import Any, Tuple

import requests
from fastapi import HTTPException
from git import GitCommandError, Repo
from github import Github
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.github.github_service import GithubService
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService


class ParsingServiceError(Exception):
    """Base exception class for ParsingService errors."""


class ParsingFailedError(ParsingServiceError):
    """Raised when a parsing fails."""


class ParseHelper:
    def __init__(self, db_session: Session):
        self.project_manager = ProjectService(db_session)
        self.db = db_session

    @staticmethod
    async def clone_or_copy_repository(
        repo_details: RepoDetails, db: Session, user_id: str
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
            github_service = GithubService(db)

            # First, attempt to get public repository details
            try:
                response, owner = github_service.get_public_github_repo(
                    repo_details.repo_name
                )
                github = Github()
                repo = github.get_repo(repo_details.repo_name)
            except Exception as public_repo_error:
                logging.error(
                    f"Failed to fetch public repository: {str(public_repo_error)}"
                )

                # If public repo fetch fails, try private repo
                try:
                    github, response, auth, owner = (
                        github_service.get_github_repo_details(repo_details.repo_name)
                    )

                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=400, detail="Failed to get installation ID"
                        )

                    app_auth = auth.get_installation_auth(response.json()["id"])
                    github = Github(auth=app_auth)
                    repo = github.get_repo(repo_details.repo_name)
                except Exception as private_repo_error:
                    if isinstance(private_repo_error, HTTPException):
                        raise private_repo_error
                    else:
                        logging.error(
                            f"Failed to fetch private repository: {str(private_repo_error)}"
                        )
                        raise HTTPException(
                            status_code=404, detail="Repository not found on GitHub"
                        )

            if repo is None:
                raise HTTPException(
                    status_code=404, detail="Failed to fetch repository"
                )

        return repo, owner, auth

    async def download_and_extract_tarball(
        self, repo, branch, target_dir, auth, repo_details, user_id
    ):
        try:
            tarball_url = repo_details.get_archive_link("tarball", branch)
            headers = {}
            if auth is not None:
                headers = {"Authorization": f"{auth.token}"}
            response = requests.get(
                tarball_url,
                stream=True,
                headers=headers,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching tarball: {e}")
            return e

        tarball_path = os.path.join(
            target_dir, f"{repo.full_name.replace('/', '-')}-{branch}.tar.gz"
        )
        try:
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except IOError as e:
            logging.error(f"Error writing tarball to file: {e}")
            return e

        final_dir = os.path.join(
            target_dir, f"{repo.full_name.replace('/', '-')}-{branch}-{user_id}"
        )
        try:
            with tarfile.open(tarball_path, "r:gz") as tar:
                for member in tar.getmembers():
                    member_path = os.path.join(
                        final_dir,
                        os.path.relpath(member.name, start=member.name.split("/")[0]),
                    )
                    if member.isdir():
                        os.makedirs(member_path, exist_ok=True)
                    else:
                        member_dir = os.path.dirname(member_path)
                        if not os.path.exists(member_dir):
                            os.makedirs(member_dir)
                        with open(member_path, "wb") as f:
                            if member.size > 0:
                                f.write(tar.extractfile(member).read())
        except (tarfile.TarError, IOError) as e:
            logging.error(f"Error extracting tarball: {e}")
            return e

        try:
            os.remove(tarball_path)
        except OSError as e:
            logging.error(f"Error removing tarball: {e}")
            return e

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
            "other": 0,
        }
        total_chars = 0

        try:
            for root, _, files in os.walk(repo_dir):
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
                            else:
                                lang_count["other"] += 1
                    except (
                        UnicodeDecodeError,
                        FileNotFoundError,
                        PermissionError,
                    ) as e:
                        logging.warning(f"Error reading file {file_path}: {e}")
                        continue
        except (TypeError, FileNotFoundError, PermissionError) as e:
            logging.error(f"Error accessing directory '{repo_dir}': {e}")

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
        if not project_id:
            pid = str(uuid7())
            project_id = await self.project_manager.register_project(
                f"{repo.full_name}",
                branch,
                user_id,
                pid,
            )

        if isinstance(repo_details, Repo):
            extracted_dir = repo_details.working_tree_dir
            try:
                current_dir = os.getcwd()
                os.chdir(extracted_dir)  # Change to the cloned repo directory
                repo_details.git.checkout(branch)
            except GitCommandError as e:
                logging.error(f"Error checking out branch: {e}")
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

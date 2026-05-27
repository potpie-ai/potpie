import json
import os
import urllib.request
from typing import Optional

from sqlalchemy.orm import Session

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _fetch_github_branch_head_sha_http(
    repo_name: str, branch_name: str
) -> Optional[str]:
    """
    Fetch the HEAD commit SHA for a GitHub branch using only HTTP (no GitPython/PyGithub).
    Safe to call from forked processes (gunicorn workers) where GitPython causes SIGSEGV.
    """
    try:
        url = f"https://api.github.com/repos/{repo_name}/branches/{branch_name}"
        token_list = os.getenv("GH_TOKEN_LIST", "").strip()
        token = os.getenv("CODE_PROVIDER_TOKEN")
        if token_list:
            parts = [
                p.strip() for p in token_list.replace("\n", ",").split(",") if p.strip()
            ]
            if parts:
                token = token or parts[0]
        if not token:
            token = os.getenv("CODE_PROVIDER_TOKEN")
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return (data.get("commit") or {}).get("sha")
    except Exception:
        return None


class ParsingServiceError(Exception):
    """Base exception class for ParsingService errors."""


class ParsingFailedError(ParsingServiceError):
    """Raised when a parsing fails."""


class ParseHelper:
    def __init__(self, db_session: Session):
        self.project_manager = ProjectService(db_session)
        self.db = db_session
        self.github_service = CodeProviderService(db_session)
        # Post-Phase-5: parsing goes through ProjectSandbox (in-sandbox
        # tree-sitter). RepoManager and the host-FS clone helpers
        # (clone_or_copy_repository, setup_project_directory,
        # detect_repo_language) are gone. ParseHelper is retained for
        # the surviving utilities — `is_text_file` and
        # `check_commit_status` — used by parsing_repomap and the
        # parsing controller.

    @staticmethod
    def get_directory_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
            # # Skip symlinked directories
            # dirnames[:] = [
            #     d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
            # ]

            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Skip all symlinks
                if os.path.islink(fp):
                    continue
                total_size += os.path.getsize(fp)
        return total_size

    def is_text_file(self, file_path):
        def open_text_file(file_path):
            """
            Try multiple encodings to detect if file is text.

            Order of encodings to try:
            1. utf-8 (most common)
            2. utf-8-sig (UTF-8 with BOM)
            3. utf-16 (common in Windows C# files)
            4. latin-1/iso-8859-1 (fallback, accepts all byte sequences)
            """
            encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        # Read first 8KB to detect encoding
                        f.read(8192)
                    return True
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception:
                    # Handle other errors (permissions, file not found, etc.)
                    return False

            # If all encodings fail, likely a binary file
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
            "h",
            "hpp",
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
            "ps1",
            "psm1",
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

    async def check_commit_status(
        self, project_id: str, requested_commit_id: str = None
    ) -> bool:
        """
        Check if the current commit ID of the project matches the latest commit ID from the repository.

        Args:
            project_id (str): The ID of the project to check.
            requested_commit_id (str, optional): The commit ID from the current parse request.
                If provided, indicates this is a pinned commit parse (not branch-based).
        Returns:
            bool: True if the commit IDs match or if this is a pinned commit parse, False otherwise.
        """
        logger.info(
            f"check_commit_status: Checking commit status for project {project_id}, "
            f"requested_commit_id={requested_commit_id}"
        )

        project = await self.project_manager.get_project_from_db_by_id(project_id)
        if not project:
            logger.error(f"Project with ID {project_id} not found")
            return False

        current_commit_id = project.get("commit_id")
        repo_name = project.get("project_name")
        branch_name = project.get("branch_name")

        logger.info(
            f"check_commit_status: Project {project_id} - repo={repo_name}, "
            f"branch={branch_name}, current_commit_id={current_commit_id}"
        )

        # Check if this is a pinned commit parse
        # If the user explicitly provided a commit_id in the parse request,
        # this is a pinned commit parse (not branch-based)
        if requested_commit_id is not None:
            logger.info(
                f"check_commit_status: Pinned commit parse detected "
                f"(requested_commit_id={requested_commit_id})"
            )
            # For pinned commits, check if the requested commit matches the stored commit
            if requested_commit_id == current_commit_id:
                logger.info(
                    f"check_commit_status: Pinned commit {requested_commit_id} matches "
                    f"stored commit, no reparse needed"
                )
                return True
            else:
                logger.info(
                    f"check_commit_status: Pinned commit changed from {current_commit_id} "
                    f"to {requested_commit_id}, reparse needed"
                )
                return False

        # If we reach here, this is a branch-based parse (not pinned commit)
        # We need to compare the stored commit with the latest branch commit

        if not repo_name:
            logger.error(
                f"Repository name or branch name not found for project ID {project_id}"
            )
            return False

        if not branch_name:
            logger.info(
                f"check_commit_status: Branch is empty (pinned commit parse) - "
                f"sticking to commit and not updating it for: {project_id}"
            )
            return True

        if len(repo_name.split("/")) < 2:
            # Local repo, always parse local repos
            logger.info("check_commit_status: Local repo detected, forcing reparse")
            return False

        try:
            # If current_commit_id is None, we should reparse
            if current_commit_id is None:
                logger.info(
                    f"check_commit_status: Project {project_id} has no commit_id, will reparse"
                )
                return False

            # Use HTTP-only GitHub API to avoid GitPython/libgit2 in forked gunicorn workers (SIGSEGV)
            logger.info(
                f"check_commit_status: Branch-based parse - getting repo info for {repo_name}"
            )
            latest_commit_id = await asyncio.to_thread(
                _fetch_github_branch_head_sha_http, repo_name, branch_name
            )

            # Compare current commit with latest commit
            is_up_to_date = current_commit_id == latest_commit_id
            logger.info(
                f"check_commit_status: Project {project_id} commit status for branch {branch_name}: "
                f"{'Up to date' if is_up_to_date else 'Outdated'} - "
                f"Current: {current_commit_id}, Latest: {latest_commit_id}"
            )

            return is_up_to_date
        except Exception:
            logger.exception(
                "check_commit_status: Error fetching latest commit",
                repo_name=repo_name,
                branch_name=branch_name,
                project_id=project_id,
            )
            return False

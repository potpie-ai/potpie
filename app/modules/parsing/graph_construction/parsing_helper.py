import math
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser  # noqa: E402
import json
import logging
import os
import shutil
import tarfile
import requests
from fastapi import HTTPException
from git import Repo, GitCommandError
from app.modules.projects.projects_service import ProjectService
import networkx as nx
from sqlalchemy.orm import Session  # Import SQLAlchemy Session
# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
Tag = namedtuple("Tag", "rel_fname fname line end_line name kind type".split())

class ParseHelper:
    def __init__(self, db_session: Session):
        self.project_manager = ProjectService(db_session)  # Initialize ProjectService with db session

    def download_and_extract_tarball(self, repo, branch, target_dir, auth, repo_details, user_id):
        try:
            tarball_url = repo_details.get_archive_link("tarball", branch)
            response = requests.get(
                tarball_url,
                stream=True,
                headers={"Authorization": f"{auth.token}"},
            )
            response.raise_for_status()  # Check for request errors
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching tarball: {e}")
            return e

        tarball_path = os.path.join(target_dir, f"{repo.name}-{branch}.tar.gz")
        try:
            with open(tarball_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except IOError as e:
            logging.error(f"Error writing tarball to file: {e}")
            return e

        final_dir = os.path.join(target_dir, f"{repo}-{branch}-{user_id}")
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


    def setup_project_directory(
        self, owner, repo, branch, auth, repo_details, user_id, project_id=None
    ):
        should_parse_repo = True
        default = False

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
            if branch == repo_details.default_branch:
                default = True
            extracted_dir = self.download_and_extract_tarball(
                repo, branch, os.getenv("PROJECT_PATH"), auth, repo_details, user_id
            )
            branch_details = repo_details.get_branch(branch)
            latest_commit_sha = branch_details.commit.sha

        momentum_dir = os.path.join(extracted_dir, ".momentum")
        os.makedirs(momentum_dir, exist_ok=True)
        with open(os.path.join(momentum_dir, "momentum.db"), "w") as fp:
            pass

        repo_metadata = extract_repository_metadata(repo_details)
        repo_metadata["error_message"] = None

        if os.getenv("isDevelopmentMode") == "disabled":
            python_percentage = (
                (
                    repo_metadata["languages"]["breakdown"]["Python"]
                    / repo_metadata["languages"]["total_bytes"]
                    * 100
                )
                if "Python" in repo_metadata["languages"]["breakdown"]
                else 0
            )
            if python_percentage < 50:
                repo_metadata["error_message"] = (
                    "Repository doesn't consist of a language currently supported."
                )
                should_parse_repo = False
            else:
                repo_metadata["error_message"] = None

        project_id = self.project_manager.register_project(
            f"{repo.full_name}",
            branch,
            user_id,
            latest_commit_sha,
            json.dumps(repo_metadata).encode("utf-8"),
            project_id,
        )
        return extracted_dir, project_id, should_parse_repo



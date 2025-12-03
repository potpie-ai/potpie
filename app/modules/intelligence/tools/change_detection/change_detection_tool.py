import asyncio
import os
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
from typing import Dict, List, Optional

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tree_sitter_languages import get_parser
import pathspec

from app.core.database import get_db
from app.core.config_provider import config_provider
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)
from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService
from neo4j import GraphDatabase


class ChangeDetectionInput(BaseModel):
    project_id: str = Field(
        ..., description="The ID of the project being evaluated, this is a UUID."
    )


class ChangeDetail(BaseModel):
    updated_code: str = Field(..., description="The updated code for the node")
    entrypoint_code: str = Field(..., description="The code for the entry point")
    citations: List[str] = Field(
        ..., description="List of file names referenced in the response"
    )


class ChangeDetectionResponse(BaseModel):
    patches: Dict[str, str] = Field(..., description="Dictionary of file patches")
    changes: List[ChangeDetail] = Field(
        ..., description="List of changes with updated and entry point code"
    )


class ChangeDetectionTool:
    name = "Get code changes"
    description = """Analyzes differences between branches in a Git repository and retrieves updated function details.
        :param project_id: string, the ID of the project being evaluated (UUID).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000"
            }

        Returns dictionary containing:
        - patches: Dict[str, str] - file patches
        - changes: List[ChangeDetail] - list of changes with updated and entry point code
        """

    def __init__(self, sql_db, user_id):
        self.sql_db = sql_db
        self.user_id = user_id
        self.search_service = SearchService(self.sql_db)
        # Initialize Neo4j driver for direct queries
        neo4j_config = config_provider.get_neo4j_config()
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["username"], neo4j_config["password"]),
        )

    def _find_node_by_file_and_name(
        self, project_id: str, file_path: str, function_name: str
    ) -> Optional[str]:
        """
        Find a node_id in Neo4j by file path and function name.

        Args:
            project_id: The project/repo ID
            file_path: The file path (e.g., 'tests/test_sdk.py')
            function_name: The function name (e.g., 'test_document_lifecycle')

        Returns:
            The node_id if found, None otherwise
        """
        # Try multiple query strategies:
        # 1. Exact match on file_path ending (handles full paths)
        # 2. Contains match on file_path (handles both relative and full paths)
        # 3. Match on just the filename
        # 4. Try with full absolute path (if we can construct it)
        queries = [
            # Strategy 1: File path ends with the relative path
            """
            MATCH (n:NODE {repoId: $project_id})
            WHERE n.file_path ENDS WITH $file_path
            AND n.name = $function_name
            AND (n:FUNCTION OR n:CLASS)
            RETURN n.node_id AS node_id, n.file_path AS file_path, n.name AS name
            LIMIT 5
            """,
            # Strategy 2: File path contains the relative path
            """
            MATCH (n:NODE {repoId: $project_id})
            WHERE n.file_path CONTAINS $file_path
            AND n.name = $function_name
            AND (n:FUNCTION OR n:CLASS)
            RETURN n.node_id AS node_id, n.file_path AS file_path, n.name AS name
            LIMIT 5
            """,
            # Strategy 3: Match on just the filename
            """
            MATCH (n:NODE {repoId: $project_id})
            WHERE n.file_path ENDS WITH $filename
            AND n.name = $function_name
            AND (n:FUNCTION OR n:CLASS)
            RETURN n.node_id AS node_id, n.file_path AS file_path, n.name AS name
            LIMIT 5
            """,
            # Strategy 4: Try with dot-separated path format (e.g., .Users.dhirenmathur.Documents.mongo-proxy.tests.test_sdk.py)
            """
            MATCH (n:NODE {repoId: $project_id})
            WHERE n.file_path ENDS WITH $dot_path
            AND n.name = $function_name
            AND (n:FUNCTION OR n:CLASS)
            RETURN n.node_id AS node_id, n.file_path AS file_path, n.name AS name
            LIMIT 5
            """,
        ]

        # Extract just the filename for strategy 3
        filename = file_path.split("/")[-1] if "/" in file_path else file_path

        # Create dot-separated path format for strategy 4
        # Convert 'tests/test_sdk.py' to '.tests.test_sdk.py'
        dot_path = "." + file_path.replace("/", ".")

        try:
            with self.neo4j_driver.session() as session:
                for i, query in enumerate(queries, 1):
                    if i == 3:
                        # Use filename for strategy 3
                        result = session.run(
                            query,
                            project_id=project_id,
                            filename=filename,
                            function_name=function_name,
                        )
                    elif i == 4:
                        # Use dot_path for strategy 4
                        result = session.run(
                            query,
                            project_id=project_id,
                            dot_path=dot_path,
                            function_name=function_name,
                        )
                    else:
                        result = session.run(
                            query,
                            project_id=project_id,
                            file_path=file_path,
                            function_name=function_name,
                        )

                    records = list(result)
                    if records:
                        # Return the first match
                        return records[0]["node_id"]

                # No matches found with any strategy
                return None
        except Exception as e:
            logger.error(
                f"[CHANGE_DETECTION] Error searching for node by file and name: {e}",
                exc_info=True,
            )
            return None

    def _get_gitignore_spec(self, repo_path: str) -> Optional[pathspec.PathSpec]:
        """
        Create a PathSpec object from the .gitignore file in the repository.

        Args:
            repo_path: Path to the repository root

        Returns:
            PathSpec object or None if .gitignore doesn't exist
        """
        gitignore_path = os.path.join(repo_path, ".gitignore")
        if not os.path.exists(gitignore_path):
            return None

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                gitignore_content = f.read()

            # Create a PathSpec object from the .gitignore content
            return pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, gitignore_content.splitlines()
            )
        except Exception as e:
            logger.warning(
                f"[CHANGE_DETECTION] Error reading .gitignore file: {str(e)}"
            )
            return None

    def _parse_diff_detail(self, patch_details):
        changed_files = {}
        current_file = None
        for filename, patch in patch_details.items():
            lines = patch.split("\n")
            current_file = filename
            changed_files[current_file] = set()
            for line in lines:
                if line.startswith("@@"):
                    parts = line.split()
                    add_start_line, add_num_lines = (
                        map(int, parts[2][1:].split(","))
                        if "," in parts[2]
                        else (int(parts[2][1:]), 1)
                    )
                    for i in range(add_start_line, add_start_line + add_num_lines):
                        changed_files[current_file].add(i)
        return changed_files

    async def _find_changed_functions(self, changed_files, project_id):
        result = []
        for relative_file_path, lines in changed_files.items():
            try:
                project = await ProjectService(self.sql_db).get_project_from_db_by_id(
                    project_id
                )
                code_service = CodeProviderService(self.sql_db)
                # Use repo_path for local repos, otherwise use project_name
                # Only use project_name if repo_path is not set AND it's a valid remote repo format
                repo_path = project.get("repo_path")
                if repo_path:
                    repo_identifier = repo_path
                else:
                    # Check if project_name is a local path before using it
                    project_name = project["project_name"]
                    if os.path.isabs(project_name) or os.path.isdir(
                        os.path.expanduser(project_name)
                    ):
                        repo_identifier = project_name
                    else:
                        # Not a local path, use project_name (will be validated by provider)
                        repo_identifier = project_name

                file_content = code_service.get_file_content(
                    repo_identifier,
                    relative_file_path,
                    0,
                    0,
                    project["branch_name"],
                    project_id,
                    project["commit_id"],
                )
                tags = RepoMap.get_tags_from_code(relative_file_path, file_content)

                language = RepoMap.get_language_for_file(relative_file_path)
                if language:
                    parser = get_parser(language.name)
                    tree = parser.parse(bytes(file_content, "utf8"))
                    root_node = tree.root_node

                nodes = {}
                for tag in tags:
                    if tag.kind == "def":
                        if tag.type == "class":
                            node_type = "CLASS"
                        elif tag.type in ["method", "function"]:
                            node_type = "FUNCTION"

                        else:
                            node_type = "other"

                        node_name = f"{relative_file_path}:{tag.name}"

                        if language:
                            node = RepoMap.find_node_by_range(
                                root_node, tag.line, node_type
                            )
                        if node:
                            nodes[node_name] = node

                for node_name, node in nodes.items():
                    start_line = node.start_point[0]
                    end_line = node.end_point[0]
                    if any(start_line < line < end_line for line in lines):
                        result.append(node_name)
            except Exception as e:
                logger.error(f"Exception {e}")
        return result

    async def get_updated_function_list(self, patch_details, project_id):
        changed_files = self._parse_diff_detail(patch_details)
        return await self._find_changed_functions(changed_files, project_id)

    @staticmethod
    def _find_inbound_neighbors(tx, node_id, project_id, with_bodies):
        query = f"""
        MATCH (start:Function {{id: $endpoint_id, project_id: $project_id}})
        CALL {{
            WITH start
            MATCH (neighbor:Function {{project_id: $project_id}})-[:CALLS*]->(start)
            RETURN neighbor{', neighbor.body AS body' if with_bodies else ''}
        }}
        RETURN start, collect({{neighbor: neighbor{', body: neighbor.body' if with_bodies else ''}}}) AS neighbors
        """
        endpoint_id = node_id
        result = tx.run(query, endpoint_id, project_id)
        record = result.single()
        if not record:
            return []

        start_node = dict(record["start"])
        neighbors = record["neighbors"]
        combined = [start_node] + neighbors if neighbors else [start_node]
        return combined

    def traverse(self, identifier, project_id, neighbors_fn):
        neighbors_query = neighbors_fn(with_bodies=False)
        with self.driver.session() as session:
            return session.read_transaction(
                self._traverse, identifier, project_id, neighbors_query
            )

    def find_entry_points(self, identifiers, project_id):
        all_inbound_nodes = set()

        for identifier in identifiers:
            traversal_result = self.traverse(
                identifier=identifier,
                project_id=project_id,
                neighbors_fn=ChangeDetectionTool._find_inbound_neighbors,
            )
            for item in traversal_result:
                if isinstance(item, dict):
                    all_inbound_nodes.update([frozenset(item.items())])

        entry_points = set()
        for node in all_inbound_nodes:
            node_dict = dict(node)
            traversal_result = self.traverse(
                identifier=node_dict["id"],
                project_id=project_id,
                neighbors_fn=ChangeDetectionTool._find_inbound_neighbors,
            )
            if len(traversal_result) == 1:
                entry_points.add(node)

        return entry_points

    async def get_code_changes(self, project_id):
        logger.info(
            f"[CHANGE_DETECTION] Starting get_code_changes for project_id: {project_id}"
        )
        patches_dict = {}
        project_details = await ProjectService(self.sql_db).get_project_from_db_by_id(
            project_id
        )
        logger.info(f"[CHANGE_DETECTION] Retrieved project details: {project_details}")

        if project_details is None:
            logger.error(
                f"[CHANGE_DETECTION] Project details not found for project_id: {project_id}"
            )
            raise HTTPException(status_code=400, detail="Project Details not found.")

        if project_details["user_id"] != self.user_id:
            logger.error(
                f"[CHANGE_DETECTION] User mismatch: project user_id={project_details['user_id']}, requesting user={self.user_id}"
            )
            raise ValueError(
                f"Project id {project_id} not found for user {self.user_id}"
            )

        repo_name = project_details["project_name"]
        branch_name = project_details["branch_name"]
        repo_path = project_details["repo_path"]
        logger.info(
            f"[CHANGE_DETECTION] Project info - repo: {repo_name}, branch: {branch_name}, path: {repo_path}"
        )

        # Use CodeProviderService to get the appropriate service instance
        code_service = CodeProviderService(self.sql_db)
        logger.info(
            f"[CHANGE_DETECTION] CodeProviderService created, service_instance type: {type(code_service.service_instance).__name__}"
        )

        # Import ProviderWrapper to check instance type
        from app.modules.code_provider.code_provider_service import ProviderWrapper

        try:
            # Handle ProviderWrapper (new provider factory pattern)
            if isinstance(code_service.service_instance, ProviderWrapper):
                logger.info("[CHANGE_DETECTION] Using ProviderWrapper for diff")

                # Get the actual repo name for API calls (handles GitBucket conversion)
                from app.modules.parsing.utils.repo_name_normalizer import (
                    get_actual_repo_name_for_lookup,
                )
                from app.modules.code_provider.provider_factory import (
                    CodeProviderFactory,
                )
                import os

                provider_type = os.getenv("CODE_PROVIDER", "github").lower()

                # Use repo_path for local repos, otherwise normalize the repo_name
                if repo_path:
                    actual_repo_name = repo_path
                    logger.info(
                        f"[CHANGE_DETECTION] Using local repo_path: {actual_repo_name}"
                    )
                else:
                    actual_repo_name = get_actual_repo_name_for_lookup(
                        repo_name, provider_type
                    )
                    logger.info(
                        f"[CHANGE_DETECTION] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
                    )

                # For local repos, skip provider creation and use git diff directly
                if repo_path and os.path.isdir(repo_path):
                    # Local repository - use git diff for accurate change detection
                    from git import Repo as GitRepo

                    git_repo = GitRepo(actual_repo_name)

                    # Get default branch (usually main or master)
                    try:
                        # Try to get the default branch from git config
                        default_branch = git_repo.git.symbolic_ref(
                            "refs/remotes/origin/HEAD"
                        ).split("/")[-1]
                    except Exception:
                        # Fallback to common default branch names
                        if "main" in git_repo.heads:
                            default_branch = "main"
                        elif "master" in git_repo.heads:
                            default_branch = "master"
                        else:
                            default_branch = git_repo.active_branch.name

                    current_branch = branch_name or git_repo.active_branch.name
                    logger.info(
                        f"[CHANGE_DETECTION] Local repo - comparing {default_branch}..{current_branch}"
                    )

                    # Get all changes from default branch (includes committed + uncommitted)
                    # This is equivalent to: git diff <default_branch>
                    patches_dict = {}
                    try:
                        # Load .gitignore patterns to filter out ignored files
                        gitignore_spec = self._get_gitignore_spec(actual_repo_name)

                        # Use git diff <default_branch> to get all changes from that branch
                        # This includes both committed changes on current branch AND uncommitted changes
                        diff_output = git_repo.git.diff(default_branch, unified=3)

                        # Parse diff output to get file-level patches
                        if diff_output:
                            current_file = None
                            current_patch = []
                            for line in diff_output.splitlines():
                                if line.startswith("diff --git"):
                                    # Save previous file
                                    if current_file:
                                        # Only add file if it doesn't match .gitignore patterns
                                        if (
                                            not gitignore_spec
                                            or not gitignore_spec.match_file(
                                                current_file
                                            )
                                        ):
                                            patches_dict[current_file] = "\n".join(
                                                current_patch
                                            )
                                        else:
                                            logger.debug(
                                                f"[CHANGE_DETECTION] Excluding ignored file: {current_file}"
                                            )
                                    # Extract filename
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        current_file = parts[2].lstrip("a/")
                                        current_patch = [line]
                                elif current_file:
                                    current_patch.append(line)
                            # Save last file
                            if current_file:
                                # Only add file if it doesn't match .gitignore patterns
                                if not gitignore_spec or not gitignore_spec.match_file(
                                    current_file
                                ):
                                    patches_dict[current_file] = "\n".join(
                                        current_patch
                                    )
                                else:
                                    logger.debug(
                                        f"[CHANGE_DETECTION] Excluding ignored file: {current_file}"
                                    )

                        logger.info(
                            f"[CHANGE_DETECTION] Local repo - found {len(patches_dict)} changed files (diff from {default_branch})"
                        )
                    except Exception as e:
                        logger.error(
                            f"[CHANGE_DETECTION] Error getting local changes: {e}"
                        )
                        patches_dict = {}
                else:
                    # Remote repository - create provider with proper auth
                    provider = CodeProviderFactory.create_provider_with_fallback(
                        actual_repo_name
                    )

                    github_client = provider.client
                    repo = github_client.get_repo(actual_repo_name)
                    default_branch = repo.default_branch
                    logger.info(
                        f"[CHANGE_DETECTION] Remote repo - default branch: {default_branch}, comparing with: {branch_name}"
                    )

                    # Use provider's compare_branches method
                    logger.info(
                        "[CHANGE_DETECTION] Using provider's compare_branches method"
                    )
                    comparison_result = provider.compare_branches(
                        actual_repo_name, default_branch, branch_name
                    )

                    # Extract patches from comparison result
                    patches_dict = {
                        file["filename"]: file["patch"]
                        for file in comparison_result["files"]
                        if "patch" in file
                    }
                    logger.info(
                        f"[CHANGE_DETECTION] Comparison complete: {len(patches_dict)} files with patches, {comparison_result['commits']} commits"
                    )

            elif isinstance(code_service.service_instance, GithubService):
                logger.info("[CHANGE_DETECTION] Using GithubService for diff")
                github, _, _ = code_service.service_instance.get_github_repo_details(
                    repo_name
                )
                logger.info("[CHANGE_DETECTION] Got github client from service")

                # Get the actual repo name for API calls (handles GitBucket conversion)
                from app.modules.parsing.utils.repo_name_normalizer import (
                    get_actual_repo_name_for_lookup,
                )
                import os

                provider_type = os.getenv("CODE_PROVIDER", "github").lower()
                actual_repo_name = get_actual_repo_name_for_lookup(
                    repo_name, provider_type
                )
                logger.info(
                    f"[CHANGE_DETECTION] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
                )

                repo = github.get_repo(actual_repo_name)
                logger.info(f"[CHANGE_DETECTION] Got repo object: {repo.name}")
                default_branch = repo.default_branch
                logger.info(
                    f"[CHANGE_DETECTION] Default branch: {default_branch}, comparing with: {branch_name}"
                )

                # GitBucket workaround: Use commits API to get diff
                if provider_type == "gitbucket":
                    logging.info(
                        "[CHANGE_DETECTION] Using commits API for GitBucket diff"
                    )

                    try:
                        # Get commits on the branch
                        logger.info(
                            f"[CHANGE_DETECTION] Getting commits for branch: {branch_name}"
                        )
                        commits = repo.get_commits(sha=branch_name)

                        patches_dict = {}
                        commit_count = 0

                        # Get all commits until we reach the default branch
                        for commit in commits:
                            commit_count += 1
                            # Check if this commit is on the default branch
                            try:
                                default_commits = list(
                                    repo.get_commits(sha=default_branch)
                                )
                                default_commit_shas = [c.sha for c in default_commits]

                                if commit.sha in default_commit_shas:
                                    logger.info(
                                        f"[CHANGE_DETECTION] Reached common ancestor at commit {commit.sha[:7]}"
                                    )
                                    break
                            except:
                                pass

                            # Get the commit details with files
                            logger.info(
                                f"[CHANGE_DETECTION] Processing commit {commit.sha[:7]}: {commit.commit.message.split(chr(10))[0]}"
                            )

                            for file in commit.files:
                                if file.patch and file.filename not in patches_dict:
                                    patches_dict[file.filename] = file.patch
                                    logger.info(
                                        f"[CHANGE_DETECTION] Added patch for file: {file.filename}"
                                    )

                            # Limit to reasonable number of commits
                            if commit_count >= 50:
                                logger.warning(
                                    "[CHANGE_DETECTION] Reached commit limit of 50, stopping"
                                )
                                break

                        logger.info(
                            f"[CHANGE_DETECTION] GitBucket diff complete: {len(patches_dict)} files with patches from {commit_count} commits"
                        )
                    except Exception as api_error:
                        logger.error(
                            f"[CHANGE_DETECTION] GitBucket commits API error: {type(api_error).__name__}: {str(api_error)}",
                            exc_info=True,
                        )
                        raise
                else:
                    # Use PyGithub for GitHub
                    git_diff = repo.compare(default_branch, branch_name)
                    logger.info(
                        f"[CHANGE_DETECTION] Comparison complete, files changed: {len(git_diff.files)}"
                    )
                    patches_dict = {
                        file.filename: file.patch
                        for file in git_diff.files
                        if file.patch
                    }
                    logger.info(
                        f"[CHANGE_DETECTION] Patches extracted: {len(patches_dict)} files with patches"
                    )
            elif isinstance(code_service.service_instance, LocalRepoService):
                logger.info("[CHANGE_DETECTION] Using LocalRepoService for diff")
                patches_dict = code_service.service_instance.get_local_repo_diff(
                    repo_path, branch_name
                )
                logger.info(
                    f"[CHANGE_DETECTION] Local diff complete: {len(patches_dict)} files"
                )
        except Exception as e:
            logger.error(
                f"[CHANGE_DETECTION] Exception during diff: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=400, detail=f"Error while fetching changes: {str(e)}"
            )
        finally:
            if project_details is not None:
                logger.info(
                    f"[CHANGE_DETECTION] Processing patches: {len(patches_dict)} files"
                )
                identifiers = []
                node_ids = []
                try:
                    identifiers = await self.get_updated_function_list(
                        patches_dict, project_id
                    )
                    logger.info(
                        f"[CHANGE_DETECTION] Found {len(identifiers)} changed functions: {identifiers}"
                    )
                    for identifier in identifiers:
                        node_id_query = " ".join(identifier.split(":"))
                        relevance_search = await self.search_service.search_codebase(
                            project_id, node_id_query
                        )
                        if relevance_search:
                            node_id = relevance_search[0].get("node_id")
                            if node_id:
                                node_ids.append(node_id)
                        else:
                            # Fallback: try to find node by file path and function name
                            # Identifier format is 'file_path:function_name'
                            # Parse identifier into file_path and function_name
                            if ":" in identifier:
                                file_path, function_name = identifier.rsplit(":", 1)

                                # Try to find node by file path and name
                                found_node_id = self._find_node_by_file_and_name(
                                    project_id, file_path, function_name
                                )

                                if found_node_id:
                                    node_ids.append(found_node_id)
                                else:
                                    # Last resort: try GetCodeFromNodeIdTool with identifier as-is
                                    # (in case the identifier IS the actual node_id format)
                                    fallback_result = GetCodeFromNodeIdTool(
                                        self.sql_db, self.user_id
                                    ).run(project_id, identifier)

                                    # Check if result has an error or missing node_id
                                    if "error" in fallback_result:
                                        logger.warning(
                                            f"[CHANGE_DETECTION] Could not find node for identifier '{identifier}': {fallback_result['error']}"
                                        )
                                        continue

                                    node_id = fallback_result.get("node_id")
                                    if node_id:
                                        node_ids.append(node_id)

                    # Fetch code for node ids and store in a dict
                    node_code_dict = {}
                    for node_id in node_ids:
                        node_code = GetCodeFromNodeIdTool(
                            self.sql_db, self.user_id
                        ).run(project_id, node_id)

                        # Check for errors in the response
                        if "error" in node_code:
                            logger.warning(
                                f"[CHANGE_DETECTION] Error getting code for node {node_id}: {node_code['error']}"
                            )
                            continue

                        # Check for required fields
                        if (
                            "code_content" not in node_code
                            or "file_path" not in node_code
                        ):
                            logger.warning(
                                f"[CHANGE_DETECTION] Missing required fields for node {node_id}"
                            )
                            continue

                        node_code_dict[node_id] = {
                            "code_content": node_code["code_content"],
                            "file_path": node_code["file_path"],
                        }

                    entry_points = InferenceService(
                        self.sql_db, "dummy"
                    ).get_entry_points_for_nodes(node_ids, project_id)

                    changes_list = []
                    for node, entry_point in entry_points.items():
                        # Skip if node is not in node_code_dict (was filtered out due to errors)
                        if node not in node_code_dict:
                            logger.warning(
                                f"[CHANGE_DETECTION] Skipping node {node} - not in node_code_dict"
                            )
                            continue

                        entry_point_code = GetCodeFromNodeIdTool(
                            self.sql_db, self.user_id
                        ).run(project_id, entry_point[0])

                        # Check for errors in entry_point_code
                        if "error" in entry_point_code:
                            logger.warning(
                                f"[CHANGE_DETECTION] Error getting entry point code for {entry_point[0]}: {entry_point_code['error']}"
                            )
                            continue

                        # Check for required fields in entry_point_code
                        if (
                            "code_content" not in entry_point_code
                            or "file_path" not in entry_point_code
                        ):
                            logger.warning(
                                f"[CHANGE_DETECTION] Missing required fields in entry point code: {entry_point_code}"
                            )
                            continue

                        changes_list.append(
                            ChangeDetail(
                                updated_code=node_code_dict[node]["code_content"],
                                entrypoint_code=entry_point_code["code_content"],
                                citations=[
                                    node_code_dict[node]["file_path"],
                                    entry_point_code["file_path"],
                                ],
                            )
                        )

                    # For local repos with too many changes, return only file names instead of full patches
                    # to avoid context length issues
                    if len(patches_dict) > 50:
                        logger.warning(
                            f"[CHANGE_DETECTION] Too many patches ({len(patches_dict)}), returning file names only"
                        )
                        # Convert patches to just file names with line counts
                        patches_summary = {
                            filename: f"Changed ({len(patch.splitlines())} lines)"
                            for filename, patch in patches_dict.items()
                        }
                        result = ChangeDetectionResponse(
                            patches=patches_summary, changes=changes_list
                        )
                    else:
                        result = ChangeDetectionResponse(
                            patches=patches_dict, changes=changes_list
                        )
                    logger.info(
                        f"[CHANGE_DETECTION] Returning result with {len(patches_dict)} patches and {len(changes_list)} changes"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"[CHANGE_DETECTION] Exception in finally block - project_id: {project_id}, error: {type(e).__name__}: {str(e)}",
                        exc_info=True,
                    )

                if len(identifiers) == 0:
                    logger.info(
                        "[CHANGE_DETECTION] No identifiers found, returning empty list"
                    )
                    return []

    async def arun(self, project_id: str) -> str:
        return await self.get_code_changes(project_id)

    def run(self, project_id: str) -> str:
        return asyncio.run(self.get_code_changes(project_id))


def get_change_detection_tool(user_id: str) -> StructuredTool:
    """
    Get a list of LangChain Tool objects for use in agents.
    """
    change_detection_tool = ChangeDetectionTool(next(get_db()), user_id)
    return StructuredTool.from_function(
        coroutine=change_detection_tool.arun,
        func=change_detection_tool.run,
        name="Get code changes",
        description="""
            Get the changes in the codebase.
            This tool analyzes the differences between branches in a Git repository and retrieves updated function details, including their entry points and citations.
            Inputs for the get_code_changes method:
            - project_id (str): The ID of the project being evaluated, this is a UUID.
            The output includes a dictionary of file patches and a list of changes with updated code and entry point code.
            """,
        args_schema=ChangeDetectionInput,
    )

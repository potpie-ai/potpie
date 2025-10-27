import asyncio
import logging
from typing import Dict, List

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tree_sitter_languages import get_parser

from app.core.database import get_db
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
                file_content = code_service.get_file_content(
                    project["project_name"],
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
                logging.error(f"Exception {e}")
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
        logging.info(
            f"[CHANGE_DETECTION] Starting get_code_changes for project_id: {project_id}"
        )
        global patches_dict, repo
        patches_dict = {}
        project_details = await ProjectService(self.sql_db).get_project_from_db_by_id(
            project_id
        )
        logging.info(f"[CHANGE_DETECTION] Retrieved project details: {project_details}")

        if project_details is None:
            logging.error(
                f"[CHANGE_DETECTION] Project details not found for project_id: {project_id}"
            )
            raise HTTPException(status_code=400, detail="Project Details not found.")

        if project_details["user_id"] != self.user_id:
            logging.error(
                f"[CHANGE_DETECTION] User mismatch: project user_id={project_details['user_id']}, requesting user={self.user_id}"
            )
            raise ValueError(
                f"Project id {project_id} not found for user {self.user_id}"
            )

        repo_name = project_details["project_name"]
        branch_name = project_details["branch_name"]
        repo_path = project_details["repo_path"]
        logging.info(
            f"[CHANGE_DETECTION] Project info - repo: {repo_name}, branch: {branch_name}, path: {repo_path}"
        )

        # Use CodeProviderService to get the appropriate service instance
        code_service = CodeProviderService(self.sql_db)
        logging.info(
            f"[CHANGE_DETECTION] CodeProviderService created, service_instance type: {type(code_service.service_instance).__name__}"
        )

        # Import ProviderWrapper to check instance type
        from app.modules.code_provider.code_provider_service import ProviderWrapper

        try:
            # Handle ProviderWrapper (new provider factory pattern)
            if isinstance(code_service.service_instance, ProviderWrapper):
                logging.info("[CHANGE_DETECTION] Using ProviderWrapper for diff")

                # Get the actual repo name for API calls (handles GitBucket conversion)
                from app.modules.parsing.utils.repo_name_normalizer import (
                    get_actual_repo_name_for_lookup,
                )
                import os

                provider_type = os.getenv("CODE_PROVIDER", "github").lower()
                actual_repo_name = get_actual_repo_name_for_lookup(
                    repo_name, provider_type
                )
                logging.info(
                    f"[CHANGE_DETECTION] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
                )

                # Get default branch first
                github_client = code_service.service_instance.provider.client
                repo = github_client.get_repo(actual_repo_name)
                default_branch = repo.default_branch
                logging.info(
                    f"[CHANGE_DETECTION] Default branch: {default_branch}, comparing with: {branch_name}"
                )

                # Use provider's compare_branches method
                provider = code_service.service_instance.provider
                logging.info(
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
                logging.info(
                    f"[CHANGE_DETECTION] Comparison complete: {len(patches_dict)} files with patches, {comparison_result['commits']} commits"
                )

            elif isinstance(code_service.service_instance, GithubService):
                logging.info("[CHANGE_DETECTION] Using GithubService for diff")
                github, _, _ = code_service.service_instance.get_github_repo_details(
                    repo_name
                )
                logging.info("[CHANGE_DETECTION] Got github client from service")

                # Get the actual repo name for API calls (handles GitBucket conversion)
                from app.modules.parsing.utils.repo_name_normalizer import (
                    get_actual_repo_name_for_lookup,
                )
                import os

                provider_type = os.getenv("CODE_PROVIDER", "github").lower()
                actual_repo_name = get_actual_repo_name_for_lookup(
                    repo_name, provider_type
                )
                logging.info(
                    f"[CHANGE_DETECTION] Provider type: {provider_type}, Original repo: {repo_name}, Actual repo for API: {actual_repo_name}"
                )

                repo = github.get_repo(actual_repo_name)
                logging.info(f"[CHANGE_DETECTION] Got repo object: {repo.name}")
                default_branch = repo.default_branch
                logging.info(
                    f"[CHANGE_DETECTION] Default branch: {default_branch}, comparing with: {branch_name}"
                )

                # GitBucket workaround: Use commits API to get diff
                if provider_type == "gitbucket":

                    logging.info(
                        "[CHANGE_DETECTION] Using commits API for GitBucket diff"
                    )

                    try:
                        # Get commits on the branch
                        logging.info(
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
                                    logging.info(
                                        f"[CHANGE_DETECTION] Reached common ancestor at commit {commit.sha[:7]}"
                                    )
                                    break
                            except:
                                pass

                            # Get the commit details with files
                            logging.info(
                                f"[CHANGE_DETECTION] Processing commit {commit.sha[:7]}: {commit.commit.message.split(chr(10))[0]}"
                            )

                            for file in commit.files:
                                if file.patch and file.filename not in patches_dict:
                                    patches_dict[file.filename] = file.patch
                                    logging.info(
                                        f"[CHANGE_DETECTION] Added patch for file: {file.filename}"
                                    )

                            # Limit to reasonable number of commits
                            if commit_count >= 50:
                                logging.warning(
                                    "[CHANGE_DETECTION] Reached commit limit of 50, stopping"
                                )
                                break

                        logging.info(
                            f"[CHANGE_DETECTION] GitBucket diff complete: {len(patches_dict)} files with patches from {commit_count} commits"
                        )
                    except Exception as api_error:
                        logging.error(
                            f"[CHANGE_DETECTION] GitBucket commits API error: {type(api_error).__name__}: {str(api_error)}",
                            exc_info=True,
                        )
                        raise
                else:
                    # Use PyGithub for GitHub
                    git_diff = repo.compare(default_branch, branch_name)
                    logging.info(
                        f"[CHANGE_DETECTION] Comparison complete, files changed: {len(git_diff.files)}"
                    )
                    patches_dict = {
                        file.filename: file.patch
                        for file in git_diff.files
                        if file.patch
                    }
                    logging.info(
                        f"[CHANGE_DETECTION] Patches extracted: {len(patches_dict)} files with patches"
                    )
            elif isinstance(code_service.service_instance, LocalRepoService):
                logging.info("[CHANGE_DETECTION] Using LocalRepoService for diff")
                patches_dict = code_service.service_instance.get_local_repo_diff(
                    repo_path, branch_name
                )
                logging.info(
                    f"[CHANGE_DETECTION] Local diff complete: {len(patches_dict)} files"
                )
        except Exception as e:
            logging.error(
                f"[CHANGE_DETECTION] Exception during diff: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=400, detail=f"Error while fetching changes: {str(e)}"
            )
        finally:
            if project_details is not None:
                logging.info(
                    f"[CHANGE_DETECTION] Processing patches: {len(patches_dict)} files"
                )
                identifiers = []
                node_ids = []
                try:
                    identifiers = await self.get_updated_function_list(
                        patches_dict, project_id
                    )
                    logging.info(
                        f"[CHANGE_DETECTION] Found {len(identifiers)} changed functions: {identifiers}"
                    )
                    for identifier in identifiers:
                        node_id_query = " ".join(identifier.split(":"))
                        relevance_search = await self.search_service.search_codebase(
                            project_id, node_id_query
                        )
                        if relevance_search:
                            node_id = relevance_search[0]["node_id"]
                            if node_id:
                                node_ids.append(node_id)
                        else:
                            node_ids.append(
                                GetCodeFromNodeIdTool(self.sql_db, self.user_id).run(
                                    project_id, identifier
                                )["node_id"]
                            )

                    # Fetch code for node ids and store in a dict
                    node_code_dict = {}
                    for node_id in node_ids:
                        node_code = GetCodeFromNodeIdTool(
                            self.sql_db, self.user_id
                        ).run(project_id, node_id)

                        # Check for errors in the response
                        if "error" in node_code:
                            logging.warning(
                                f"[CHANGE_DETECTION] Error getting code for node {node_id}: {node_code['error']}"
                            )
                            continue

                        # Check for required fields
                        if (
                            "code_content" not in node_code
                            or "file_path" not in node_code
                        ):
                            logging.warning(
                                f"[CHANGE_DETECTION] Missing required fields for node {node_id}: {node_code}"
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
                            logging.warning(
                                f"[CHANGE_DETECTION] Skipping node {node} - not in node_code_dict"
                            )
                            continue

                        entry_point_code = GetCodeFromNodeIdTool(
                            self.sql_db, self.user_id
                        ).run(project_id, entry_point[0])

                        # Check for errors in entry_point_code
                        if "error" in entry_point_code:
                            logging.warning(
                                f"[CHANGE_DETECTION] Error getting entry point code for {entry_point[0]}: {entry_point_code['error']}"
                            )
                            continue

                        # Check for required fields in entry_point_code
                        if (
                            "code_content" not in entry_point_code
                            or "file_path" not in entry_point_code
                        ):
                            logging.warning(
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

                    result = ChangeDetectionResponse(
                        patches=patches_dict, changes=changes_list
                    )
                    logging.info(
                        f"[CHANGE_DETECTION] Returning result with {len(patches_dict)} patches and {len(changes_list)} changes"
                    )
                    return result
                except Exception as e:
                    logging.error(
                        f"[CHANGE_DETECTION] Exception in finally block - project_id: {project_id}, error: {type(e).__name__}: {str(e)}",
                        exc_info=True,
                    )

                if len(identifiers) == 0:
                    logging.info(
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

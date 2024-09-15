import logging
import os
import re

from fastapi import HTTPException
from tree_sitter_languages import get_parser

from app.core.database import get_db
from app.modules.github.github_service import GithubService
from app.modules.intelligence.tools.code_query_tools.get_code_from_node_name_tool import (
    GetCodeFromNodeNameTool,
)
from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.projects.projects_service import ProjectService

parser = get_parser("python")


class ChangeDetection:
    def __init__(self, sql_db):
        self.sql_db = sql_db

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

    def extract_file_name(self, repo_name, branch_name, path):
        try:
            pattern = self.get_pattern(repo_name, branch_name)

            match = re.search(pattern, path)
            if match:
                file_path = match.group(1)
                return file_path
            else:
                return None
        except ValueError as e:
            logging.error(f"Exception {e}")
            return None

    def get_pattern(self, repo_name, branch_name):
        # Define regex patterns for POSIX (Linux/macOS) and Windows
        posix_pattern = re.escape(f"{repo_name}-{branch_name}") + r"-\w+\/(.+)"
        windows_pattern = re.escape(f"{repo_name}-{branch_name}") + r"-\w+\\(.+)"

        # Check the operating system
        if os.name == "posix":
            pattern = posix_pattern
        elif os.name == "nt":
            pattern = windows_pattern
        else:
            raise ValueError("Unsupported operating system")

        return pattern

    async def _find_changed_functions(self, changed_files, repo_id):
        result = []
        for relative_file_path, lines in changed_files.items():
            try:
                project = await ProjectService(self.sql_db).get_project_from_db_by_id(
                    repo_id
                )
                github_service = GithubService(self.sql_db)
                file_content = github_service.get_file_content(
                    project["project_name"],
                    relative_file_path,
                    0,
                    0,
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
                            node_type = "class"
                        elif tag.type == "function":
                            node_type = "function"
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

    async def get_updated_function_list(self, patch_details, repo_id):
        changed_files = self._parse_diff_detail(patch_details)
        return await self._find_changed_functions(changed_files, repo_id)

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
        result = tx.run(query, endpoint_id=node_id, project_id=project_id)
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
                neighbors_fn=ChangeDetection._find_inbound_neighbors,
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
                neighbors_fn=ChangeDetection._find_inbound_neighbors,
            )
            if len(traversal_result) == 1:
                entry_points.add(node)

        return entry_points

    async def get_changes(self, project_id):
        global patches_dict, repo
        patches_dict = {}
        project_details = await ProjectService(self.sql_db).get_project_from_db_by_id(
            project_id
        )

        if project_details is None:
            raise HTTPException(status_code=400, detail="Project Details not found.")

        repo_name = project_details["project_name"]
        branch_name = project_details["branch_name"]
        github = None

        github, _, _ = GithubService(self.sql_db).get_github_repo_details(repo_name)

        try:
            repo = github.get_repo(repo_name)
            repo_details = repo
            default_branch = repo.default_branch
        except Exception:
            raise HTTPException(status_code=400, detail="Repository not found")

        try:
            git_diff = repo.compare(default_branch, branch_name)
            patches_dict = {
                file.filename: file.patch for file in git_diff.files if file.patch
            }

        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error while fetching changes: {str(e)}"
            )
        finally:
            if project_details is not None:
                identifiers = []
                node_ids = []
                try:
                    identifiers = await self.get_updated_function_list(
                        patches_dict, project_id
                    )
                    for identifier in identifiers:
                        node_ids.append(
                            GetCodeFromNodeNameTool(self.sql_db).get_node_data(
                                project_id, identifier
                            )["node_id"]
                        )
                    entry_points = InferenceService(
                        self.sql_db
                    ).get_entry_points_for_nodes(node_ids, project_id)
                    return entry_points
                except Exception as e:
                    logging.error(f"project_id: {project_id}, error: {str(e)}")

                if len(identifiers) == 0:
                    if github:
                        github.close()
                    return []
                if github:
                    github.close()


if __name__ == "__main__":
    # Hardcoded project ID for debugging
    project_id = "0191f3a3-349e-71eb-aa80-d20681355c71"  # Replace with the actual project ID you want to test

    # Initialize ChangeDetection with a mock SQL database connection
    sql_db = next(get_db())  # Replace with actual database connection
    change_detection = ChangeDetection(sql_db)
    import asyncio

    try:
        changes = asyncio.run(change_detection.get_changes(project_id))
        print("Changes:", changes)
    except HTTPException as e:
        print(f"HTTP Exception: {e.detail}")
    except Exception as e:
        print(f"Error: {str(e)}")

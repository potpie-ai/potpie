from typing import Dict, List

from sqlalchemy.orm import Session

from app.modules.intelligence.tools.change_detection.change_detection_tool import (
    get_change_detection_tool,
)
from app.modules.intelligence.tools.change_detection.patch_extraction_tool import (
    get_patch_extraction_tool,
)
from app.modules.intelligence.tools.change_detection.process_large_pr_tool import (
    get_process_large_pr_tool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_file_structure import (
    get_code_file_structure_tool,
    GetCodeFileStructureTool,
)
from app.modules.intelligence.tools.code_query_tools.code_analysis import (
    universal_analyze_code_tool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    get_code_graph_from_node_id_tool,
    GetCodeGraphFromNodeIdTool,
)
from app.modules.intelligence.tools.code_query_tools.get_node_neighbours_from_node_id_tool import (
    get_node_neighbours_from_node_id_tool,
)
from app.modules.intelligence.tools.code_query_tools.intelligent_code_graph_tool import (
    get_intelligent_code_graph_tool,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    get_ask_knowledge_graph_queries_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    get_code_from_multiple_node_ids_tool,
    GetCodeFromMultipleNodeIdsTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)
from app.modules.intelligence.tools.code_query_tools.get_file_content_by_path import (
    fetch_file_tool,
)
from app.modules.intelligence.tools.misc_tools.generate_diff_tool import (
    generate_patch_diff_tool,
)
from app.modules.intelligence.tools.misc_tools.verify_diff_tool import (
    verify_patch_diff_tool,
)
from app.modules.intelligence.tools.tool_schema import ToolInfo, ToolInfoWithParameters
from app.modules.intelligence.tools.web_tools import (
    github_create_branch,
    github_update_branch,
    github_create_pr,
    github_add_pr_comment,
)
from app.modules.intelligence.tools.web_tools.github_tool import github_tool
from app.modules.intelligence.tools.web_tools.webpage_extractor_tool import (
    webpage_extractor_tool,
)
from app.modules.intelligence.tools.linear_tools import (
    get_linear_issue_tool,
    update_linear_issue_tool,
)
from app.modules.intelligence.tools.web_tools.web_search_tool import web_search_tool
from app.modules.intelligence.provider.provider_service import ProviderService
from langchain_core.tools import StructuredTool
from .think_tool import think_tool
from .todo_management_tool import create_todo_management_tools
from .code_changes_manager import create_code_changes_management_tools


class ToolService:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.webpage_extractor_tool = webpage_extractor_tool(db, user_id)
        self.web_search_tool = web_search_tool(db, user_id)
        self.github_tool = github_tool(db, user_id)
        self.get_code_from_multiple_node_ids_tool = GetCodeFromMultipleNodeIdsTool(
            self.db, self.user_id
        )
        self.get_code_graph_from_node_id_tool = GetCodeGraphFromNodeIdTool(db)
        self.file_structure_tool = GetCodeFileStructureTool(db)
        self.provider_service = ProviderService.create(db, user_id)
        self.tools = self._initialize_tools()
        self.process_large_pr_tool = get_process_large_pr_tool(user_id, db)

    def get_tools(self, tool_names: List[str]) -> List[StructuredTool]:
        """get tools if exists"""
        tools = []
        for tool_name in tool_names:
            if self.tools.get(tool_name) is not None:
                tools.append(self.tools[tool_name])
        return tools

    def _initialize_tools(self) -> Dict[str, StructuredTool]:
        tools = {
            "get_code_from_probable_node_names": get_code_from_probable_node_name_tool(
                self.db, self.user_id
            ),
            "get_code_from_node_id": get_code_from_node_id_tool(self.db, self.user_id),
            "get_code_from_multiple_node_ids": get_code_from_multiple_node_ids_tool(
                self.db, self.user_id
            ),
            "ask_knowledge_graph_queries": get_ask_knowledge_graph_queries_tool(
                self.db, self.user_id
            ),
            "get_nodes_from_tags": get_nodes_from_tags_tool(self.db, self.user_id),
            "get_code_graph_from_node_id": get_code_graph_from_node_id_tool(self.db),
            "change_detection": get_change_detection_tool(self.user_id, self.db),
            "get_code_file_structure": get_code_file_structure_tool(self.db),
            "get_node_neighbours_from_node_id": get_node_neighbours_from_node_id_tool(
                self.db
            ),
            "get_linear_issue": get_linear_issue_tool(self.db, self.user_id),
            "update_linear_issue": update_linear_issue_tool(self.db, self.user_id),
            "intelligent_code_graph": get_intelligent_code_graph_tool(
                self.db, self.provider_service, self.user_id
            ),
            "think": think_tool(self.db, self.user_id),
            "github_create_branch": github_create_branch.github_create_branch_tool(
                self.db, self.user_id
            ),
            "github_update_branch": github_update_branch.github_update_branch_tool(
                self.db, self.user_id
            ),
            "github_create_pull_request": github_create_pr.github_create_pull_request_tool(
                self.db, self.user_id
            ),
            "github_add_pr_comments": github_add_pr_comment.git_add_pr_comments_tool(
                self.db, self.user_id
            ),
            "get_patch_tool": get_patch_extraction_tool(self.user_id),
            "process_large_pr_tool": get_process_large_pr_tool(self.user_id, self.db),
            "fetch_file": fetch_file_tool(self.db, self.user_id),
            "analyze_code_structure": universal_analyze_code_tool(
                self.db, self.user_id
            ),
        }

        # Add todo management tools
        todo_tools = create_todo_management_tools()
        for tool in todo_tools:
            tools[tool.name] = tool

        # Add code changes management tools
        code_changes_tools = create_code_changes_management_tools()
        for tool in code_changes_tools:
            tools[tool.name] = tool

        if self.webpage_extractor_tool:
            tools["webpage_extractor"] = self.webpage_extractor_tool

        if self.github_tool:
            tools["github_tool"] = self.github_tool

        if self.web_search_tool:
            tools["web_search_tool"] = self.web_search_tool

        file_manager = file_change_manager.FileChangeManager()
        tools["generate_patch_diff"] = generate_diffs_tool.generate_file_diff_tool(
            file_manager
        )
        tools["load_file_for_editing"] = (
            load_file_for_editing_tool.load_file_for_editing_tool(
                fetch_file_tool(self.db, self.user_id), file_manager
            )
        )
        tools["replace_lines_in_file"] = replace_lines_in_file_tool.replace_lines_tool(
            file_manager
        )
        tools["insert_lines_in_file"] = insert_lines_in_file_tool.insert_lines_tool(
            file_manager
        )
        tools["remove_lines_in_file"] = remove_lines_in_file_tool.delete_lines_tool(
            file_manager
        )
        tools["read_lines_in_changed_file"] = (
            read_lines_in_changed_file_tool.get_lines_from_file_tool(file_manager)
        )
        tools["search_in_file"] = search_pattern_in_file.search_in_file_tool(
            file_manager
        )

        return tools

    def list_tools(self) -> List[ToolInfo]:
        return [
            ToolInfo(
                id=tool_id,
                name=tool.name,
                description=tool.description,
            )
            for tool_id, tool in self.tools.items()
        ]

    def list_tools_with_parameters(self) -> Dict[str, ToolInfoWithParameters]:
        return {
            tool_id: ToolInfoWithParameters(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema.schema() if tool.args_schema else {},
            )
            for tool_id, tool in self.tools.items()
        }

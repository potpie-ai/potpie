from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.modules.intelligence.tools.change_detection.change_detection_tool import (
    get_change_detection_tool,
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
from app.modules.intelligence.tools.code_query_tools.bash_command_tool import (
    bash_command_tool,
)
from app.modules.intelligence.tools.tool_schema import ToolInfo, ToolInfoWithParameters
from app.modules.intelligence.tools.web_tools.code_provider_tool import (
    code_provider_tool,
)
from app.modules.intelligence.tools.web_tools.code_provider_create_branch import (
    code_provider_create_branch_tool,
)
from app.modules.intelligence.tools.web_tools.code_provider_create_pr import (
    code_provider_create_pull_request_tool,
)
from app.modules.intelligence.tools.web_tools.code_provider_add_pr_comment import (
    code_provider_add_pr_comments_tool,
)
from app.modules.intelligence.tools.web_tools.code_provider_update_file import (
    code_provider_update_file_tool,
)
from app.modules.intelligence.tools.web_tools.webpage_extractor_tool import (
    webpage_extractor_tool,
)
from app.modules.intelligence.tools.linear_tools import (
    get_linear_issue_tool,
    update_linear_issue_tool,
)
from app.modules.intelligence.tools.jira_tools import (
    get_jira_issue_tool,
    search_jira_issues_tool,
    create_jira_issue_tool,
    update_jira_issue_tool,
    add_jira_comment_tool,
    transition_jira_issue_tool,
    get_jira_projects_tool,
    get_jira_project_details_tool,
    get_jira_project_users_tool,
    link_jira_issues_tool,
)
from app.modules.intelligence.tools.confluence_tools import (
    get_confluence_spaces_tool,
    get_confluence_page_tool,
    search_confluence_pages_tool,
    get_confluence_space_pages_tool,
    create_confluence_page_tool,
    update_confluence_page_tool,
    add_confluence_comment_tool,
)
from app.modules.intelligence.tools.web_tools.web_search_tool import web_search_tool
from app.modules.intelligence.provider.provider_service import ProviderService
from langchain_core.tools import StructuredTool
from .todo_management_tool import create_todo_management_tools
from .code_changes_manager import create_code_changes_management_tools
from .requirement_verification_tool import create_requirement_verification_tools


class ToolService:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.webpage_extractor_tool = webpage_extractor_tool(db, user_id)
        self.web_search_tool = web_search_tool(db, user_id)
        self.code_provider_tool = code_provider_tool(db, user_id)
        self.code_provider_create_branch_tool = code_provider_create_branch_tool(
            db, user_id
        )
        self.code_provider_create_pr_tool = code_provider_create_pull_request_tool(
            db, user_id
        )
        self.code_provider_add_pr_comments_tool = code_provider_add_pr_comments_tool(
            db, user_id
        )
        self.code_provider_update_file_tool = code_provider_update_file_tool(
            db, user_id
        )
        self.get_code_from_multiple_node_ids_tool = GetCodeFromMultipleNodeIdsTool(
            self.db, self.user_id
        )
        self.get_code_graph_from_node_id_tool = GetCodeGraphFromNodeIdTool(db)
        self.file_structure_tool = GetCodeFileStructureTool(db)
        self.provider_service = ProviderService.create(db, user_id)
        self.tools = self._initialize_tools()

    def get_tools(self, tool_names: List[str]) -> List[StructuredTool]:
        """get tools if exists"""
        tools = []
        for tool_name in tool_names:
            if self.tools.get(tool_name) is not None:
                tools.append(self.tools[tool_name])
        return tools

    def _initialize_tools(self) -> Dict[str, Any]:
        tools: Dict[str, Any] = {
            "get_code_from_probable_node_name": get_code_from_probable_node_name_tool(
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
            "change_detection": get_change_detection_tool(self.user_id),
            "get_code_file_structure": get_code_file_structure_tool(self.db),
            "get_node_neighbours_from_node_id": get_node_neighbours_from_node_id_tool(
                self.db
            ),
            "get_linear_issue": get_linear_issue_tool(self.db, self.user_id),
            "update_linear_issue": update_linear_issue_tool(self.db, self.user_id),
            "get_jira_issue": get_jira_issue_tool(self.db, self.user_id),
            "search_jira_issues": search_jira_issues_tool(self.db, self.user_id),
            "create_jira_issue": create_jira_issue_tool(self.db, self.user_id),
            "update_jira_issue": update_jira_issue_tool(self.db, self.user_id),
            "add_jira_comment": add_jira_comment_tool(self.db, self.user_id),
            "transition_jira_issue": transition_jira_issue_tool(self.db, self.user_id),
            "get_jira_projects": get_jira_projects_tool(self.db, self.user_id),
            "get_jira_project_details": get_jira_project_details_tool(
                self.db, self.user_id
            ),
            "get_jira_project_users": get_jira_project_users_tool(
                self.db, self.user_id
            ),
            "link_jira_issues": link_jira_issues_tool(self.db, self.user_id),
            "get_confluence_spaces": get_confluence_spaces_tool(self.db, self.user_id),
            "get_confluence_page": get_confluence_page_tool(self.db, self.user_id),
            "search_confluence_pages": search_confluence_pages_tool(
                self.db, self.user_id
            ),
            "get_confluence_space_pages": get_confluence_space_pages_tool(
                self.db, self.user_id
            ),
            "create_confluence_page": create_confluence_page_tool(
                self.db, self.user_id
            ),
            "update_confluence_page": update_confluence_page_tool(
                self.db, self.user_id
            ),
            "add_confluence_comment": add_confluence_comment_tool(
                self.db, self.user_id
            ),
            "intelligent_code_graph": get_intelligent_code_graph_tool(
                self.db, self.provider_service, self.user_id
            ),
            "fetch_file": fetch_file_tool(self.db, self.user_id),
            "analyze_code_structure": universal_analyze_code_tool(
                self.db, self.user_id
            ),
        }

        # Add bash command tool if repo manager is enabled
        bash_tool = bash_command_tool(self.db, self.user_id)
        if bash_tool:
            tools["bash_command"] = bash_tool
        # Add todo management tools
        todo_tools = create_todo_management_tools()
        for tool in todo_tools:
            tools[tool.name] = tool

        # Add code changes management tools
        code_changes_tools = create_code_changes_management_tools()
        for tool in code_changes_tools:
            tools[tool.name] = tool

        # Add requirement verification tools
        requirement_tools = create_requirement_verification_tools()
        for tool in requirement_tools:
            tools[tool.name] = tool

        if self.webpage_extractor_tool:
            tools["webpage_extractor"] = self.webpage_extractor_tool

        if self.code_provider_tool:
            tools["code_provider_tool"] = self.code_provider_tool
            tools["github_tool"] = self.code_provider_tool

        if self.code_provider_create_branch_tool:
            tools["code_provider_create_branch"] = self.code_provider_create_branch_tool
            tools["github_create_branch"] = self.code_provider_create_branch_tool

        if self.code_provider_create_pr_tool:
            tools["code_provider_create_pr"] = self.code_provider_create_pr_tool
            tools["github_create_pull_request"] = self.code_provider_create_pr_tool

        if self.code_provider_add_pr_comments_tool:
            tools["code_provider_add_pr_comments"] = (
                self.code_provider_add_pr_comments_tool
            )
            tools["github_add_pr_comments"] = self.code_provider_add_pr_comments_tool

        if self.code_provider_update_file_tool:
            tools["code_provider_update_file"] = self.code_provider_update_file_tool
            tools["github_update_branch"] = self.code_provider_update_file_tool

        if self.web_search_tool:
            tools["web_search_tool"] = self.web_search_tool

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

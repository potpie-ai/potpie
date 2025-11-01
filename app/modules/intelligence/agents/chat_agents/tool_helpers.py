from typing import Any, Dict, List

# Import todo management for streaming todo list state
_todo_import_failed = False
try:
    from app.modules.intelligence.tools.todo_management_tool import (
        _format_current_todo_list,
    )
except ImportError:
    # Fallback if todo management tool is not available
    _todo_import_failed = True

    def _format_current_todo_list() -> str:
        return (
            "📋 **Current Todo List:** (Todo management not available - import failed)"
        )


def get_tool_run_message(tool_name: str):
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            return "Retrieving code"
        case "GetCodeanddocstringFromNodeID":
            return "Retrieving code for referenced mentions"
        case "Getcodechanges":
            return "Fetching code changes from your repo"
        case "GetNodesfromTags":
            return "Fetching code"
        case "AskKnowledgeGraphQueries":
            return "Traversing the knowledge graph"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return "Fetching code and docstrings"
        case "get_code_file_structure":
            return "Loading the dir structure"
        case "GetNodeNeighboursFromNodeID":
            return "Expanding code context"
        case "WebpageContentExtractor":
            return "Querying information from the web"
        case "GitHubContentFetcher":
            return "Fetching content from github"
        case "fetch_file":
            return "Fetching file content"
        case "WebSearchTool":
            return "Searching the web"
        case "analyze_code_structure":
            return "Analyzing code structure"
        case "create_todo":
            return "Creating todo item"
        case "update_todo_status":
            return "Updating todo status"
        case "add_todo_note":
            return "Adding todo note"
        case "get_todo":
            return "Retrieving todo details"
        case "list_todos":
            return "Listing todos"
        case "get_todo_summary":
            return "Getting todo summary"
        case "add_file_to_changes":
            return "Adding file to code changes"
        case "update_file_in_changes":
            return "Updating file in code changes"
        case "update_file_lines":
            return "Updating specific lines in file"
        case "replace_in_file":
            return "Replacing pattern in file"
        case "insert_lines":
            return "Inserting lines in file"
        case "delete_lines":
            return "Deleting lines from file"
        case "delete_file_in_changes":
            return "Marking file for deletion"
        case "get_file_from_changes":
            return "Retrieving file from code changes"
        case "list_files_in_changes":
            return "Listing files in code changes"
        case "search_content_in_changes":
            return "Searching content in code changes"
        case "clear_file_from_changes":
            return "Clearing file from code changes"
        case "clear_all_changes":
            return "Clearing all code changes"
        case "get_changes_summary":
            return "Getting code changes summary"
        case "export_changes":
            return "Exporting code changes"
        case "show_diff":
            return "Displaying code diff"
        case tool_name if tool_name.startswith("delegate_to_"):
            # Handle delegation tools - extract agent type and return appropriate message
            agent_type = tool_name[12:]  # Remove "delegate_to_" prefix
            return get_delegation_call_message(agent_type)
        case _:
            return "Querying data"


def get_tool_response_message(tool_name: str):
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            return "Code retrieved"
        case "Getcodechanges":
            return "Code changes fetched successfully"
        case "GetNodesfromTags":
            return "Code fetched"
        case "AskKnowledgeGraphQueries":
            return "Knowledge graph queries successful"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return "Fetched code and docstrings"
        case "get_code_file_structure":
            return "Project file structure loaded successfully"
        case "GetNodeNeighboursFromNodeID":
            return "Fetched referenced code"
        case "WebpageContentExtractor":
            return "Information retrieved from web"
        case "GitHubContentFetcher":
            return "File contents fetched from github"
        case "fetch_file":
            return "File content fetched successfully"
        case "analyze_code_structure":
            return "Code structure analyzed successfully"
        case "WebSearchTool":
            return "Web search successful"
        case "create_todo":
            return "Todo item created successfully"
        case "update_todo_status":
            return "Todo status updated successfully"
        case "add_todo_note":
            return "Todo note added successfully"
        case "get_todo":
            return "Todo details retrieved successfully"
        case "list_todos":
            return "Todos listed successfully"
        case "get_todo_summary":
            return "Todo summary generated successfully"
        case "add_file_to_changes":
            return "File added to code changes successfully"
        case "update_file_in_changes":
            return "File updated in code changes successfully"
        case "update_file_lines":
            return "File lines updated successfully"
        case "replace_in_file":
            return "Pattern replacement completed successfully"
        case "insert_lines":
            return "Lines inserted successfully"
        case "delete_lines":
            return "Lines deleted successfully"
        case "delete_file_in_changes":
            return "File marked for deletion successfully"
        case "get_file_from_changes":
            return "File retrieved from code changes successfully"
        case "list_files_in_changes":
            return "Files listed successfully"
        case "search_content_in_changes":
            return "Content search completed successfully"
        case "clear_file_from_changes":
            return "File cleared from code changes successfully"
        case "clear_all_changes":
            return "All code changes cleared successfully"
        case "get_changes_summary":
            return "Code changes summary retrieved successfully"
        case "export_changes":
            return "Code changes exported successfully"
        case "show_diff":
            return "Code diff displayed successfully"
        case tool_name if tool_name.startswith("delegate_to_"):
            # Handle delegation tools - extract agent type and return appropriate message
            agent_type = tool_name[12:]  # Remove "delegate_to_" prefix
            return get_delegation_response_message(agent_type)
        case _:
            return "Data queried successfully"


def get_tool_call_info_content(tool_name: str, args: Dict[str, Any]) -> str:
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            node_names = args.get("probable_node_names")
            if isinstance(node_names, List):
                return "-> checking following nodes: \n" + "\n- ".join(node_names)
            return "-> checking probable nodes"
        case "Getcodechanges":
            return ""
        case "GetNodesfromTags":
            return ""
        case "AskKnowledgeGraphQueries":
            queries = args.get("queries")
            if isinstance(queries, List):
                return "".join([f"\n- {query}" for query in queries])
            return ""
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return ""
        case "get_code_file_structure":
            path = args.get("path")
            if path != None and isinstance(path, str):
                return f"-> fetching directory structure for {path}\n"
            return "-> fetching directory structure from root of the repo\n"
        case "GetNodeNeighboursFromNodeID":
            return ""
        case "WebpageContentExtractor":
            return f"fetching -> {args.get('url')}\n"
        case "GitHubContentFetcher":
            repo_name = args.get("repo_name")
            issue_number = args.get("issue_number")
            is_pr = args.get("is_pull_request")
            if repo_name and issue_number:
                return f"-> fetching {'PR' if is_pr else 'Issue'} #{issue_number} from github/{repo_name}\n"
            return ""
        case "fetch_file":
            return f"fetching contents for file {args.get('file_path')}"
        case "analyze_code_structure":
            return f"Analyzing file - {args.get('file_path')}\n"
        case "WebSearchTool":
            return f"-> searching the web for {args.get('query')}\n"
        case "create_todo":
            title = args.get("title", "")
            priority = args.get("priority", "medium")
            return f"-> creating todo: '{title}' (priority: {priority})\n"
        case "update_todo_status":
            todo_id = args.get("todo_id", "")
            status = args.get("status", "")
            return f"-> updating todo {todo_id} to status: {status}\n"
        case "add_todo_note":
            todo_id = args.get("todo_id", "")
            return f"-> adding note to todo {todo_id}\n"
        case "get_todo":
            todo_id = args.get("todo_id", "")
            return f"-> retrieving details for todo {todo_id}\n"
        case "list_todos":
            status_filter = args.get("status_filter")
            if status_filter:
                return f"-> listing todos with status: {status_filter}\n"
            return "-> listing all todos\n"
        case "get_todo_summary":
            return "-> generating todo summary\n"
        case "add_file_to_changes":
            file_path = args.get("file_path", "")
            return f"-> adding file to code changes: {file_path}\n"
        case "update_file_in_changes":
            file_path = args.get("file_path", "")
            return f"-> updating file in code changes: {file_path}\n"
        case "update_file_lines":
            file_path = args.get("file_path", "")
            start_line = args.get("start_line", "")
            end_line = args.get("end_line", "")
            line_range = f"{start_line}-{end_line}" if end_line else str(start_line)
            return f"-> updating lines {line_range} in file: {file_path}\n"
        case "replace_in_file":
            file_path = args.get("file_path", "")
            pattern = args.get("pattern", "")
            return f"-> replacing pattern '{pattern}' in file: {file_path}\n"
        case "insert_lines":
            file_path = args.get("file_path", "")
            line_number = args.get("line_number", "")
            insert_after = args.get("insert_after", True)
            position = "after" if insert_after else "before"
            return f"-> inserting lines {position} line {line_number} in file: {file_path}\n"
        case "delete_lines":
            file_path = args.get("file_path", "")
            start_line = args.get("start_line", "")
            end_line = args.get("end_line", "")
            line_range = f"{start_line}-{end_line}" if end_line else str(start_line)
            return f"-> deleting lines {line_range} from file: {file_path}\n"
        case "delete_file_in_changes":
            file_path = args.get("file_path", "")
            return f"-> marking file for deletion: {file_path}\n"
        case "get_file_from_changes":
            file_path = args.get("file_path", "")
            return f"-> retrieving file from code changes: {file_path}\n"
        case "list_files_in_changes":
            change_type = args.get("change_type_filter", "")
            path_pattern = args.get("path_pattern", "")
            filters = []
            if change_type:
                filters.append(f"type: {change_type}")
            if path_pattern:
                filters.append(f"path: {path_pattern}")
            filter_text = f" ({', '.join(filters)})" if filters else ""
            return f"-> listing files in code changes{filter_text}\n"
        case "search_content_in_changes":
            pattern = args.get("pattern", "")
            file_pattern = args.get("file_pattern", "")
            return (
                f"-> searching content in code changes: pattern '{pattern}'"
                + (f" in files matching '{file_pattern}'" if file_pattern else "")
                + "\n"
            )
        case "clear_file_from_changes":
            file_path = args.get("file_path", "")
            return f"-> clearing file from code changes: {file_path}\n"
        case "clear_all_changes":
            return "-> clearing all code changes\n"
        case "get_changes_summary":
            return "-> getting code changes summary\n"
        case "export_changes":
            format_type = args.get("format", "dict")
            return f"-> exporting code changes in {format_type} format\n"
        case "show_diff":
            return "-> displaying code diff\n"
        case tool_name if tool_name.startswith("delegate_to_"):
            # Handle delegation tools - extract agent type and return appropriate info
            agent_type = tool_name[12:]  # Remove "delegate_to_" prefix
            task_description = args.get("task_description", "")
            context = args.get("context", "")
            return get_delegation_info_content(agent_type, task_description, context)
        case _:
            return ""


def get_tool_result_info_content(tool_name: str, content: List[Any] | str | Any) -> str:
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            if isinstance(content, List):
                try:
                    res = "\n-> retrieved code snippets: \n" + "\n- content:\n".join(
                        [
                            f"""
```{str(node.get('code_content'))[:min(len(str(node.get('code_content'))),600)]+" ..."}
```
"""
                            for node in content
                        ]
                    )
                except:
                    return ""
                return res
            return ""
        case "Getcodechanges":
            return "successfull"
        case "GetNodesfromTags":
            return ""
        case "AskKnowledgeGraphQueries":
            return "\n docstrings retrieved for the queries"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            if isinstance(content, Dict):
                res = content.values()
                text = ""
                for item in res:
                    if item and isinstance(item, Dict):
                        path = item.get("relative_file_path")
                        code_content = item.get("code_content")
                        if code_content:
                            text += f"{path}\n```{code_content[:min(len(code_content),300)]}``` \n"
                        elif item.get("error") != None:
                            text += f"Error: {item.get('error')} \n"
                return text
            return ""
        case "get_code_file_structure":
            if isinstance(content, str):
                return f"""-> fetched successfully
```
---------------
{content[:min(len(content),600)]} ...
---------------
```
            """
            return "File structure of the repo loaded successfully"
        case "GetNodeNeighboursFromNodeID":
            return "successful"
        case "WebpageContentExtractor":
            if isinstance(content, Dict):
                res = content.get("content")
                if isinstance(res, str):
                    return res[: min(len(res), 600)] + " ..."
            return ""
        case "GitHubContentFetcher":
            if isinstance(content, Dict):
                _content = content.get("content")
                if isinstance(_content, Dict):
                    title = _content.get("title")
                    status = _content.get("state")
                    body = _content.get("body")
                    res = f"""

# ***{title}***

## status: {status}
description:
{body}
"""
                    return res[: min(len(res), 600)] + " ..."
            return ""
        case "GetCodeanddocstringFromNodeID":
            if isinstance(content, Dict):
                path = content.get("file_path")
                code = content.get("code_content")
                res = f"""
# {path}
```{code}```
"""
                return res[: min(len(res), 600)] + " ..."
            return ""
        case "fetch_file":
            if isinstance(content, Dict):
                if not content.get("success"):
                    return "Failed to fetch content"
                else:
                    return f"""
```{content.get("content")}```
                """
            return ""
        case "analyze_code_structure":
            if isinstance(content, Dict):
                if not content.get("success"):
                    return "Failed to analyze code structure"
                else:
                    elements = content.get("elements")
                    if elements:
                        return f"""
{[ f''' {element.get("type")}: {element.get("name")} ''' for element in elements]}
"""
            return ""
        case "WebSearchTool":
            if isinstance(content, Dict):
                res = content.get("content")
                if isinstance(res, str):
                    return res[: min(len(res), 600)] + " ..."
            return ""
        case (
            "create_todo"
            | "update_todo_status"
            | "add_todo_note"
            | "get_todo"
            | "list_todos"
            | "get_todo_summary"
        ):
            # For todo tools, return the content directly as it already includes the formatted todo list
            if isinstance(content, str):
                return content
            return str(content)
        case (
            "add_file_to_changes"
            | "update_file_in_changes"
            | "update_file_lines"
            | "replace_in_file"
            | "insert_lines"
            | "delete_lines"
            | "delete_file_in_changes"
            | "get_file_from_changes"
            | "list_files_in_changes"
            | "search_content_in_changes"
            | "clear_file_from_changes"
            | "clear_all_changes"
            | "get_changes_summary"
            | "export_changes"
            | "show_diff"
        ):
            # For code changes manager tools, return the content directly
            if isinstance(content, str):
                return content
            return str(content)
        case tool_name if tool_name.startswith("delegate_to_"):
            # Handle delegation tools - extract agent type and return appropriate result content
            agent_type = tool_name[12:]  # Remove "delegate_to_" prefix
            result_content = str(content) if content else ""
            return get_delegation_result_content(agent_type, result_content)
        case _:
            return ""


def get_delegation_call_message(agent_type: str) -> str:
    """Get user-friendly message when delegating to a specialist agent"""
    base_message = ""
    match agent_type:
        case "think_execute":
            base_message = "🧠 Delegating to Think & Execute Agent for problem-solving and execution"
        case _:
            base_message = f"🤖 Delegating to {agent_type} specialist"

    # Add current todo list state to show progress
    todo_list = _format_current_todo_list()
    if todo_list.strip():
        base_message += f"\n\n{todo_list}"
    elif _todo_import_failed:
        base_message += "\n\n📋 **Current Todo List:** (Todo management not available - dependencies missing)"
    else:
        base_message += "\n\n📋 **Current Todo List:** No active todos"

    return base_message


def get_delegation_response_message(agent_type: str) -> str:
    """Get user-friendly message when delegation completes (now with clean task summary)"""
    base_message = ""
    match agent_type:
        case "think_execute":
            base_message = (
                "✅ Think & Execute Agent completed task - returning clean task summary"
            )
        case _:
            base_message = f"✅ {agent_type} specialist completed task - returning clean task summary"

    # Add current todo list state to show updated progress
    todo_list = _format_current_todo_list()
    if todo_list.strip():
        base_message += f"\n\n{todo_list}"
    elif _todo_import_failed:
        base_message += "\n\n📋 **Current Todo List:** (Todo management not available - dependencies missing)"
    else:
        base_message += "\n\n📋 **Current Todo List:** No active todos"

    return base_message


def get_delegation_info_content(
    agent_type: str, task_description: str, context: str = ""
) -> str:
    """Get detailed info about what the specialist agent will do"""
    info = ""
    match agent_type:
        case "think_execute":
            info = f"**Think & Execute Task:**\n{task_description}"
            if context:
                info += f"\n\n**Context:**\n{context}"
        case _:
            info = f"**{agent_type} Task:**\n{task_description}"
            if context:
                info += f"\n\n**Context:**\n{context}"

    # Add current todo list state for context
    todo_list = _format_current_todo_list()
    if todo_list.strip():
        info += f"\n\n{todo_list}"
    elif _todo_import_failed:
        info += "\n\n📋 **Current Todo List:** (Todo management not available - dependencies missing)"
    else:
        info += "\n\n📋 **Current Todo List:** No active todos"

    return info


def get_delegation_info_with_todo_context(
    agent_type: str, task_description: str, todo_id: str = "", context: str = ""
) -> str:
    """Get delegation info with todo context for better tracking"""
    base_info = get_delegation_info_content(agent_type, task_description, context)

    if todo_id:
        base_info += f"\n\n**Todo ID:** {todo_id}"
        base_info += "\n*This task is being tracked in the todo management system*"

    return base_info


def get_delegation_result_content(agent_type: str, result: str) -> str:
    """Get formatted result from specialist agent (now returns detailed task result)"""
    # Display the full task result without truncation - it can be detailed and include code snippets
    display_result = result

    match agent_type:
        case "think_execute":
            return f"**Task Completion Result:**\n\n{display_result}"
        case _:
            return f"**{agent_type} Task Result:**\n\n{display_result}"

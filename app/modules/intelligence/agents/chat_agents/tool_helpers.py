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
        case "codebase_analyzer":
            base_message = (
                "🔍 Delegating to Codebase Analyzer to examine how code works"
            )
        case "codebase_locator":
            base_message = "📍 Delegating to Codebase Locator to find relevant files"
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
        case "codebase_analyzer":
            base_message = (
                "✅ Codebase Analyzer completed analysis - returning clean task summary"
            )
        case "codebase_locator":
            base_message = "✅ Codebase Locator found relevant files - returning clean task summary"
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
        case "codebase_analyzer":
            info = f"**Codebase Analyzer Task:**\n{task_description}"
            if context:
                info += f"\n\n**Context:**\n{context}"
        case "codebase_locator":
            info = f"**Codebase Locator Task:**\n{task_description}"
            if context:
                info += f"\n\n**Context:**\n{context}"
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
    """Get formatted result from specialist agent (now returns detailed task summary)"""
    # Display the full task summary without truncation - it can be detailed and include code snippets
    display_result = result

    match agent_type:
        case "codebase_analyzer":
            return f"**Codebase Analysis Summary:**\n\n{display_result}"
        case "codebase_locator":
            return f"**File Location Summary:**\n\n{display_result}"
        case "think_execute":
            return f"**Task Completion Summary:**\n\n{display_result}"
        case _:
            return f"**{agent_type} Task Summary:**\n\n{display_result}"

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


def get_tool_run_message(tool_name: str, args: Dict[str, Any] | None = None):
    """Get user-friendly message when a tool starts running.

    Args:
        tool_name: Name of the tool being run
        args: Optional dict of tool arguments to include file paths in messages
    """

    # Helper to extract file path from args
    def _get_file_path() -> str:
        if args:
            return args.get("file_path", "")
        return ""

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
            file_path = _get_file_path()
            return (
                f"Fetching file: {file_path}" if file_path else "Fetching file content"
            )
        case "WebSearchTool":
            return "Searching the web"
        case "analyze_code_structure":
            return "Analyzing code structure"
        case "bash_command":
            if args:
                command = args.get("command", "")
                if command:
                    # Truncate long commands for display
                    display_cmd = (
                        command if len(command) <= 80 else command[:77] + "..."
                    )
                    return f"Running: {display_cmd}"
            return "Executing bash command on codebase"
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
        case "add_requirements":
            return "Updating requirements document"
        case "delete_requirements":
            return "Clearing requirements document"
        case "get_requirements":
            return "Retrieving requirements document"
        case "think":
            return "Processing thoughts"
        case "add_file_to_changes":
            file_path = _get_file_path()
            return (
                f"Adding file: {file_path}"
                if file_path
                else "Adding file to code changes"
            )
        case "update_file_in_changes":
            file_path = _get_file_path()
            return (
                f"Updating file: {file_path}"
                if file_path
                else "Updating file in code changes"
            )
        case "update_file_lines":
            file_path = _get_file_path()
            return (
                f"Updating lines in: {file_path}"
                if file_path
                else "Updating specific lines in file"
            )
        case "replace_in_file":
            file_path = _get_file_path()
            return (
                f"Replacing pattern in: {file_path}"
                if file_path
                else "Replacing pattern in file"
            )
        case "insert_lines":
            file_path = _get_file_path()
            return (
                f"Inserting lines in: {file_path}"
                if file_path
                else "Inserting lines in file"
            )
        case "delete_lines":
            file_path = _get_file_path()
            return (
                f"Deleting lines from: {file_path}"
                if file_path
                else "Deleting lines from file"
            )
        case "delete_file_in_changes":
            file_path = _get_file_path()
            return (
                f"Marking for deletion: {file_path}"
                if file_path
                else "Marking file for deletion"
            )
        case "get_file_from_changes":
            file_path = _get_file_path()
            return (
                f"Retrieving changes for: {file_path}"
                if file_path
                else "Retrieving file from code changes"
            )
        case "list_files_in_changes":
            return "Listing files in code changes"
        case "search_content_in_changes":
            return "Searching content in code changes"
        case "clear_file_from_changes":
            file_path = _get_file_path()
            return (
                f"Clearing changes for: {file_path}"
                if file_path
                else "Clearing file from code changes"
            )
        case "clear_all_changes":
            return "Clearing all code changes"
        case "get_changes_summary":
            return "Getting code changes summary"
        case "export_changes":
            return "Exporting code changes"
        case "show_updated_file":
            if args:
                file_paths = args.get("file_paths")
                if file_paths:
                    if isinstance(file_paths, list):
                        return f"Displaying updated: {', '.join(file_paths)}"
                    return f"Displaying updated: {file_paths}"
            return "Displaying updated file content"
        case "show_diff":
            return "Displaying code diff"
        case tool_name if tool_name.startswith("delegate_to_"):
            # Handle delegation tools - extract agent type and return appropriate message
            agent_type = tool_name[12:]  # Remove "delegate_to_" prefix
            return get_delegation_call_message(agent_type)
        case "GetJiraIssue":
            return "Fetching Jira issue details"
        case "SearchJiraIssues":
            return "Searching Jira issues"
        case "CreateJiraIssue":
            return "Creating new Jira issue"
        case "UpdateJiraIssue":
            return "Updating Jira issue"
        case "AddJiraComment":
            return "Adding comment to Jira issue"
        case "TransitionJiraIssue":
            return "Changing Jira issue status"
        case "LinkJiraIssues":
            return "Linking Jira issues"
        case "GetJiraProjects":
            return "Fetching Jira projects"
        case "GetJiraProjectDetails":
            return "Fetching Jira project details"
        case "GetJiraProjectUsers":
            return "Fetching Jira project users"
        case "GetConfluenceSpaces":
            return "Fetching Confluence spaces"
        case "GetConfluencePage":
            return "Retrieving Confluence page"
        case "SearchConfluencePages":
            return "Searching Confluence pages"
        case "GetConfluenceSpacePages":
            return "Fetching pages in Confluence space"
        case "CreateConfluencePage":
            return "Creating new Confluence page"
        case "UpdateConfluencePage":
            return "Updating Confluence page"
        case "AddConfluenceComment":
            return "Adding comment to Confluence page"
        case _:
            return "Querying data"


def get_tool_response_message(
    tool_name: str, args: Dict[str, Any] | None = None, result: Any = None
):
    """Get user-friendly message when a tool completes.

    Args:
        tool_name: Name of the tool that completed
        args: Optional dict of tool arguments to include file paths, commands, etc.
        result: Optional tool result to include small response data
    """

    # Helper to check if result is small enough to include
    def _is_small_result(res: Any) -> bool:
        if res is None:
            return False
        if isinstance(res, str):
            return len(res) <= 200
        if isinstance(res, dict):
            # Check if it's a simple success/failure dict
            if len(res) <= 3 and all(
                k in ("success", "output", "error", "exit_code") for k in res.keys()
            ):
                return True
            return False
        return False

    # Helper to format small result
    def _format_small_result(res: Any) -> str:
        if isinstance(res, dict):
            if res.get("success") is False:
                error = res.get("error", "")
                if error and len(error) <= 150:
                    return f" (Error: {error[:150]})"
            elif res.get("success") is True:
                output = res.get("output", "")
                if output and len(output) <= 150:
                    return f" (Output: {output[:150]})"
        elif isinstance(res, str) and len(res) <= 200:
            return f" ({res[:200]})"
        return ""

    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            return "Code retrieved"
        case "GetCodeanddocstringFromNodeID":
            return "Code retrieved for referenced mentions"
        case "Getcodechanges":
            return "Code changes fetched successfully"
        case "GetNodesfromTags":
            return "Code fetched"
        case "AskKnowledgeGraphQueries":
            return "Knowledge graph queries successful"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return "Fetched code and docstrings"
        case "get_code_file_structure":
            path = args.get("path") if args else None
            if path:
                return f"Project file structure loaded successfully for {path}"
            return "Project file structure loaded successfully"
        case "GetNodeNeighboursFromNodeID":
            return "Fetched referenced code"
        case "WebpageContentExtractor":
            url = args.get("url") if args else None
            if url:
                return f"Information retrieved from {url}"
            return "Information retrieved from web"
        case "GitHubContentFetcher":
            repo_name = args.get("repo_name") if args else None
            issue_number = args.get("issue_number") if args else None
            is_pr = args.get("is_pull_request", False) if args else False
            if repo_name and issue_number:
                content_type = "PR" if is_pr else "Issue"
                return f"{content_type} #{issue_number} fetched from {repo_name}"
            return "File contents fetched from github"
        case "fetch_file":
            file_path = args.get("file_path") if args else None
            if file_path:
                result_suffix = (
                    _format_small_result(result) if _is_small_result(result) else ""
                )
                return f"File content fetched successfully: {file_path}{result_suffix}"
            result_suffix = (
                _format_small_result(result) if _is_small_result(result) else ""
            )
            return f"File content fetched successfully{result_suffix}"
        case "analyze_code_structure":
            file_path = args.get("file_path") if args else None
            if file_path:
                return f"Code structure analyzed successfully: {file_path}"
            return "Code structure analyzed successfully"
        case "WebSearchTool":
            query = args.get("query") if args else None
            if query:
                return f"Web search successful for: {query}"
            return "Web search successful"
        case "bash_command":
            if args:
                command = args.get("command", "")
                if command:
                    # Truncate long commands for display
                    display_cmd = (
                        command if len(command) <= 60 else command[:57] + "..."
                    )
                    result_suffix = (
                        _format_small_result(result) if _is_small_result(result) else ""
                    )
                    return f"Bash command executed successfully: {display_cmd}{result_suffix}"
            result_suffix = (
                _format_small_result(result) if _is_small_result(result) else ""
            )
            return f"Bash command executed successfully{result_suffix}"
        case "create_todo":
            title = args.get("title") if args else None
            if title:
                return f"Todo item created successfully: {title[:50]}"
            return "Todo item created successfully"
        case "update_todo_status":
            todo_id = args.get("todo_id") if args else None
            status = args.get("status") if args else None
            if todo_id and status:
                return f"Todo status updated successfully: {todo_id} -> {status}"
            elif todo_id:
                return f"Todo status updated successfully: {todo_id}"
            return "Todo status updated successfully"
        case "add_todo_note":
            todo_id = args.get("todo_id") if args else None
            if todo_id:
                return f"Todo note added successfully: {todo_id}"
            return "Todo note added successfully"
        case "get_todo":
            todo_id = args.get("todo_id") if args else None
            if todo_id:
                return f"Todo details retrieved successfully: {todo_id}"
            return "Todo details retrieved successfully"
        case "list_todos":
            status_filter = args.get("status_filter") if args else None
            if status_filter:
                return f"Todos listed successfully (filter: {status_filter})"
            return "Todos listed successfully"
        case "get_todo_summary":
            return "Todo summary generated successfully"
        case "add_requirements":
            return "Requirements document updated successfully"
        case "delete_requirements":
            return "Requirements document cleared successfully"
        case "get_requirements":
            return "Requirements document retrieved successfully"
        case "think":
            thought = args.get("thought") if args else None
            if thought and len(thought) <= 100:
                return f"Thoughts processed successfully: {thought[:100]}"
            return "Thoughts processed successfully"
        case "add_file_to_changes":
            file_path = args.get("file_path") if args else None
            if file_path:
                return f"File added to code changes successfully: {file_path}"
            return "File added to code changes successfully"
        case "update_file_in_changes":
            file_path = args.get("file_path") if args else None
            if file_path:
                return f"File updated in code changes successfully: {file_path}"
            return "File updated in code changes successfully"
        case "update_file_lines":
            file_path = args.get("file_path") if args else None
            if file_path:
                start_line = args.get("start_line") if args else None
                end_line = args.get("end_line") if args else None
                if start_line:
                    line_range = (
                        f"{start_line}-{end_line}" if end_line else str(start_line)
                    )
                    return f"File lines updated successfully: {file_path} (lines {line_range})"
                return f"File lines updated successfully: {file_path}"
            return "File lines updated successfully"
        case "replace_in_file":
            file_path = args.get("file_path") if args else None
            if file_path:
                pattern = args.get("pattern") if args else None
                if pattern:
                    return f"Pattern replacement completed successfully: {file_path} (pattern: {pattern[:50]})"
                return f"Pattern replacement completed successfully: {file_path}"
            return "Pattern replacement completed successfully"
        case "insert_lines":
            file_path = args.get("file_path") if args else None
            if file_path:
                line_number = args.get("line_number") if args else None
                if line_number:
                    return f"Lines inserted successfully: {file_path} (at line {line_number})"
                return f"Lines inserted successfully: {file_path}"
            return "Lines inserted successfully"
        case "delete_lines":
            file_path = args.get("file_path") if args else None
            if file_path:
                start_line = args.get("start_line") if args else None
                end_line = args.get("end_line") if args else None
                if start_line:
                    line_range = (
                        f"{start_line}-{end_line}" if end_line else str(start_line)
                    )
                    return (
                        f"Lines deleted successfully: {file_path} (lines {line_range})"
                    )
                return f"Lines deleted successfully: {file_path}"
            return "Lines deleted successfully"
        case "delete_file_in_changes":
            file_path = args.get("file_path") if args else None
            if file_path:
                return f"File marked for deletion successfully: {file_path}"
            return "File marked for deletion successfully"
        case "get_file_from_changes":
            file_path = args.get("file_path") if args else None
            if file_path:
                return f"File retrieved from code changes successfully: {file_path}"
            return "File retrieved from code changes successfully"
        case "list_files_in_changes":
            return "Files listed successfully"
        case "search_content_in_changes":
            return "Content search completed successfully"
        case "clear_file_from_changes":
            file_path = args.get("file_path") if args else None
            if file_path:
                return f"File cleared from code changes successfully: {file_path}"
            return "File cleared from code changes successfully"
        case "clear_all_changes":
            return "All code changes cleared successfully"
        case "get_changes_summary":
            return "Code changes summary retrieved successfully"
        case "export_changes":
            format_type = args.get("format") if args else None
            if format_type:
                return f"Code changes exported successfully ({format_type} format)"
            return "Code changes exported successfully"
        case "show_updated_file":
            if args:
                file_paths = args.get("file_paths")
                if file_paths:
                    if isinstance(file_paths, list):
                        return f"Updated file content displayed successfully: {', '.join(file_paths)}"
                    return f"Updated file content displayed successfully: {file_paths}"
            return "Updated file content displayed successfully"
        case "show_diff":
            return "Code diff displayed successfully"
        case tool_name if tool_name.startswith("delegate_to_"):
            # Handle delegation tools - extract agent type and return appropriate message
            agent_type = tool_name[12:]  # Remove "delegate_to_" prefix
            return get_delegation_response_message(agent_type)
        case "GetJiraIssue":
            issue_key = args.get("issue_key") if args else None
            if issue_key:
                return f"Jira issue details retrieved: {issue_key}"
            return "Jira issue details retrieved"
        case "SearchJiraIssues":
            jql = args.get("jql") if args else None
            if jql and len(jql) <= 80:
                return f"Jira issues search completed: {jql}"
            return "Jira issues search completed"
        case "CreateJiraIssue":
            project_key = args.get("project_key") if args else None
            summary = args.get("summary") if args else None
            if project_key and summary:
                return (
                    f"Jira issue created successfully: {project_key} - {summary[:50]}"
                )
            elif project_key:
                return f"Jira issue created successfully: {project_key}"
            return "Jira issue created successfully"
        case "UpdateJiraIssue":
            issue_key = args.get("issue_key") if args else None
            if issue_key:
                return f"Jira issue updated successfully: {issue_key}"
            return "Jira issue updated successfully"
        case "AddJiraComment":
            issue_key = args.get("issue_key") if args else None
            if issue_key:
                return f"Comment added to Jira issue: {issue_key}"
            return "Comment added to Jira issue"
        case "TransitionJiraIssue":
            issue_key = args.get("issue_key") if args else None
            transition = args.get("transition") if args else None
            if issue_key and transition:
                return f"Jira issue status changed: {issue_key} -> {transition}"
            elif issue_key:
                return f"Jira issue status changed: {issue_key}"
            return "Jira issue status changed"
        case "LinkJiraIssues":
            issue_key = args.get("issue_key") if args else None
            linked_issue_key = args.get("linked_issue_key") if args else None
            if issue_key and linked_issue_key:
                return f"Jira issues linked successfully: {issue_key} <-> {linked_issue_key}"
            return "Jira issues linked successfully"
        case "GetJiraProjects":
            return "Jira projects retrieved"
        case "GetJiraProjectDetails":
            project_key = args.get("project_key") if args else None
            if project_key:
                return f"Jira project details retrieved: {project_key}"
            return "Jira project details retrieved"
        case "GetJiraProjectUsers":
            project_key = args.get("project_key") if args else None
            query = args.get("query") if args else None
            if project_key and query:
                return f"Jira project users retrieved: {project_key} (query: {query})"
            elif project_key:
                return f"Jira project users retrieved: {project_key}"
            return "Jira project users retrieved"
        case "GetConfluenceSpaces":
            space_type = args.get("space_type") if args else None
            if space_type and space_type != "all":
                return f"Confluence spaces retrieved ({space_type})"
            return "Confluence spaces retrieved"
        case "GetConfluencePage":
            page_id = args.get("page_id") if args else None
            if page_id:
                return f"Confluence page retrieved: {page_id}"
            return "Confluence page retrieved"
        case "SearchConfluencePages":
            cql = args.get("cql") if args else None
            if cql and len(cql) <= 80:
                return f"Confluence pages search completed: {cql}"
            return "Confluence pages search completed"
        case "GetConfluenceSpacePages":
            space_id = args.get("space_id") if args else None
            if space_id:
                return f"Confluence space pages retrieved: {space_id}"
            return "Confluence space pages retrieved"
        case "CreateConfluencePage":
            space_id = args.get("space_id") if args else None
            title = args.get("title") if args else None
            if space_id and title:
                return (
                    f"Confluence page created successfully: {space_id} - {title[:50]}"
                )
            elif space_id:
                return f"Confluence page created successfully: {space_id}"
            return "Confluence page created successfully"
        case "UpdateConfluencePage":
            page_id = args.get("page_id") if args else None
            if page_id:
                return f"Confluence page updated successfully: {page_id}"
            return "Confluence page updated successfully"
        case "AddConfluenceComment":
            page_id = args.get("page_id") if args else None
            if page_id:
                return f"Comment added to Confluence page: {page_id}"
            return "Comment added to Confluence page"
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
            if path is not None and isinstance(path, str):
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
        case "bash_command":
            command = args.get("command")
            working_dir = args.get("working_directory")
            if command:
                dir_info = f" in directory '{working_dir}'" if working_dir else ""
                return f"-> executing command: {command}{dir_info}\n"
            return "-> executing bash command\n"
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
        case "add_requirements":
            requirements = args.get("requirements", "")
            preview = requirements[:150] if len(requirements) > 150 else requirements
            return f"-> updating requirements document: {preview}{'...' if len(requirements) > 150 else ''}\n"
        case "delete_requirements":
            return "-> clearing requirements document\n"
        case "get_requirements":
            return "-> retrieving requirements document\n"
        case "think":
            thought = args.get("thought", "")
            preview = thought[:200] if len(thought) > 200 else thought
            return f"-> processing thoughts: {preview}{'...' if len(thought) > 200 else ''}\n"
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
        case "show_updated_file":
            file_paths = args.get("file_paths", None)
            if file_paths:
                return f"-> displaying updated file content: {', '.join(file_paths) if isinstance(file_paths, list) else file_paths}\n"
            else:
                return "-> displaying all updated files\n"
        case "show_diff":
            return "-> displaying code diff\n"
        case tool_name if tool_name.startswith("delegate_to_"):
            # Handle delegation tools - extract agent type and return appropriate info
            agent_type = tool_name[12:]  # Remove "delegate_to_" prefix
            task_description = args.get("task_description", "")
            context = args.get("context", "")
            return get_delegation_info_content(agent_type, task_description, context)
        case "GetJiraIssue":
            issue_key = args.get("issue_key")
            if issue_key:
                return f"-> fetching issue {issue_key}"
            return ""
        case "SearchJiraIssues":
            jql = args.get("jql")
            if jql:
                return f"-> JQL: {jql}"
            return ""
        case "CreateJiraIssue":
            project_key = args.get("project_key")
            summary = args.get("summary")
            if project_key and summary:
                return f"-> creating issue in {project_key}: {summary}"
            return ""
        case "UpdateJiraIssue":
            issue_key = args.get("issue_key")
            if issue_key:
                return f"-> updating issue {issue_key}"
            return ""
        case "AddJiraComment":
            issue_key = args.get("issue_key")
            if issue_key:
                return f"-> adding comment to {issue_key}"
            return ""
        case "TransitionJiraIssue":
            issue_key = args.get("issue_key")
            transition = args.get("transition")
            if issue_key and transition:
                return f"-> moving {issue_key} to '{transition}'"
            return ""
        case "LinkJiraIssues":
            issue_key = args.get("issue_key")
            linked_issue_key = args.get("linked_issue_key")
            link_type = args.get("link_type")
            if issue_key and linked_issue_key:
                return f"-> linking {issue_key} {link_type or 'to'} {linked_issue_key}"
            return ""
        case "GetJiraProjects":
            return "-> fetching all accessible projects"
        case "GetJiraProjectDetails":
            project_key = args.get("project_key")
            if project_key:
                return f"-> fetching details for project {project_key}"
            return ""
        case "GetJiraProjectUsers":
            project_key = args.get("project_key")
            query = args.get("query")
            if project_key and query:
                return f"-> searching users in {project_key}: {query}"
            elif project_key:
                return f"-> fetching users in {project_key}"
            return ""
        case "GetConfluenceSpaces":
            space_type = args.get("space_type")
            limit = args.get("limit")
            if space_type and space_type != "all":
                return f"-> fetching {space_type} spaces (limit: {limit or 25})"
            return f"-> fetching all accessible spaces (limit: {limit or 25})"
        case "GetConfluencePage":
            page_id = args.get("page_id")
            if page_id:
                return f"-> fetching page {page_id}"
            return ""
        case "SearchConfluencePages":
            cql = args.get("cql")
            if cql:
                return f"-> CQL: {cql}"
            return ""
        case "GetConfluenceSpacePages":
            space_id = args.get("space_id")
            status = args.get("status")
            if space_id:
                status_text = (
                    f" ({status} pages)" if status and status != "current" else ""
                )
                return f"-> fetching pages in space {space_id}{status_text}"
            return ""
        case "CreateConfluencePage":
            space_id = args.get("space_id")
            title = args.get("title")
            if space_id and title:
                return f"-> creating page in space {space_id}: {title}"
            return ""
        case "UpdateConfluencePage":
            page_id = args.get("page_id")
            version_number = args.get("version_number")
            if page_id:
                return f"-> updating page {page_id} (version {version_number})"
            return ""
        case "AddConfluenceComment":
            page_id = args.get("page_id")
            parent_comment_id = args.get("parent_comment_id")
            if page_id:
                comment_type = "reply" if parent_comment_id else "comment"
                return f"-> adding {comment_type} to page {page_id}"
            return ""
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
```{str(node.get("code_content"))[: min(len(str(node.get("code_content"))), 600)] + " ..."}
```
"""
                            for node in content
                        ]
                    )
                except Exception:
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
                            text += f"{path}\n```{code_content[: min(len(code_content), 300)]}``` \n"
                        elif item.get("error") is not None:
                            text += f"Error: {item.get('error')} \n"
                return text
            return ""
        case "get_code_file_structure":
            if isinstance(content, str):
                return f"""-> fetched successfully
```
---------------
{content[: min(len(content), 600)]} ...
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
{[f''' {element.get("type")}: {element.get("name")} ''' for element in elements]}
"""
            return ""
        case "WebSearchTool":
            if isinstance(content, Dict):
                res = content.get("content")
                if isinstance(res, str):
                    return res[: min(len(res), 600)] + " ..."
            return ""
        case "bash_command":
            if isinstance(content, Dict):
                success = content.get("success", False)
                output = content.get("output", "")
                error = content.get("error", "")
                exit_code = content.get("exit_code", -1)

                if not success:
                    error_msg = f"Command failed with exit code {exit_code}"
                    if error:
                        error_msg += (
                            f"\n\nError output:\n```\n{error[: min(len(error), 500)]}"
                        )
                        if len(error) > 500:
                            error_msg += " ..."
                        error_msg += "\n```"
                    if output:
                        error_msg += f"\n\nStandard output:\n```\n{output[: min(len(output), 500)]}"
                        if len(output) > 500:
                            error_msg += " ..."
                        error_msg += "\n```"
                    return error_msg
                else:
                    result_msg = (
                        f"Command executed successfully (exit code: {exit_code})"
                    )
                    if output:
                        result_msg += (
                            f"\n\nOutput:\n```\n{output[: min(len(output), 1000)]}"
                        )
                        if len(output) > 1000:
                            result_msg += "\n... (output truncated)"
                        result_msg += "\n```"
                    if error:
                        result_msg += f"\n\nWarning/Error output:\n```\n{error[: min(len(error), 500)]}"
                        if len(error) > 500:
                            result_msg += " ..."
                        result_msg += "\n```"
                    return result_msg
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
            | "show_updated_file"
            | "show_diff"
        ):
            # For code changes manager tools, return the content directly
            if isinstance(content, str):
                return content
            return str(content)
        case "add_requirements" | "delete_requirements" | "get_requirements":
            # For requirement verification tools, return the content directly
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
    """Get user-friendly message when delegating to a subagent"""
    match agent_type:
        case "think_execute":
            return "🚀 Starting subagent with full tool access - streaming work in real-time..."
        case "jira":
            return "🎫 Starting Jira integration agent - handling Jira operations..."
        case "github":
            return "🐙 Starting GitHub integration agent - handling repository operations..."
        case "confluence":
            return "📄 Starting Confluence integration agent - handling documentation operations..."
        case "linear":
            return "📋 Starting Linear integration agent - handling issue management..."
        case _:
            return f"🚀 Starting {agent_type} subagent - streaming work in real-time..."


def get_delegation_response_message(agent_type: str) -> str:
    """Get user-friendly message when delegation completes"""
    match agent_type:
        case "think_execute":
            return "✅ Subagent completed - returning task result to supervisor"
        case "jira":
            return "✅ Jira agent completed - returning results to supervisor"
        case "github":
            return "✅ GitHub agent completed - returning results to supervisor"
        case "confluence":
            return "✅ Confluence agent completed - returning results to supervisor"
        case "linear":
            return "✅ Linear agent completed - returning results to supervisor"
        case _:
            return f"✅ {agent_type} subagent completed - returning task result to supervisor"


def get_delegation_info_content(
    agent_type: str, task_description: str, context: str = ""
) -> str:
    """Get detailed info about what the subagent will do"""
    # Get agent-specific prefix
    agent_prefix = ""
    match agent_type:
        case "think_execute":
            agent_prefix = "🤖 **General Subagent**"
        case "jira":
            agent_prefix = "🎫 **Jira Integration Agent**"
        case "github":
            agent_prefix = "🐙 **GitHub Integration Agent**"
        case "confluence":
            agent_prefix = "📄 **Confluence Integration Agent**"
        case "linear":
            agent_prefix = "📋 **Linear Integration Agent**"
        case _:
            agent_prefix = f"🤖 **{agent_type.title()} Agent**"

    info = f"{agent_prefix}\n\n**Task:**\n{task_description}"
    if context:
        # Truncate context preview for display
        context_preview = context[:500] + "..." if len(context) > 500 else context
        info += f"\n\n**Context Provided:**\n{context_preview}"
    else:
        info += "\n\n⚠️ *No context provided - subagent will need to gather information independently*"
    return info


def get_delegation_info_with_todo_context(
    agent_type: str, task_description: str, todo_id: str = "", context: str = ""
) -> str:
    """Get delegation info with todo context for better tracking"""
    base_info = get_delegation_info_content(agent_type, task_description, context)

    if todo_id:
        base_info += f"\n\n**Linked Todo:** {todo_id}"
        base_info += "\n*Subagent work tracked in todo system*"

    return base_info


def get_delegation_result_content(agent_type: str, result: str) -> str:
    """Get formatted result from subagent - returns the task result for supervisor coordination"""
    # Get agent-specific label
    agent_label = ""
    match agent_type:
        case "think_execute":
            agent_label = "General Subagent Result"
        case "jira":
            agent_label = "Jira Agent Result"
        case "github":
            agent_label = "GitHub Agent Result"
        case "confluence":
            agent_label = "Confluence Agent Result"
        case "linear":
            agent_label = "Linear Agent Result"
        case _:
            agent_label = f"{agent_type.title()} Agent Result"

    # Display the full task result without truncation - it can be detailed and include code snippets
    return f"**{agent_label}:**\n\n{result}"

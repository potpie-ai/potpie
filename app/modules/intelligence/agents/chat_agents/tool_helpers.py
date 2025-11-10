from typing import Any, Dict, List


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
        case "GetJiraIssue":
            return "Jira issue details retrieved"
        case "SearchJiraIssues":
            return "Jira issues search completed"
        case "CreateJiraIssue":
            return "Jira issue created successfully"
        case "UpdateJiraIssue":
            return "Jira issue updated successfully"
        case "AddJiraComment":
            return "Comment added to Jira issue"
        case "TransitionJiraIssue":
            return "Jira issue status changed"
        case "LinkJiraIssues":
            return "Jira issues linked successfully"
        case "GetJiraProjects":
            return "Jira projects retrieved"
        case "GetJiraProjectDetails":
            return "Jira project details retrieved"
        case "GetJiraProjectUsers":
            return "Jira project users retrieved"
        case "GetConfluenceSpaces":
            return "Confluence spaces retrieved"
        case "GetConfluencePage":
            return "Confluence page retrieved"
        case "SearchConfluencePages":
            return "Confluence pages search completed"
        case "GetConfluenceSpacePages":
            return "Confluence space pages retrieved"
        case "CreateConfluencePage":
            return "Confluence page created successfully"
        case "UpdateConfluencePage":
            return "Confluence page updated successfully"
        case "AddConfluenceComment":
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
        case "analyze_code_structure":
            if isinstance(content, Dict):
                if not content.get("success"):
                    return "Failed to analyze code structure"
                else:
                    return f"""
{[f''' {element.get("type")}: {element.get("name")} ''' for element in content.get("elements")]}
"""
        case "WebSearchTool":
            if isinstance(content, Dict):
                res = content.get("content")
                if isinstance(res, str):
                    return res[: min(len(res), 600)] + " ..."
            return ""
        case _:
            return ""

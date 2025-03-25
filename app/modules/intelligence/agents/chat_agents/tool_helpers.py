from typing import Any, Dict, List


def get_tool_run_message(tool_name: str):
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            return "retrieving code from probable nodes"
        case "GetCodeanddocstringFromNodeID":
            return "retrieving code for referenced node"
        case "Getcodechanges":
            return "fetching code changes from your repo"
        case "GetNodesfromTags":
            return "fetching nodes from tags"
        case "AskKnowledgeGraphQueries":
            return "traversing the knowledge graph"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return "fetching code and docstrings"
        case "get_code_file_structure":
            return "loading the dir structure"
        case "GetNodeNeighboursFromNodeID":
            return "identifying referenced code"
        case "WebpageContentExtractor":
            return "querying information from the web"
        case "GitHubContentFetcher":
            return "fetching content from github"
        case _:
            return "querying data"


def get_tool_response_message(tool_name: str):
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            return "code retrieved for probable nodes"
        case "GetCodeanddocstringFromNodeID":
            return "code retrieved for referenced node"
        case "Getcodechanges":
            return "code changes fetched successfully"
        case "GetNodesfromTags":
            return "fetched nodes from tags"
        case "AskKnowledgeGraphQueries":
            return "knowledge graph queries successful"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            return "fetched code and docstrings"
        case "get_code_file_structure":
            return "dir structure loaded successfully"
        case "GetNodeNeighboursFromNodeID":
            return "fetched referenced code"
        case "WebpageContentExtractor":
            return "information retrieved from web"
        case "GitHubContentFetcher":
            return "file contents fetched from github"
        case _:
            return "data queried successfully"


def get_tool_call_info_content(tool_name: str, args: Dict[str, Any]) -> str:
    match tool_name:
        case "GetCodeanddocstringFromProbableNodeName":
            node_names = args.get("probable_node_names")
            if isinstance(node_names, List):
                return "-> checking following nodes: \n" + "\n- ".join(node_names)
            return "-> checking probable nodes"
        case "Getcodechanges":
            return str(args)
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
            return "Code changes fetched successfully"
        case "GetNodesfromTags":
            return ""
        case "AskKnowledgeGraphQueries":
            return "\n docstrings retrieved for the queries"
        case "GetCodeanddocstringFromMultipleNodeIDs":
            if isinstance(content, Dict):
                res = content.values()
                return "\n".join(
                    [
                        f"{item.get('relative_file_path')}\n```\n{item.get('code_content')[:min(len(item.get('code_content')),300)]} ... \n```"
                        for item in res
                    ]
                )
            return str(content)
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
            return "Fetched referenced code"
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

```
{code}
```

"""
                return res[: min(len(res), 600)] + " ..."
            return str(content)
        case _:
            return ""

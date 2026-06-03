"""ToolService registrations for local search tools.

The individual local_search_tools modules expose functions that accept a single
Pydantic input model. ToolService, LangChain, and the registry discovery flow
work best with functions whose parameters match the model fields, so the small
wrappers below adapt field kwargs back into those input models.
"""

from typing import List, Optional

from langchain_core.tools import StructuredTool

from .execute_terminal_command_tool import (
    ExecuteTerminalCommandInput,
    execute_terminal_command_tool,
)
from .search_bash_tool import SearchBashInput, search_bash_tool
from .search_text_tool import SearchTextInput, search_text_tool
from .terminal_session_tools import (
    TerminalSessionOutputInput,
    TerminalSessionSignalInput,
    terminal_session_output_tool,
    terminal_session_signal_tool,
)


def _execute_terminal_command(
    command: str,
    working_directory=None,
    timeout: int = 30000,
    mode: str = "sync",
) -> str:
    return execute_terminal_command_tool(
        ExecuteTerminalCommandInput(
            command=command,
            working_directory=working_directory,
            timeout=timeout,
            mode=mode,
        )
    )


def _terminal_session_output(session_id: str, offset: int = 0) -> str:
    return terminal_session_output_tool(
        TerminalSessionOutputInput(session_id=session_id, offset=offset)
    )


def _terminal_session_signal(session_id: str, signal: str = "SIGINT") -> str:
    return terminal_session_signal_tool(
        TerminalSessionSignalInput(session_id=session_id, signal=signal)
    )


def _search_text(
    query: str,
    file_pattern: Optional[str] = None,
    case_sensitive: bool = False,
    use_regex: bool = False,
    max_results: Optional[int] = 100,
    use_bash: bool = False,
) -> str:
    return search_text_tool(
        SearchTextInput(
            query=query,
            file_pattern=file_pattern,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
            max_results=max_results,
            use_bash=use_bash,
        )
    )


def _search_bash(
    command: str,
    working_directory: Optional[str] = None,
    timeout: Optional[int] = 30000,
    project_id: Optional[str] = None,
) -> str:
    return search_bash_tool(
        SearchBashInput(
            command=command,
            working_directory=working_directory,
            timeout=timeout,
            project_id=project_id,
        )
    )


def create_local_search_tools() -> List[StructuredTool]:
    """Create local search tools for ToolService registration."""

    return [
        StructuredTool.from_function(
            func=_search_text,
            name="search_text",
            description=(
                "Search for a text or regex pattern across the workspace via the "
                "VS Code tunnel. Use for symbol names, error strings, file paths, "
                "and test fixtures. Set use_bash=true for faster large-codebase "
                "searches when a simple grep-style query is enough."
            ),
            args_schema=SearchTextInput,
        ),
        StructuredTool.from_function(
            func=_search_bash,
            name="search_bash",
            description=(
                "Run read-only bash search commands in the workspace via the VS "
                "Code tunnel, especially `rg -n`, `find`, `awk`, `sed`, `head`, "
                "`tail`, `ls`, `wc`, `sort`, and `uniq`. Use this for compound "
                "ripgrep queries, globbing, source/test fixture mining, and "
                "comparing helper implementations. Write operations and unsafe "
                "commands are blocked."
            ),
            args_schema=SearchBashInput,
        ),
        StructuredTool.from_function(
            func=_execute_terminal_command,
            name="execute_terminal_command",
            description=execute_terminal_command_tool.__doc__ or (
                "Execute a shell command on the user's local machine via the VS "
                "Code extension (Socket.IO workspace connection). Supports sync "
                "(default) and async modes. Use async mode for long-running "
                "commands; poll with terminal_session_output."
            ),
            args_schema=ExecuteTerminalCommandInput,
        ),
        StructuredTool.from_function(
            func=_terminal_session_output,
            name="terminal_session_output",
            description=terminal_session_output_tool.__doc__ or (
                "Get output from an async terminal session started by "
                "execute_terminal_command in async mode. Provide the session_id "
                "and an offset to stream incremental output."
            ),
            args_schema=TerminalSessionOutputInput,
        ),
        StructuredTool.from_function(
            func=_terminal_session_signal,
            name="terminal_session_signal",
            description=terminal_session_signal_tool.__doc__ or (
                "Send a signal (default SIGINT) to a running async terminal "
                "session to stop or interrupt the process."
            ),
            args_schema=TerminalSessionSignalInput,
        ),
    ]

"""Delegation utility functions for multi-agent system"""

import re
import hashlib
from enum import Enum

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class AgentType(Enum):
    """Types of specialized agents in the multi-agent system"""

    SUPERVISOR = "supervisor"
    THINK_EXECUTE = "think_execute"  # Generic Think and Execute Agent
    JIRA = "jira"  # Jira Integration Agent
    GITHUB = "github"  # GitHub Integration Agent
    CONFLUENCE = "confluence"  # Confluence Integration Agent
    LINEAR = "linear"  # Linear Integration Agent


def is_delegation_tool(tool_name: str) -> bool:
    """Check if a tool call is a delegation to a subagent"""
    # Support both old format (delegate_to_github) and new format (delegate_to_github_agent)
    return tool_name.startswith("delegate_to_")


def extract_agent_type_from_delegation_tool(tool_name: str) -> str:
    """Extract agent type from delegation tool name"""
    if tool_name.startswith("delegate_to_"):
        # Remove "delegate_to_" prefix
        agent_part = tool_name[12:]
        # Remove "_agent" suffix if present (new format)
        if agent_part.endswith("_agent"):
            return agent_part[:-6]  # Remove "_agent" suffix
        return agent_part
    return tool_name


def extract_task_result_from_response(response: str) -> str:
    """
    Extract the Task Result section from a subagent response.
    Returns the full Task Result without truncation - can include detailed content and code snippets.
    If no Task Result section is found, return the full response.
    Enhanced to handle error cases and provide better fallbacks.
    """
    if not response or not response.strip():
        logger.warning("Empty response provided to extract_task_result_from_response")
        return ""

    # Check for error indicators first
    error_indicators = [
        r"(?i)❌\s*error",
        r"(?i)⚠️\s*error",
        r"(?i)🚨\s*error",
        r"(?i)error\s*occurred",
        r"(?i)failed\s*to",
        r"(?i)exception",
        r"(?i)traceback",
    ]

    for error_pattern in error_indicators:
        if re.search(error_pattern, response):
            return response

    # Pattern to match Task Result section (case insensitive)
    # Updated patterns to better capture the end of result sections
    patterns = [
        r"(?i)#{1,4}\s*task\s*result[:\s]*\n(.*?)(?=\n#{1,4}\s*(?!task\s*result)\w+|\Z)",
        r"(?i)\*\*task\s*result[:\s]*\*\*\n(.*?)(?=\n\*\*(?!task\s*result)\w+|\Z)",
        r"(?i)task\s*result[:\s]*\n(.*?)(?=\n\w+:|\n#{1,4}\s*\w+|\Z)",
        r"(?i)## result[:\s]*\n(.*?)(?=\n#{1,4}\s*(?!result)\w+|\Z)",
        r"(?i)\*\*result[:\s]*\*\*\n(.*?)(?=\n\*\*(?!result)\w+|\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            if summary:
                return summary

    # If no Task Result section is found, look for conclusion or final sections
    conclusion_patterns = [
        r"(?i)#{1,4}\s*conclusion[:\s]*\n(.*?)(?=\n#{1,4}\s*\w+|\Z)",
        r"(?i)#{1,4}\s*summary[:\s]*\n(.*?)(?=\n#{1,4}\s*\w+|\Z)",
        r"(?i)#{1,4}\s*findings[:\s]*\n(.*?)(?=\n#{1,4}\s*\w+|\Z)",
    ]

    for pattern in conclusion_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            if summary:
                logger.warning(
                    f"No Task Result found, but found conclusion/summary section (length: {len(summary)} chars)"
                )
                return summary

    # If still no structured section found, return the last meaningful paragraphs without truncation
    lines = response.strip().split("\n")
    if len(lines) > 10:  # If response is substantial, try to get the final content
        # Look for the last substantial paragraphs
        meaningful_lines = []
        for line in reversed(lines):
            line = line.strip()
            if line:  # Keep all non-empty lines including code blocks
                meaningful_lines.append(line)
                if len(meaningful_lines) >= 10:  # Get more content for detailed summary
                    break

        if meaningful_lines:
            summary = "\n".join(reversed(meaningful_lines))
            logger.warning(
                f"No Task Summary section found, using last meaningful content as detailed summary (length: {len(summary)} chars)"
            )
            return summary

    # Final fallback: return full response without truncation
    logger.warning(
        f"No Task Summary section found, returning full response (length: {len(response)} chars)"
    )
    return response


def create_delegation_prompt(
    task_description: str,
    project_context: str,
    supervisor_context: str = "",
) -> str:
    """Create the delegation prompt for subagents.

    Subagents are ISOLATED - they receive ONLY:
    1. The task description
    2. Project context (IDs, names)
    3. Context explicitly provided by the supervisor

    They do NOT receive conversation history or previous tool results.
    """
    context_sections = []

    if project_context:
        context_sections.append(f"**PROJECT:**\n{project_context}")

    if supervisor_context and supervisor_context.strip():
        context_sections.append(f"**CONTEXT FROM SUPERVISOR:**\n{supervisor_context}")

    full_context = (
        "\n\n".join(context_sections)
        if context_sections
        else "No additional context provided."
    )

    return f"""You are a SUBAGENT executing a focused task. You have access to ALL tools to complete this work.

**YOUR TASK:**
{task_description}

**AVAILABLE CONTEXT:**
{full_context}

**IMPORTANT - YOU ARE ISOLATED:**
- You do NOT have access to the supervisor's conversation history
- You do NOT see previous tool calls or their results
- The context above is ALL the information provided to you
- Use your tools to gather any additional information you need

**EXECUTION APPROACH:**
1. Read the task and context carefully
2. Use tools to gather information (file reads, code searches, etc.)
3. Execute the task completely - be thorough
4. Stream your progress - the user sees your work in real-time
5. Make reasonable assumptions and state them

**TOOL USAGE:**
- Use fetch_file with with_line_numbers=true for precise file editing
- Use code changes tools (add_file_to_changes, update_file_lines, insert_lines, delete_lines) for modifications
- Use show_updated_file and show_diff to display your changes
- Use bash_command for running commands if needed

**OUTPUT:**
- Stream your thinking and work as you go
- End with "## Task Result" containing a concise, actionable summary
- The supervisor will use your Task Result for coordination

Now execute the task completely."""


def format_delegation_error(
    agent_type: AgentType,
    task_description: str,
    error_type: str,
    error_message: str,
    raw_response: str = "",
) -> str:
    """Format error responses for delegation failures"""
    if error_type == "no_result":
        return f"""
## Task Result

❌ **ERROR: No valid task result found**

**Agent Type:** {agent_type.value}
**Task:** {task_description}
**Issue:** The subagent did not provide a properly formatted Task Result section

**Raw Response (truncated):**
{raw_response[:500]}{'...' if len(raw_response) > 500 else ''}

**Recommendation:** The supervisor should retry the delegation with clearer instructions.
        """.strip()
    elif error_type == "empty_result":
        return f"""
## Task Result

❌ **ERROR: Empty or invalid result**

**Agent Type:** {agent_type.value}
**Task:** {task_description}
**Issue:** The subagent returned no output or an invalid response

**Recommendation:** The supervisor should retry the delegation or try a different approach.
        """.strip()
    else:  # exception
        return f"""
## Task Result

❌ **ERROR: Delegation failed**

**Agent Type:** {agent_type.value}
**Task:** {task_description}
**Error:** {error_message}
**Error Type:** {error_type}

**Recommendation:** The supervisor should investigate the error and retry with a different approach or agent.
        """.strip()


def create_delegation_cache_key(task_description: str, context: str) -> str:
    """Create a unique cache key for delegation result caching"""
    content = f"{task_description}::{context}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

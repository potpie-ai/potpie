"""
Requirement Verification Tool for Agent State Management

This tool allows agents to track requirements for complex tasks as a markdown-formatted
list, ensuring all output requirements are captured and verified before finalizing responses.
"""

from contextvars import ContextVar
from typing import List, Optional
from pydantic import BaseModel, Field


class RequirementManager:
    """Manages requirements as a single markdown string"""

    def __init__(self):
        self.requirements: str = ""  # Markdown-formatted list of requirements

    def set_requirements(self, requirements: str) -> bool:
        """Set the requirements (replaces existing)"""
        self.requirements = requirements.strip()
        return True

    def clear_requirements(self) -> bool:
        """Clear all requirements"""
        self.requirements = ""
        return True

    def get_requirements(self) -> str:
        """Get the current requirements"""
        return self.requirements


# Context variable for requirement manager - provides isolation per execution context
# This ensures parallel agent runs have separate, isolated state
_requirement_manager_ctx: ContextVar[Optional[RequirementManager]] = ContextVar(
    "_requirement_manager_ctx", default=None
)


def _get_requirement_manager() -> RequirementManager:
    """Get the current requirement manager for this execution context, creating a new one if needed.

    Uses ContextVar to ensure each async execution context (agent run) has its own isolated instance.
    This allows parallel agent runs to have separate state without interference.
    """
    manager = _requirement_manager_ctx.get()
    if manager is None:
        manager = RequirementManager()
        _requirement_manager_ctx.set(manager)
    return manager


def _reset_requirement_manager() -> None:
    """Reset the requirement manager for a new agent run - creates a completely fresh instance in this execution context.

    This ensures each agent run starts with a clean state, isolated from other parallel runs.
    """
    new_manager = RequirementManager()
    _requirement_manager_ctx.set(new_manager)


# Pydantic models for tool inputs
class AddRequirementsInput(BaseModel):
    requirements: str = Field(
        description="Requirements in markdown format. Use a markdown list (bullets or numbered) with each requirement as a separate item. Be specific and measurable. Example: '- Function handles null inputs gracefully\n- API returns 200 status code\n- Code changes limited to single file'"
    )


def add_requirements_tool(input_data: AddRequirementsInput) -> str:
    """Add or update requirements document"""
    try:
        req_manager = _get_requirement_manager()
        req_manager.set_requirements(input_data.requirements)

        result = "✅ Requirements document updated\n\n"
        result += "📋 **Current Requirements:**\n\n"
        if req_manager.get_requirements():
            result += req_manager.get_requirements()
        else:
            result += "No requirements added yet."

        # Automatically create a todo for verification
        try:
            from app.modules.intelligence.tools.todo_management_tool import (
                _get_todo_manager,
            )

            todo_manager = _get_todo_manager()
            # Check if verification todo already exists
            todos = todo_manager.list_todos()
            verification_todo_exists = any(
                "verify all requirements" in todo.get("title", "").lower()
                for todo in todos
            )

            if not verification_todo_exists:
                verification_todo_id = todo_manager.create_todo(
                    title="Verify all requirements",
                    description="Before finalizing your response, read the requirements document and verify that each requirement is met. Check each item systematically.",
                    priority="high",
                )
                result += f"\n\n📝 Created verification todo (ID: {verification_todo_id}) to verify all requirements before final response.\n"
        except ImportError:
            # Todo management not available, skip
            pass

        return result
    except Exception as e:
        return f"❌ Error updating requirements: {str(e)}"


def delete_requirements_tool() -> str:
    """Clear all requirements"""
    try:
        req_manager = _get_requirement_manager()
        req_manager.clear_requirements()
        return "✅ Requirements document cleared."
    except Exception as e:
        return f"❌ Error clearing requirements: {str(e)}"


def get_requirements_tool() -> str:
    """Get the current requirements document"""
    try:
        req_manager = _get_requirement_manager()
        requirements = req_manager.get_requirements()

        if not requirements:
            return "📋 **Requirements Document:** No requirements added yet.\n\nUse `add_requirements` to document output requirements at the start of complex tasks."

        result = "📋 **Requirements Document:**\n\n"
        result += requirements
        result += "\n\n**IMPORTANT:** Before finalizing your response, verify that each requirement above is met. Go through each item systematically and confirm it's satisfied."
        return result
    except Exception as e:
        return f"❌ Error retrieving requirements: {str(e)}"


# Create the structured tools
class SimpleTool:
    """Simple tool wrapper that mimics StructuredTool interface"""

    def __init__(self, name: str, description: str, func, args_schema):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema


def create_requirement_verification_tools() -> List[SimpleTool]:
    """Create all requirement verification tools"""

    tools = [
        SimpleTool(
            name="add_requirements",
            description="""CRITICAL: Document all output requirements at the START of complex tasks.

**WHEN TO USE (MANDATORY FOR COMPLEX OUTPUTS):**
- At the very beginning: Before starting any work, capture ALL requirements
- When user specifies deliverables: Document each requirement as a separate item
- For multi-step tasks: List all success criteria upfront
- When requirements are unclear: Ask clarifying questions, then document

**HOW TO USE:**
- Format as markdown list (bullets `-` or numbered `1.`)
- Each requirement should be a separate line/item
- Be SPECIFIC and MEASURABLE (avoid vague statements)
- Examples:
  ```
  - Function handles null inputs gracefully and returns appropriate error
  - API endpoint returns 200 status code for valid requests
  - Code changes are limited to single file (app/main.py)
  - All tests pass without modification
  - Response includes detailed error messages
  ```

**BEST PRACTICES:**
- Capture requirements FIRST, before implementation
- Be comprehensive - better to have too many than miss critical ones
- Update requirements if user clarifies or changes scope (use this tool again to replace)
- Each requirement should be verifiable (you can check if it's met)
- Focus on OUTPUT requirements, not implementation details
- Use clear, concise language - one requirement per line

**IMPORTANT:** This replaces any existing requirements. If updating, include ALL requirements (old + new) in the markdown list.""",
            func=add_requirements_tool,
            args_schema=AddRequirementsInput,
        ),
        SimpleTool(
            name="delete_requirements",
            description="""Clear the requirements document completely.

**WHEN TO USE:**
- Starting a completely new task with no requirements
- Requirements are no longer relevant
- User explicitly asks to clear requirements

**NOTE:** To update requirements, use `add_requirements` again with the updated list (it replaces existing). Only use `delete_requirements` when you want to remove all requirements entirely.""",
            func=delete_requirements_tool,
            args_schema=None,
        ),
        SimpleTool(
            name="get_requirements",
            description="""MANDATORY: Read the requirements document before finalizing your response.

**CRITICAL USAGE:**
- ALWAYS call this tool BEFORE giving your final response
- This is your quality gate - verify each requirement is met
- Go through the list systematically and check each item

**WHEN TO USE:**
- Before finalizing: Read requirements and verify each one is satisfied
- During work: Review requirements to ensure you're on track
- When stuck: Check requirements to refocus on what's needed
- After updates: Re-read to ensure changes didn't break anything

**VERIFICATION PROCESS:**
1. Call this tool to get the requirements document
2. Read through each requirement item
3. For each requirement, verify it's met in your work/output
4. If any requirement is NOT met:
   - Fix the issue before finalizing
   - Or clearly explain why it cannot be met
5. Only finalize when ALL requirements are verified

**BEST PRACTICES:**
- Call this as the LAST step before your final response
- Be honest - if a requirement isn't met, address it
- Consider delegating verification to a subagent for complex requirements
- Use this to catch issues before the user does
- Treat this as a mandatory checklist before completion""",
            func=get_requirements_tool,
            args_schema=None,
        ),
    ]

    return tools

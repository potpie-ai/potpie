"""
Execution variables constants.

This module defines all known execution variable names used throughout the workflow system.
Centralizing these constants helps maintain consistency and makes it easier to track
which variables are available in the execution context.
"""

from enum import Enum


class ExecutionVariables(str, Enum):
    """
    Known execution variable names used in workflow execution.

    These constants represent the variable names that can be set and accessed
    during workflow execution. Using these constants instead of string literals
    helps prevent typos and makes refactoring easier.
    """

    # GitHub trigger variables
    CURRENT_BRANCH = "CURRENT_BRANCH"
    CURRENT_REPO = "CURRENT_REPO"

    # Agent execution variables
    PROCESSED_BRANCH = "PROCESSED_BRANCH"
    AGENT_RESULT = "AGENT_RESULT"

    # Future variables can be added here as needed
    # USER_ID = "USER_ID"
    # REPO_NAME = "REPO_NAME"
    # EVENT_TYPE = "EVENT_TYPE"
    # TRIGGER_ACTION = "TRIGGER_ACTION"


# Convenience functions for working with execution variables
def get_variable_name(variable: ExecutionVariables) -> str:
    """Get the string value of an execution variable constant."""
    return variable.value


def is_known_variable(variable_name: str) -> bool:
    """Check if a variable name is a known execution variable."""
    return variable_name in [var.value for var in ExecutionVariables]


def get_all_known_variables() -> list[str]:
    """Get a list of all known execution variable names."""
    return [var.value for var in ExecutionVariables]

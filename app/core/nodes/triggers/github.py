"""
GitHub trigger node types and processor.

This module defines GitHub-specific trigger nodes for various GitHub events
like pull requests and issues.
"""

from typing import Literal, Union, Optional
from pydantic import BaseModel, Field
from app.core.executions.event import Event
from app.core.executions.state import NodeExecutionResult, NodeExecutionStatus
from app.core.nodes.base import NodeType, NodeCategory, NodeGroup, WorkflowNodeDetails
from app.core.nodes.triggers.base import TriggerNode
from app.core.nodes.data_models import (
    GithubIssueOpenedTriggerData,
    GithubPROpenedTriggerData,
    GithubPRClosedTriggerData,
    GithubPRReopenedTriggerData,
    GithubPRMergedTriggerData,
)


# GitHub trigger node classes
class GithubTriggerBase(TriggerNode):
    """Base class for GitHub triggers."""

    group: NodeGroup = NodeGroup.GITHUB


class GithubPROpenedTrigger(GithubTriggerBase):
    """GitHub PR opened trigger node."""

    type: NodeType = NodeType.TRIGGER_GITHUB_PR_OPENED
    data: GithubPROpenedTriggerData


class GithubPRClosedTrigger(GithubTriggerBase):
    """GitHub PR closed trigger node."""

    type: NodeType = NodeType.TRIGGER_GITHUB_PR_CLOSED
    data: GithubPRClosedTriggerData


class GithubPRReopenedTrigger(GithubTriggerBase):
    """GitHub PR reopened trigger node."""

    type: NodeType = NodeType.TRIGGER_GITHUB_PR_REOPENED
    data: GithubPRReopenedTriggerData


class GithubPRMergedTrigger(GithubTriggerBase):
    """GitHub PR merged trigger node."""

    type: NodeType = NodeType.TRIGGER_GITHUB_PR_MERGED
    data: GithubPRMergedTriggerData


class GithubIssueOpenedTrigger(GithubTriggerBase):
    """GitHub issue opened trigger node."""

    type: NodeType = NodeType.TRIGGER_GITHUB_ISSUE_OPENED
    data: GithubIssueOpenedTriggerData


GithubTrigger = Union[
    GithubPROpenedTrigger,
    GithubPRClosedTrigger,
    GithubPRReopenedTrigger,
    GithubPRMergedTrigger,
    GithubIssueOpenedTrigger,
]


# Node definitions for UI
ALL_GITHUB_TRIGGERS = [
    WorkflowNodeDetails(
        unique_identifier="trigger-github-pr-opened",
        name="GitHub PR Opened",
        description="Triggers when a new pull request is opened",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.GITHUB,
        type=NodeType.TRIGGER_GITHUB_PR_OPENED,
        icon="git-pull-request",
        color="#28a745",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "repo_name": {"type": "string", "title": "Repository Name"},
                "hash": {"type": "string", "title": "Trigger Hash"},
            },
            "required": ["repo_name", "hash"],
        },
    ),
    WorkflowNodeDetails(
        unique_identifier="trigger-github-pr-closed",
        name="GitHub PR Closed",
        description="Triggers when a pull request is closed",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.GITHUB,
        type=NodeType.TRIGGER_GITHUB_PR_CLOSED,
        icon="git-pull-request",
        color="#dc3545",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "repo_name": {"type": "string", "title": "Repository Name"},
                "hash": {"type": "string", "title": "Trigger Hash"},
            },
            "required": ["repo_name", "hash"],
        },
    ),
    WorkflowNodeDetails(
        unique_identifier="trigger-github-pr-reopened",
        name="GitHub PR Reopened",
        description="Triggers when a pull request is reopened",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.GITHUB,
        type=NodeType.TRIGGER_GITHUB_PR_REOPENED,
        icon="git-pull-request",
        color="#ffc107",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "repo_name": {"type": "string", "title": "Repository Name"},
                "hash": {"type": "string", "title": "Trigger Hash"},
            },
            "required": ["repo_name", "hash"],
        },
    ),
    WorkflowNodeDetails(
        unique_identifier="trigger-github-pr-merged",
        name="GitHub PR Merged",
        description="Triggers when a pull request is merged",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.GITHUB,
        type=NodeType.TRIGGER_GITHUB_PR_MERGED,
        icon="git-merge",
        color="#007bff",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "repo_name": {"type": "string", "title": "Repository Name"},
                "hash": {"type": "string", "title": "Trigger Hash"},
            },
            "required": ["repo_name", "hash"],
        },
    ),
    WorkflowNodeDetails(
        unique_identifier="trigger-github-issue-opened",
        name="GitHub Issue Opened",
        description="Triggers when a new issue is opened",
        category=NodeCategory.TRIGGER,
        group=NodeGroup.GITHUB,
        type=NodeType.TRIGGER_GITHUB_ISSUE_OPENED,
        icon="alert-circle",
        color="#fd7e14",
        inputs=[],
        outputs=["output"],
        config_schema={
            "type": "object",
            "properties": {
                "repo_name": {"type": "string", "title": "Repository Name"},
                "hash": {"type": "string", "title": "Trigger Hash"},
            },
            "required": ["repo_name", "hash"],
        },
    ),
]

"""
Trigger node types and registry.

This module exports all trigger node types and their definitions.
"""

from .base import TriggerNode, TriggerNodeType
from .github import (
    GithubTrigger,
    GithubPROpenedTrigger,
    GithubPRClosedTrigger,
    GithubPRReopenedTrigger,
    GithubPRMergedTrigger,
    GithubIssueOpenedTrigger,
    ALL_GITHUB_TRIGGERS,
)
from .linear import (
    LinearTrigger,
    LinearIssueCreatedTrigger,
    ALL_LINEAR_TRIGGERS,
)
from .sentry import (
    SentryTrigger,
    SentryIssueCreatedTrigger,
    ALL_SENTRY_TRIGGERS,
)
from .webhook import (
    WebhookTrigger,
    ALL_WEBHOOK_TRIGGERS,
)

# All trigger definitions
ALL_TRIGGERS = [*ALL_GITHUB_TRIGGERS, *ALL_LINEAR_TRIGGERS, *ALL_SENTRY_TRIGGERS, *ALL_WEBHOOK_TRIGGERS]

__all__ = [
    "TriggerNode",
    "TriggerNodeType",
    "GithubTrigger",
    "GithubPROpenedTrigger",
    "GithubPRClosedTrigger",
    "GithubPRReopenedTrigger",
    "GithubPRMergedTrigger",
    "GithubIssueOpenedTrigger",
    "LinearTrigger",
    "LinearIssueCreatedTrigger",
    "SentryTrigger",
    "SentryIssueCreatedTrigger",
    "WebhookTrigger",
    "ALL_GITHUB_TRIGGERS",
    "ALL_LINEAR_TRIGGERS",
    "ALL_SENTRY_TRIGGERS",
    "ALL_WEBHOOK_TRIGGERS",
    "ALL_TRIGGERS",
]

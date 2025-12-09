"""
Trigger executors for handling trigger execution logic.

This module contains executor classes that handle the execution logic for different trigger types,
separating execution logic from data models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.core.executions.event import Event
from app.core.executions.state import NodeExecutionResult, NodeExecutionStatus
from app.core.nodes.triggers.base import TriggerNode


class TriggerExecutor(ABC):
    """Base class for trigger executors."""

    @abstractmethod
    async def execute(self, trigger: TriggerNode, event: Event) -> NodeExecutionResult:
        """
        Execute a trigger with an event.

        Args:
            trigger: The trigger node to execute
            event: The event that triggered the workflow

        Returns:
            NodeExecutionResult: Result of trigger execution
        """
        pass

    @abstractmethod
    def can_execute(self, trigger: TriggerNode) -> bool:
        """
        Check if this executor can handle the given trigger.

        Args:
            trigger: The trigger node to check

        Returns:
            bool: True if this executor can handle the trigger
        """
        pass

    def get_next_nodes(
        self,
        result: NodeExecutionResult,
        adjacency_list: Dict[str, List[str]],
        trigger_id: str,
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the trigger execution result.

        This method allows each trigger type to control the flow of execution.
        The default implementation returns all adjacent nodes, but subclasses
        can override this to implement custom routing logic.

        Args:
            result: The result of this trigger's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes
            trigger_id: The ID of the trigger node

        Returns:
            List of node IDs that should be queued for execution next
        """
        # Default behavior: return all adjacent nodes
        # This will be overridden by specific trigger executors
        return adjacency_list.get(trigger_id, [])


class GitHubTriggerExecutor(TriggerExecutor):
    """Executor for GitHub triggers."""

    def can_execute(self, trigger: TriggerNode) -> bool:
        """Check if this executor can handle GitHub triggers."""
        from app.core.nodes.triggers.github import (
            GithubPROpenedTrigger,
            GithubPRClosedTrigger,
            GithubPRReopenedTrigger,
            GithubPRMergedTrigger,
            GithubIssueOpenedTrigger,
        )

        return isinstance(
            trigger,
            (
                GithubPROpenedTrigger,
                GithubPRClosedTrigger,
                GithubPRReopenedTrigger,
                GithubPRMergedTrigger,
                GithubIssueOpenedTrigger,
            ),
        )

    async def execute(self, trigger: TriggerNode, event: Event) -> NodeExecutionResult:
        """
        Execute a GitHub trigger with an event.

        Args:
            trigger: The GitHub trigger node to execute
            event: The event that triggered the workflow

        Returns:
            NodeExecutionResult: Result of trigger execution
        """
        from app.core.nodes.triggers.github_utils import (
            get_github_event_type,
            get_github_action,
            matches_github_trigger,
            get_issue_details,
            get_pr_details,
            get_repo_details,
            get_current_branch,
            get_current_repo,
        )
        from app.core.execution_variables import ExecutionVariables

        # Extract GitHub event information
        event_type = get_github_event_type(event.headers or {})
        action = get_github_action(event.payload)

        # Check if this event matches the trigger type
        if not matches_github_trigger(event_type, action, trigger.type, event.payload):
            return NodeExecutionResult(
                status=NodeExecutionStatus.SKIPPED,
                output=f"Event skipped: Event type '{event_type}' with action '{action}' does not match trigger type '{trigger.type}'",
            )

        # Process the trigger based on its type
        from app.core.nodes.triggers.github import (
            GithubPROpenedTrigger,
            GithubPRClosedTrigger,
            GithubPRReopenedTrigger,
            GithubPRMergedTrigger,
            GithubIssueOpenedTrigger,
        )

        if isinstance(trigger, GithubPROpenedTrigger):
            current_branch = get_current_branch(event.payload)
            current_repo = get_current_repo(event.payload)
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=f"""
                A new PR was OPENED for the repo:
                
                Repo Details:
                {get_repo_details(event.payload)}
                
                PR Details:
                {get_pr_details(event.payload)}
                """,
                execution_variables={
                    ExecutionVariables.CURRENT_BRANCH: current_branch,
                    ExecutionVariables.CURRENT_REPO: current_repo,
                },
            )
        elif isinstance(trigger, GithubPRClosedTrigger):
            current_branch = get_current_branch(event.payload)
            current_repo = get_current_repo(event.payload)
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=f"""
                A new PR was CLOSED for the repo:
                
                Repo Details:
                {get_repo_details(event.payload)}
                
                PR Details:
                {get_pr_details(event.payload)}
                """,
                execution_variables={
                    ExecutionVariables.CURRENT_BRANCH: current_branch,
                    ExecutionVariables.CURRENT_REPO: current_repo,
                },
            )
        elif isinstance(trigger, GithubPRReopenedTrigger):
            current_branch = get_current_branch(event.payload)
            current_repo = get_current_repo(event.payload)
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=f"""
                A new PR was REOPENED for the repo:
                
                Repo Details:
                {get_repo_details(event.payload)}
                
                PR Details:
                {get_pr_details(event.payload)}
                """,
                execution_variables={
                    ExecutionVariables.CURRENT_BRANCH: current_branch,
                    ExecutionVariables.CURRENT_REPO: current_repo,
                },
            )
        elif isinstance(trigger, GithubPRMergedTrigger):
            current_branch = get_current_branch(event.payload)
            current_repo = get_current_repo(event.payload)
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=f"""
                A new PR was MERGED for the repo:
                
                Repo Details:
                {get_repo_details(event.payload)}
                
                PR Details:
                {get_pr_details(event.payload)}
                """,
                execution_variables={
                    ExecutionVariables.CURRENT_BRANCH: current_branch,
                    ExecutionVariables.CURRENT_REPO: current_repo,
                },
            )
        elif isinstance(trigger, GithubIssueOpenedTrigger):
            current_branch = get_current_branch(event.payload)
            current_repo = get_current_repo(event.payload)
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=f"""
                A new ISSUE was OPENED for the repo:
                
                Repo Details:
                {get_repo_details(event.payload)}
                
                Issue Details:
                {get_issue_details(event.payload)}
                """,
                execution_variables={
                    ExecutionVariables.CURRENT_BRANCH: current_branch,
                    ExecutionVariables.CURRENT_REPO: current_repo,
                },
            )
        else:
            raise ValueError(f"Unknown GitHub trigger: {trigger}")

    def get_next_nodes(
        self,
        result: NodeExecutionResult,
        adjacency_list: Dict[str, List[str]],
        trigger_id: str,
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the GitHub trigger execution result.

        For GitHub triggers, we only continue to next nodes if the trigger was successful
        and the event matched the trigger type. If the event was skipped (e.g., wrong event type),
        we don't queue any next nodes.

        Args:
            result: The result of this trigger's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes
            trigger_id: The ID of the trigger node

        Returns:
            List of node IDs that should be queued for execution next
        """
        # Check if the trigger was skipped (event didn't match trigger type)
        if (
            result.status.value == "completed"
            and isinstance(result.output, str)
            and "Event skipped" in result.output
        ):
            # Event was skipped, don't queue any next nodes
            return []

        # For successful triggers, continue to all adjacent nodes
        return adjacency_list.get(trigger_id, [])


class LinearTriggerExecutor(TriggerExecutor):
    """Executor for Linear triggers."""

    def can_execute(self, trigger: TriggerNode) -> bool:
        """Check if this executor can handle Linear triggers."""
        from app.core.nodes.triggers.linear import LinearIssueCreatedTrigger

        return isinstance(trigger, LinearIssueCreatedTrigger)

    async def execute(self, trigger: TriggerNode, event: Event) -> NodeExecutionResult:
        """
        Execute a Linear trigger with an event.

        Args:
            trigger: The Linear trigger node to execute
            event: The event that triggered the workflow

        Returns:
            NodeExecutionResult: Result of trigger execution
        """
        from app.core.nodes.triggers.linear import LinearIssueCreatedTrigger

        if isinstance(trigger, LinearIssueCreatedTrigger):
            # Extract issue details from the event payload
            # The webhook data is nested under service_result.webhook_data
            # Log the event in a detailed way for debugging and traceability
            import logging

            logger = logging.getLogger("linear.trigger.executor")

            logger.info("=" * 80)
            logger.info(f"🔔 Linear Trigger Event Received - Event ID: {event.id}")
            logger.info(f"📅 Event Timestamp: {event.time}")
            logger.info(f"🔗 Event Source: {event.source}")
            logger.info(f"📋 Event Source Type: {event.source_type}")
            logger.info(f"🎯 Trigger Node ID: {getattr(trigger, 'id', 'unknown')}")
            logger.info(f"📝 Trigger Node Type: {getattr(trigger, 'type', 'unknown')}")
            payload_str = str(event.payload)
            if len(payload_str) > 500:
                logger.info(f"📦 Event Payload (truncated): {payload_str[:500]}...")
            else:
                logger.info(f"📦 Event Payload: {payload_str}")
            if event.headers:
                logger.info(f"📋 Event Headers: {event.headers}")
            else:
                logger.info("📋 Event Headers: None")
            logger.info("=" * 80)
            # The webhook data is directly in the event payload
            webhook_data = event.payload
            issue_data = webhook_data.get("data", {})
            actor = webhook_data.get("actor", {})

            # Log extracted data for debugging
            logger.info(f"🔍 Extracted webhook_data keys: {list(webhook_data.keys())}")
            logger.info(f"🔍 Extracted issue_data keys: {list(issue_data.keys())}")
            logger.info(f"🔍 Extracted actor keys: {list(actor.keys())}")

            # Check if this is a "create" event - only execute workflow for issue creation
            action = webhook_data.get("action", "")
            logger.info(f"🎯 Linear event action: {action}")

            if action != "create":
                logger.info(
                    f"⏭️ Skipping workflow execution - event action '{action}' is not 'create'"
                )
                return NodeExecutionResult(
                    status=NodeExecutionStatus.SKIPPED,
                    output=f"Linear event action '{action}' does not match trigger condition (create). Workflow execution skipped.",
                )

            # Extract team and project information
            team = issue_data.get("team", {})
            project = issue_data.get("project", {})
            state = issue_data.get("state", {})

            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=f"""
                A new Linear issue was CREATED:
                
                Issue Details:
                Issue ID: {issue_data.get('id', 'Unknown')}
                Issue Number: {issue_data.get('identifier', 'Unknown')}
                Issue Title: {issue_data.get('title', 'Unknown')}
                Issue URL: {issue_data.get('url', webhook_data.get('url', 'Unknown'))}
                Created By: {actor.get('name', 'Unknown')} ({actor.get('email', 'Unknown')})
                Team: {team.get('name', 'Unknown')} ({team.get('key', 'Unknown')})
                Project: {project.get('name', 'No Project')}
                Priority: {issue_data.get('priorityLabel', 'Unknown')}
                State: {state.get('name', 'Unknown')}
                Description: {issue_data.get('description', 'No description')}
                Created At: {issue_data.get('createdAt', 'Unknown')}
                Updated At: {issue_data.get('updatedAt', 'Unknown')}
                """,
            )
        else:
            raise ValueError(f"Unknown Linear trigger: {trigger}")

    def get_next_nodes(
        self,
        result: NodeExecutionResult,
        adjacency_list: Dict[str, List[str]],
        trigger_id: str,
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the Linear trigger execution result.

        For Linear triggers, we only continue to next nodes if the trigger was successful
        and the event matched the trigger type.

        Args:
            result: The result of this trigger's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes
            trigger_id: The ID of the trigger node

        Returns:
            List of node IDs that should be queued for execution next
        """
        # Check if the trigger was skipped (event didn't match trigger type)
        if (
            result.status.value == "completed"
            and isinstance(result.output, str)
            and "Event skipped" in result.output
        ):
            # Event was skipped, don't queue any next nodes
            return []

        # For successful triggers, continue to all adjacent nodes
        return adjacency_list.get(trigger_id, [])


class SentryTriggerExecutor(TriggerExecutor):
    """Executor for Sentry triggers."""

    def can_execute(self, trigger: TriggerNode) -> bool:
        """Check if this executor can handle Sentry triggers."""
        from app.core.nodes.triggers.sentry import SentryIssueCreatedTrigger

        return isinstance(trigger, SentryIssueCreatedTrigger)

    async def execute(self, trigger: TriggerNode, event: Event) -> NodeExecutionResult:
        """
        Execute a Sentry trigger with an event.

        Args:
            trigger: The Sentry trigger node to execute
            event: The event that triggered the workflow

        Returns:
            NodeExecutionResult: Result of trigger execution
        """
        from app.core.nodes.triggers.sentry import SentryIssueCreatedTrigger

        if isinstance(trigger, SentryIssueCreatedTrigger):
            # Extract issue details from the event payload
            issue_data = event.payload.get("data", {})
            issue = issue_data.get("issue", {})

            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=f"""
                A new Sentry issue was CREATED:
                
                Issue Details:
                Issue ID: {issue.get('id', 'Unknown')}
                Issue Title: {issue.get('title', 'Unknown')}
                Issue URL: {issue.get('url', 'Unknown')}
                Project: {issue.get('project', {}).get('name', 'Unknown')}
                Environment: {issue.get('environment', 'Unknown')}
                Level: {issue.get('level', 'Unknown')}
                Culprit: {issue.get('culprit', 'Unknown')}
                """,
            )
        else:
            raise ValueError(f"Unknown Sentry trigger: {trigger}")

    def get_next_nodes(
        self,
        result: NodeExecutionResult,
        adjacency_list: Dict[str, List[str]],
        trigger_id: str,
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the Sentry trigger execution result.

        For Sentry triggers, we only continue to next nodes if the trigger was successful
        and the event matched the trigger type.

        Args:
            result: The result of this trigger's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes
            trigger_id: The ID of the trigger node

        Returns:
            List of node IDs that should be queued for execution next
        """
        # Check if the trigger was skipped (event didn't match trigger type)
        if (
            result.status.value == "completed"
            and isinstance(result.output, str)
            and "Event skipped" in result.output
        ):
            # Event was skipped, don't queue any next nodes
            return []

        # For successful triggers, continue to all adjacent nodes
        return adjacency_list.get(trigger_id, [])


class WebhookTriggerExecutor(TriggerExecutor):
    """Executor for webhook triggers."""

    def can_execute(self, trigger: TriggerNode) -> bool:
        """Check if this executor can handle webhook triggers."""
        from app.core.nodes.triggers.webhook import WebhookTrigger

        return isinstance(trigger, WebhookTrigger)

    async def execute(self, trigger: TriggerNode, event: Event) -> NodeExecutionResult:
        """
        Execute a webhook trigger with an event.

        Args:
            trigger: The webhook trigger node to execute
            event: The event that triggered the workflow

        Returns:
            NodeExecutionResult: Result of trigger execution
        """
        try:
            # For webhook triggers, we pass through the event data
            # The event data contains the webhook payload and headers
            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output={
                    "payload": event.payload,
                    "headers": event.headers,
                    "source": event.source,
                },
            )
        except Exception as e:
            return NodeExecutionResult(
                status=NodeExecutionStatus.FAILED,
                output=str(e),
            )

    def get_next_nodes(
        self,
        result: NodeExecutionResult,
        adjacency_list: Dict[str, List[str]],
        trigger_id: str,
    ) -> List[str]:
        """
        Determine which nodes should be queued next based on the webhook trigger execution result.

        Args:
            result: The result of this trigger's execution
            adjacency_list: The workflow's adjacency list mapping node IDs to their next nodes
            trigger_id: The ID of the trigger node

        Returns:
            List of node IDs that should be queued for execution next
        """
        # For webhook triggers, we always proceed to the next nodes
        return adjacency_list.get(trigger_id, [])


class TriggerExecutorRegistry:
    """Registry for trigger executors."""

    def __init__(self):
        self._executors: List[TriggerExecutor] = []

    def register_executor(self, executor: TriggerExecutor):
        """Register a trigger executor."""
        self._executors.append(executor)

    def get_executor(self, trigger: TriggerNode) -> TriggerExecutor:
        """Get the appropriate executor for a trigger node."""
        for executor in self._executors:
            if executor.can_execute(trigger):
                return executor
        raise ValueError(f"No executor found for trigger type: {trigger.type}")

    def register_default_executors(self):
        """Register all default trigger executors."""
        self.register_executor(GitHubTriggerExecutor())
        self.register_executor(LinearTriggerExecutor())
        self.register_executor(SentryTriggerExecutor())
        self.register_executor(WebhookTriggerExecutor())


# Global registry instance
_global_executor_registry: Optional[TriggerExecutorRegistry] = None


def get_trigger_executor_registry() -> TriggerExecutorRegistry:
    """Get the global trigger executor registry."""
    global _global_executor_registry
    if _global_executor_registry is None:
        _global_executor_registry = TriggerExecutorRegistry()
        _global_executor_registry.register_default_executors()
    return _global_executor_registry


def set_trigger_executor_registry(registry: TriggerExecutorRegistry):
    """Set the global trigger executor registry (for testing)."""
    global _global_executor_registry
    _global_executor_registry = registry

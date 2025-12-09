"""
Typed data models for node configuration.

This module defines strongly-typed data models for all node types,
replacing the generic Dict[str, Any] data field with proper validation.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from app.core.nodes.base import NodeType


class BaseNodeData(BaseModel):
    """Base class for all node data models."""

    model_config = {
        "extra": "ignore",  # Allow extra fields for backward compatibility
        "populate_by_name": True,
        "use_enum_values": True,
    }

    def model_dump(self, **kwargs):
        """Override model_dump to always use snake_case field names."""
        # Since all field names are already in snake_case, just return the default dump
        return super().model_dump(**kwargs)

    def validate_data(self) -> tuple[bool, List[str]]:
        """
        Validate the node data and return validation result.

        Returns:
            tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        # Basic validation - check if required fields are present
        for field_name, field_info in self.model_fields.items():
            if field_info.is_required and getattr(self, field_name) is None:
                errors.append(f"Required field '{field_name}' is missing or null")

        return len(errors) == 0, errors


# ============================================================================
# TRIGGER NODE DATA MODELS
# ============================================================================


class GithubTriggerData(BaseNodeData):
    """Base data model for GitHub triggers."""

    repo_name: Optional[str] = Field(
        None, description="Repository name (e.g., 'owner/repo')"
    )
    hash: Optional[str] = Field(None, description="Trigger hash for webhook URL")

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate GitHub trigger data."""
        is_valid, errors = super().validate_data()

        # Validate repo_name format if provided
        if self.repo_name and "/" not in self.repo_name:
            errors.append("Repository name must be in format 'owner/repo'")

        # Validate hash if provided
        if self.hash and len(self.hash) < 8:
            errors.append("Hash must be at least 8 characters long")

        return len(errors) == 0, errors


class GithubIssueOpenedTriggerData(GithubTriggerData):
    """Data model for GitHub issue opened trigger."""

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate GitHub issue opened trigger data."""
        return super().validate_data()


class GithubPROpenedTriggerData(GithubTriggerData):
    """Data model for GitHub PR opened trigger."""

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate GitHub PR opened trigger data."""
        return super().validate_data()


class GithubPRClosedTriggerData(GithubTriggerData):
    """Data model for GitHub PR closed trigger."""

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate GitHub PR closed trigger data."""
        return super().validate_data()


class GithubPRReopenedTriggerData(GithubTriggerData):
    """Data model for GitHub PR reopened trigger."""

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate GitHub PR reopened trigger data."""
        return super().validate_data()


class GithubPRMergedTriggerData(GithubTriggerData):
    """Data model for GitHub PR merged trigger."""

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate GitHub PR merged trigger data."""
        return super().validate_data()


class LinearTriggerData(BaseNodeData):
    """Base data model for Linear triggers."""

    integration_id: Optional[str] = Field(None, description="Linear integration ID")
    unique_identifier: Optional[str] = Field(
        None,
        description="Unique identifier for the Linear integration (organizationId)",
    )
    organization_id: Optional[str] = Field(
        None, description="Linear organization ID (used as trigger hash)"
    )

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate Linear trigger data."""
        is_valid, errors = super().validate_data()

        # Validate integration_id if provided
        if self.integration_id and len(self.integration_id) < 8:
            errors.append("Integration ID must be at least 8 characters long")

        # Validate unique_identifier if provided
        if self.unique_identifier and len(self.unique_identifier) < 1:
            errors.append("Unique identifier cannot be empty")

        # Validate organization_id if provided
        if self.organization_id and len(self.organization_id) < 1:
            errors.append("Organization ID cannot be empty")

        return len(errors) == 0, errors


class LinearIssueCreatedTriggerData(LinearTriggerData):
    """Data model for Linear issue created trigger."""

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate Linear issue created trigger data."""
        return super().validate_data()


class SentryTriggerData(BaseNodeData):
    """Base data model for Sentry triggers."""

    hash: Optional[str] = Field(None, description="Trigger hash for webhook URL")

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate Sentry trigger data."""
        is_valid, errors = super().validate_data()

        # Validate hash if provided
        if self.hash and len(self.hash) < 8:
            errors.append("Hash must be at least 8 characters long")

        return len(errors) == 0, errors


class SentryIssueCreatedTriggerData(SentryTriggerData):
    """Data model for Sentry issue created trigger."""

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate Sentry issue created trigger data."""
        return super().validate_data()


class WebhookTriggerData(BaseNodeData):
    """Data model for webhook triggers."""

    hash: Optional[str] = Field(None, description="Trigger hash for webhook URL")

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate webhook trigger data."""
        is_valid, errors = super().validate_data()

        # Validate hash if provided
        if self.hash and len(self.hash) < 8:
            errors.append("Hash must be at least 8 characters long")

        return len(errors) == 0, errors


# ============================================================================
# AGENT NODE DATA MODELS
# ============================================================================


class CustomAgentData(BaseNodeData):
    """Data model for custom agent configuration."""

    agent_id: Optional[str] = Field(None, description="Agent ID")
    task: Optional[str] = Field(None, description="Task description for the agent")
    repo_name: Optional[str] = Field(
        None, description="Repository name (optional if use_current_repo is True)"
    )
    branch_name: Optional[str] = Field(
        None, description="Branch name (optional if use_current_branch is True)"
    )
    use_current_repo: Optional[bool] = Field(
        default=False,
        description="Use current repository instead of specified repository",
    )
    use_current_branch: Optional[bool] = Field(
        default=False, description="Use current branch instead of specified branch"
    )

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate custom agent data."""
        is_valid, errors = super().validate_data()

        # Validate required fields
        if not self.agent_id:
            errors.append("Agent ID is required")

        if not self.task:
            errors.append("Task description is required")

        # Validate repository configuration
        if not self.use_current_repo and not self.repo_name:
            errors.append(
                "Repository name is required when not using current repository"
            )

        # Validate branch configuration
        if not self.use_current_branch and not self.branch_name:
            errors.append("Branch name is required when not using current branch")

        # Validate repo name format if provided
        if self.repo_name and "/" not in self.repo_name:
            errors.append("Repository name must be in format 'owner/repo'")

        return len(errors) == 0, errors


class ActionAgentData(BaseNodeData):
    """Data model for action agent configuration."""

    mcp_servers: Optional[List[str]] = Field(
        None, description="List of MCP server URLs"
    )
    name: Optional[str] = Field(None, description="Agent name")
    task: Optional[str] = Field(None, description="Task description for the agent")

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate action agent data."""
        is_valid, errors = super().validate_data()

        # Validate required fields
        if not self.mcp_servers or len(self.mcp_servers) == 0:
            errors.append("At least one MCP server is required")

        if not self.name:
            errors.append("Agent name is required")

        if not self.task:
            errors.append("Task description is required")

        # Validate MCP server URLs
        if self.mcp_servers:
            from app.utils.url_validation import validate_url, URLValidationError
            
            for i, server_url in enumerate(self.mcp_servers):
                if not server_url or not server_url.strip():
                    errors.append(f"MCP server at index {i} cannot be empty")
                else:
                    try:
                        # Validate URL and reject dangerous schemes like data: URIs
                        validate_url(server_url, allowed_schemes={"http", "https"})
                    except URLValidationError as e:
                        errors.append(f"MCP server at index {i} is invalid: {str(e)}")

        return len(errors) == 0, errors


# ============================================================================
# FLOW CONTROL NODE DATA MODELS
# ============================================================================


class ConditionalNodeData(BaseNodeData):
    """Data model for conditional flow control node."""

    condition: Optional[str] = Field(
        None, description="Condition expression to evaluate"
    )
    use_llm: Optional[bool] = Field(
        default=True,
        description="Whether to use LLM for condition evaluation (always True - LLM evaluation is always used)",
    )
    llm_context: Optional[str] = Field(
        None, description="Additional context for LLM evaluation"
    )

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate conditional node data."""
        is_valid, errors = super().validate_data()

        # Validate required fields
        if not self.condition:
            errors.append("Condition expression is required")

        # Validate condition length
        if self.condition and len(self.condition.strip()) < 3:
            errors.append("Condition expression must be at least 3 characters long")

        return len(errors) == 0, errors


class CollectNodeData(BaseNodeData):
    """Data model for collect flow control node."""

    collection_key: Optional[str] = Field(None, description="Key to collect data under")
    timeout_seconds: Optional[int] = Field(
        default=300, description="Collection timeout in seconds"
    )

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate collect node data."""
        is_valid, errors = super().validate_data()

        # Validate required fields
        if not self.collection_key:
            errors.append("Collection key is required")

        # Validate timeout
        if self.timeout_seconds is not None:
            if self.timeout_seconds < 1:
                errors.append("Timeout must be at least 1 second")
            elif self.timeout_seconds > 86400:  # 24 hours
                errors.append("Timeout cannot exceed 24 hours (86400 seconds)")

        return len(errors) == 0, errors


class SelectorNodeData(BaseNodeData):
    """Data model for selector flow control node."""

    selection_criteria: Optional[str] = Field(
        None, description="Selection criteria expression"
    )

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate selector node data."""
        is_valid, errors = super().validate_data()

        # Validate required fields
        if not self.selection_criteria:
            errors.append("Selection criteria is required")

        # Validate criteria length
        if self.selection_criteria and len(self.selection_criteria.strip()) < 3:
            errors.append("Selection criteria must be at least 3 characters long")

        return len(errors) == 0, errors


# ============================================================================
# MANUAL STEP NODE DATA MODELS
# ============================================================================


class ApprovalNodeData(BaseNodeData):
    """Data model for approval manual step node."""

    approvers: Optional[List[str]] = Field(
        None, description="List of approver user IDs"
    )
    approval_message: Optional[str] = Field(
        None, description="Message to show to approvers"
    )
    timeout_hours: Optional[int] = Field(
        default=24, description="Approval timeout in hours"
    )
    channel: Optional[str] = Field(
        default="app", description="Notification channel: 'email' or 'app' (web UI)"
    )

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate approval node data."""
        is_valid, errors = super().validate_data()

        # Validate required fields
        if not self.approvers or len(self.approvers) == 0:
            errors.append("At least one approver is required")

        # Validate approvers list
        if self.approvers:
            for i, approver in enumerate(self.approvers):
                if not approver or not approver.strip():
                    errors.append(f"Approver at index {i} cannot be empty")

        # Validate timeout
        if self.timeout_hours is not None:
            if self.timeout_hours < 1:
                errors.append("Timeout must be at least 1 hour")
            elif self.timeout_hours > 168:  # 1 week
                errors.append("Timeout cannot exceed 1 week (168 hours)")

        return len(errors) == 0, errors


class InputNodeData(BaseNodeData):
    """Data model for input manual step node."""

    input_fields: Optional[List[Dict[str, Any]]] = Field(
        None, description="Input field definitions"
    )
    assignee: Optional[str] = Field(
        None, description="User ID assigned to provide input"
    )
    timeout_hours: Optional[int] = Field(
        default=24, description="Input timeout in hours"
    )
    channel: Optional[str] = Field(
        default="app", description="Notification channel: 'email' or 'app' (web UI)"
    )

    def validate_data(self) -> tuple[bool, List[str]]:
        """Validate input node data."""
        is_valid, errors = super().validate_data()

        # Validate required fields
        if not self.assignee:
            errors.append("Assignee is required")

        if not self.input_fields or len(self.input_fields) == 0:
            errors.append("At least one input field is required")

        # Validate input fields
        if self.input_fields:
            for i, field in enumerate(self.input_fields):
                if not isinstance(field, dict):
                    errors.append(f"Input field at index {i} must be a dictionary")
                    continue

                if "name" not in field or not field["name"]:
                    errors.append(
                        f"Input field at index {i} must have a 'name' property"
                    )

                if "type" not in field or not field["type"]:
                    errors.append(
                        f"Input field at index {i} must have a 'type' property"
                    )

        # Validate timeout
        if self.timeout_hours is not None:
            if self.timeout_hours < 1:
                errors.append("Timeout must be at least 1 hour")
            elif self.timeout_hours > 168:  # 1 week
                errors.append("Timeout cannot exceed 1 week (168 hours)")

        return len(errors) == 0, errors


# ============================================================================
# NODE DATA REGISTRY
# ============================================================================

# Registry mapping node types to their data models
NODE_DATA_MODELS: Dict[NodeType, type[BaseNodeData]] = {
    # GitHub Triggers
    NodeType.TRIGGER_GITHUB_ISSUE_OPENED: GithubIssueOpenedTriggerData,
    NodeType.TRIGGER_GITHUB_PR_OPENED: GithubPROpenedTriggerData,
    NodeType.TRIGGER_GITHUB_PR_CLOSED: GithubPRClosedTriggerData,
    NodeType.TRIGGER_GITHUB_PR_REOPENED: GithubPRReopenedTriggerData,
    NodeType.TRIGGER_GITHUB_PR_MERGED: GithubPRMergedTriggerData,
    # Linear Triggers
    NodeType.TRIGGER_LINEAR_ISSUE_CREATED: LinearIssueCreatedTriggerData,
    # Sentry Triggers
    NodeType.TRIGGER_SENTRY_ISSUE_CREATED: SentryIssueCreatedTriggerData,
    # Webhook Triggers
    NodeType.TRIGGER_WEBHOOK: WebhookTriggerData,
    # Agents
    NodeType.CUSTOM_AGENT: CustomAgentData,
    NodeType.ACTION_AGENT: ActionAgentData,
    # Flow Control
    NodeType.FLOW_CONTROL_CONDITIONAL: ConditionalNodeData,
    NodeType.FLOW_CONTROL_COLLECT: CollectNodeData,
    NodeType.FLOW_CONTROL_SELECTOR: SelectorNodeData,
    # Manual Steps
    NodeType.MANUAL_STEP_APPROVAL: ApprovalNodeData,
    NodeType.MANUAL_STEP_INPUT: InputNodeData,
}


def get_data_model_for_node_type(node_type: NodeType) -> type[BaseNodeData]:
    """Get the data model class for a given node type."""
    if node_type not in NODE_DATA_MODELS:
        raise ValueError(f"No data model found for node type: {node_type}")
    return NODE_DATA_MODELS[node_type]


def create_node_data(node_type: NodeType, data: Dict[str, Any]) -> BaseNodeData:
    """Create a typed node data object from raw data."""
    data_model_class = get_data_model_for_node_type(node_type)
    return data_model_class(**data)


# Union type for all node data models
NodeData = Union[
    # Triggers
    GithubIssueOpenedTriggerData,
    GithubPROpenedTriggerData,
    GithubPRClosedTriggerData,
    GithubPRReopenedTriggerData,
    GithubPRMergedTriggerData,
    LinearIssueCreatedTriggerData,
    SentryIssueCreatedTriggerData,
    WebhookTriggerData,
    # Agents
    CustomAgentData,
    ActionAgentData,
    # Flow Control
    ConditionalNodeData,
    CollectNodeData,
    SelectorNodeData,
    # Manual Steps
    ApprovalNodeData,
    InputNodeData,
]

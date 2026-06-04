from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --- Manifest ---


class Manifest(BaseModel):
    version: int = 1
    profile: str
    steps: list[str] = Field(
        default_factory=lambda: [
            "user",
            "projects",
            "custom_agents",
            "conversations",
        ]
    )


# --- User ---


class AuthProviderSeed(BaseModel):
    provider_type: str
    provider_uid: str
    provider_data: dict[str, Any] | None = None
    is_primary: bool = False


class UserSeed(BaseModel):
    email: str | None = None
    display_name: str
    email_verified: bool = True
    uid: str | None = None
    provider_info: dict[str, Any] = Field(default_factory=dict)
    provider_username: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    auth_providers: list[AuthProviderSeed] = Field(default_factory=list)


# --- Projects ---


class ProjectSeed(BaseModel):
    id: str
    repo_name: str
    branch_name: str
    status: str = "ready"
    commit_id: str | None = None
    repo_path: str | None = None
    properties_base64: str | None = None


class ProjectsFile(BaseModel):
    projects: list[ProjectSeed]


# --- Custom agents ---


class CustomAgentSeed(BaseModel):
    id: str
    role: str | None = None
    goal: str | None = None
    backstory: str | None = None
    system_prompt: str | None = None
    tasks: list[Any] | dict[str, Any] = Field(default_factory=list)
    deployment_url: str | None = None
    deployment_status: str | None = "STOPPED"
    visibility: str = "private"


class CustomAgentsFile(BaseModel):
    agents: list[CustomAgentSeed]


# --- Conversations / messages ---


MessageTypeLiteral = Literal["HUMAN", "AI_GENERATED", "SYSTEM_GENERATED"]
MessageStatusLiteral = Literal["ACTIVE", "ARCHIVED", "DELETED"]


class MessageSeed(BaseModel):
    id: str
    content: str
    type: MessageTypeLiteral
    status: MessageStatusLiteral = "ACTIVE"
    sender_id: str | None = None
    citations: str | None = None
    has_attachments: bool = False
    tool_calls: list[Any] | dict[str, Any] | None = None
    thinking: str | None = None
    created_at: datetime | None = None


class ConversationSeed(BaseModel):
    id: str
    title: str
    status: str = "active"
    project_ids: list[str] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    shared_with_emails: list[str] | None = None
    visibility: str | None = "private"
    messages: list[MessageSeed] = Field(default_factory=list)


class ConversationsFile(BaseModel):
    conversations: list[ConversationSeed]

    @field_validator("conversations")
    @classmethod
    def non_empty_ids(cls, v: list[ConversationSeed]) -> list[ConversationSeed]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate conversation id")
        return v

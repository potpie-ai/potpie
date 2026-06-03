"""Agent harness recommendations for Potpie skills."""

from __future__ import annotations

VALID_AGENTS = ("default", "codex", "claude", "cursor")
AGENT_ALIASES = {"claude-code": "claude"}

RECOMMENDED_BY_AGENT: dict[str, list[str]] = {
    "default": ["potpie-agent-context", "potpie-pot-scope"],
    "cursor": [
        "potpie-agent-context",
        "potpie-pot-scope",
        "potpie-cli",
        "potpie-cli-troubleshooting",
    ],
    "claude": ["potpie-agent-context", "potpie-pot-scope", "potpie-cli"],
    "codex": ["potpie-agent-context", "potpie-pot-scope", "potpie-cli"],
}


class InvalidAgentError(ValueError):
    """Raised when the CLI receives an unknown agent harness."""


def normalize_agent(agent: str | None) -> str:
    value = (agent or "default").strip().lower()
    value = AGENT_ALIASES.get(value, value)
    if value not in VALID_AGENTS:
        raise InvalidAgentError(
            f"Unknown agent {agent!r}. Choose one of: {', '.join(VALID_AGENTS)}."
        )
    return value


def recommended_skill_ids(agent: str | None) -> list[str]:
    return list(RECOMMENDED_BY_AGENT[normalize_agent(agent)])


def llm_guidance(agent: str) -> dict[str, object]:
    normalized = normalize_agent(agent)
    read_for_cli = ["potpie-cli"]
    if normalized == "cursor":
        read_for_cli.append("potpie-cli-troubleshooting")
    return {
        "agent": normalized,
        "readBeforeWork": ["potpie-agent-context"],
        "readForCliTasks": read_for_cli,
        "readForPotScopeTasks": ["potpie-pot-scope"],
        "doNotEditInstalledSkillsDirectly": True,
    }

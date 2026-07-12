"""Root-owned skill catalog, resources, validation, and harness targets."""

from __future__ import annotations

from pathlib import Path

from potpie.skills.contracts import (
    SkillInfo,
    SkillNudge,
    SkillOperationResult,
    SkillStatus,
)
from potpie.skills.resource_provider import (
    ROOT_TEMPLATE_RESOURCES,
    TemplateResourceProvider,
)
from potpie.skills.service import DefaultSkillManager
from potpie.skills.targets import (
    ClaudeAgentTarget,
    CodexAgentTarget,
    CursorAgentTarget,
    OpenCodeAgentTarget,
)


def create_skill_service(
    *,
    data_dir: Path | None = None,
    template_resources: TemplateResourceProvider = ROOT_TEMPLATE_RESOURCES,
) -> DefaultSkillManager:
    target_kwargs = {"home": data_dir} if data_dir is not None else {}
    return DefaultSkillManager(
        targets={
            "claude": ClaudeAgentTarget(
                template_resources=template_resources, **target_kwargs
            ),
            "codex": CodexAgentTarget(
                template_resources=template_resources, **target_kwargs
            ),
            "cursor": CursorAgentTarget(
                template_resources=template_resources, **target_kwargs
            ),
            "opencode": OpenCodeAgentTarget(
                template_resources=template_resources, **target_kwargs
            ),
        },
        template_resources=template_resources,
    )


__all__ = [
    "DefaultSkillManager",
    "SkillInfo",
    "SkillNudge",
    "SkillOperationResult",
    "SkillStatus",
    "create_skill_service",
]

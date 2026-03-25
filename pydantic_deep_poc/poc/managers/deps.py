"""DeepAgentDeps extended with shared PoC RunContext (CCM, todos, requirements)."""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any

from pydantic_ai_backends import BackendProtocol, StateBackend
from pydantic_deep.deps import DeepAgentDeps

from poc.managers.run_context import RunContext
from poc.tools.shell_policy import ShellPolicy


@dataclass
class PoCDeepDeps(DeepAgentDeps):
    """Shared `poc_run` across supervisor and subagents for CCM + todo + requirements."""

    poc_run: RunContext = field(default_factory=RunContext)
    shell_policy: ShellPolicy = field(default=ShellPolicy.UNRESTRICTED)
    deps_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def clone_for_subagent(self, max_depth: int = 0) -> "PoCDeepDeps":
        p = DeepAgentDeps.clone_for_subagent(self, max_depth)
        return PoCDeepDeps(
            backend=p.backend,
            files=p.files,
            todos=p.todos,
            subagents=p.subagents,
            uploads=p.uploads,
            ask_user=p.ask_user,
            context_middleware=p.context_middleware,
            share_todos=p.share_todos,
            poc_run=self.poc_run,
            shell_policy=self.shell_policy,
            deps_id=str(uuid.uuid4()),
        )

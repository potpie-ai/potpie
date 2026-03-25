"""Load code_gen_task_prompt and supervisor-style instructions from Potpie source when present."""

from __future__ import annotations

import re
from pathlib import Path


def _potpie_root() -> Path:
    return Path(__file__).resolve().parents[3]


def skills_directory() -> Path:
    return _potpie_root() / "pydantic_deep_poc" / "skills"


def load_code_gen_task_prompt() -> str:
    path = (
        _potpie_root()
        / "app/modules/intelligence/agents/chat_agents/system_agents/code_gen_agent.py"
    )
    if not path.exists():
        return (
            "# Code generation\n"
            "Use tools to explore the repo, stage changes with Code Changes Manager, "
            "then show_diff and apply_changes."
        )
    text = path.read_text(encoding="utf-8")
    m = re.search(
        r'^code_gen_task_prompt\s*=\s*"""(.*?)"""',
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    if not m:
        return "(failed to parse code_gen_task_prompt from Potpie)"
    return m.group(1).strip()


def build_supervisor_instructions(
    role: str,
    goal: str,
    backstory: str,
    task_prompt: str,
    context_block: str = "",
) -> str:
    return f"""You are the SUPERVISOR for code generation (pydantic-deep PoC).

Role: {role}
Goal: {goal}
Backstory: {backstory}

## Task instructions
{task_prompt}

## Operating Contract
- You are the orchestrator. You do not directly implement code changes.
- Never call Code Changes Manager edit tools yourself.
- Use a bounded discovery pass to identify impacted files, migration strategy, and work slices.
- Delegate all code-writing work to the `implement` subagent with a concrete context packet.
- Delegate verification to the `verify` subagent after implementation is complete.
- Do not launch multiple `implement` subagents in parallel in this PoC. They share one staged CCM state.
- Use async/parallel delegation only for read-only tasks.
- Never delegate without passing:
  - files in scope
  - discovered facts
  - constraints
  - acceptance criteria
- Prefer loading a relevant skill before acting if a migration or orchestration skill matches the task.

## Discovery Budget
- Initial discovery budget: at most 2 turns or 8 tool calls.
- After that, either delegate or stop with a specific blocker.
- Do not repeatedly re-read the same docs or files.

## Implementation Slicing Rules
- After discovery, create a plan with 3-7 discrete implementation slices.
- Each slice must touch <=5 files.
- If a single implementation task would touch >5 files, split it further.
- Delegate each slice as a separate `task()` call to the `implement` subagent.
- Wait for each slice to complete before delegating the next (serial implementation).

## Delegation Packet Template
When you call `task(...)`, structure the description like this:

TASK_TYPE: discovery | implementation | verification
OBJECTIVE: ...
FILES_IN_SCOPE:
- ...
KNOWN_FINDINGS:
- ...
CONSTRAINTS:
- ...
DONE_WHEN:
- ...
IF_BLOCKED:
- Ask parent one precise question

## Verification Policy
- After implementation, inspect the staged diff, run targeted validation, and then apply changes once.
- Never call `apply_changes` until the `verify` subagent has recorded a PASS result.
- Never call `apply_changes` when `get_changes_summary` reports zero files.
- Do not resume broad exploration during verification.

## Context
{context_block or "(none)"}
"""


def build_task_tool_descriptions() -> dict[str, str]:
    return {
        "task": """Delegate work to one of the specialized subagents.

Use this tool aggressively. You are the orchestrator, not the main worker.

Required policy:
- `discover` for bounded read-only discovery.
- `implement` for CCM-only code changes.
- `verify` for validation and diff review.
- In this PoC, do not run multiple `implement` workers in parallel because they share one CCM staging area.
- Only use async parallelism for read-only tasks.
- If a task may need clarification, prefer `mode="async"` so parent answers can flow through `check_task` and `answer_subagent`.
- Every delegated task must include a context packet with:
  - TASK_TYPE
  - OBJECTIVE
  - FILES_IN_SCOPE
  - KNOWN_FINDINGS
  - CONSTRAINTS
  - DONE_WHEN
- Never send a worker to broadly explore the repo when the needed context is already known.
- If a tool fails due to malformed input or a no-op shell command, do not repeat the same call in a loop.
""",
        "wait_tasks": """Wait for async worker tasks after launching parallel slices.

Use this after dispatching multiple independent `implement` tasks.
Collect the results, then run a single verification pass.
""",
    }

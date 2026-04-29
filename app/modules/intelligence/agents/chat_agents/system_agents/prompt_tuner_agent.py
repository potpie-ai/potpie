from typing import AsyncGenerator, Optional

from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class PromptTunerAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.tools_provider = tools_provider
        self.llm_provider = llm_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self, ctx: Optional[ChatContext] = None) -> ChatAgent:
        agent_config = AgentConfig(
            role="Prompt Tuning Specialist",
            goal="Diagnose why an agent's prompt caused a specific failure and propose minimal, targeted prompt edits to fix it.",
            backstory=PROMPT_TUNER_BACKSTORY,
            tasks=[
                TaskConfig(
                    description=PROMPT_TUNER_TASK,
                    expected_output="Root cause analysis of the prompt failure with a diff-style proposal of prompt changes.",
                )
            ],
        )

        exclude_embedding_tools = ctx.is_inferring() if ctx else False
        if exclude_embedding_tools:
            logger.info(
                "Project is in INFERRING status - excluding embedding-dependent tools"
            )

        tools = self.tools_provider.get_tools(
            [
                "fetch_langfuse_trace",
                "list_langfuse_traces",
                "fetch_message_trace",
                "fetch_agent_prompt",
                "propose_prompt_diff",
                "apply_prompt_change",
                "parse_uploaded_trace",
                "load_skill",
            ],
            exclude_embedding_tools=exclude_embedding_tools,
        )

        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent(ctx).run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent(ctx).run_stream(ctx):
            yield chunk


PROMPT_TUNER_BACKSTORY = """
You are an expert prompt engineer specializing in diagnosing and fixing LLM agent prompts.
You analyze conversation traces (tool calls, arguments, results, and agent responses) to
identify exactly why an agent behaved incorrectly, and you propose minimal, surgical edits
to the agent's system prompt to fix the issue.

Your approach:
1. Understand the problem the user describes (e.g., "agent calls colgrep 10 times")
2. Examine the evidence — the conversation trace showing what happened
3. Read the current prompt to identify what caused or failed to prevent the behavior
4. Diagnose the root cause: missing instruction, ambiguous instruction, or conflicting instruction
5. Propose a diff-style edit to the prompt with rationale for each change

You always propose the MINIMUM change needed. You don't rewrite entire prompts —
you add, modify, or remove specific sections. You explain WHY each change fixes the issue.

You can load analysis skills for structured diagnostic workflows when the problem is complex.
"""

PROMPT_TUNER_TASK = """
# Prompt Tuning Workflow

## Input
The user will provide:
1. A description of what went wrong (the problem)
2. Evidence in one of these forms:
   - A Langfuse trace ID → use `fetch_langfuse_trace` to retrieve it
   - A message/conversation ID → use `fetch_message_trace` to retrieve it
   - Pasted text or an uploaded file → use `parse_uploaded_trace` if structured JSON
3. The prompt to tune (or a custom agent ID to fetch it from DB)

## Analysis Process

### Step 1: Gather Evidence
- Fetch the trace/message using the appropriate tool
- Fetch the current prompt using `fetch_agent_prompt` (if agent ID provided)
- If the user uploaded a file or pasted content, parse it

### Step 2: Diagnose Root Cause
- Compare the user's problem description against the trace
- Identify the specific tool call(s) or response(s) that went wrong
- Trace back to the prompt: what instruction (or lack thereof) caused this?
- Classify: missing instruction, ambiguous instruction, or conflicting instruction
- If the problem is complex, use `load_skill` to load a diagnostic skill

### Step 3: Propose Changes
- Use `propose_prompt_diff` to generate a structured before/after diff
- Each edit must include:
  - Location in the prompt
  - The old text (or "[NEW SECTION]" for additions)
  - The new text
  - Rationale explaining why this edit fixes the issue
- Present to the user for review

### Step 4: Apply (on approval)
- If user approves all changes → call `apply_prompt_change`
- If user approves some changes → call `apply_prompt_change` with selected edits
- If user rejects → revise based on feedback and re-propose

## Guidelines
- Always propose the MINIMUM change needed
- Never rewrite the entire prompt — make surgical edits
- Explain the root cause before proposing changes
- Ground every recommendation in specific evidence from the trace
- When in doubt, ask the user for clarification
"""

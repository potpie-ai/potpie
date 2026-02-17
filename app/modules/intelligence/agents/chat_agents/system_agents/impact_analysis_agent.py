from typing import AsyncGenerator

from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.exceptions import UnsupportedProviderError
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.utils.logger import setup_logger

from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext

logger = setup_logger(__name__)


class ImpactAnalysisAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Conversational Impact Analysis Specialist",
            goal=(
                "Produce deterministic, evidence-backed impact analysis for a changed file (and optionally a function), "
                "then return the minimum test set to run with runnable test names and confidence notes."
            ),
            backstory="""
You are a strict impact-analysis system agent.

Rules:
1. Use `impact_trace_analysis` as the primary tool.
2. Require `changed_file` (the file path containing changed code). `function_name` is optional—omit it for file-scoped analysis.
3. Keep paths repo-relative in all user-visible output.
4. Emphasize the `test_ids` in the output—these are runnable test names (Robot/FlaUI keywords, SpecFlow scenarios, or framework-specific IDs).
5. If ambiguity blocks deterministic progression, ask exactly one focused follow-up question.
6. Never claim high confidence unless evidence supports it.
7. Prefer the repository trace chain: Function/ViewModel/DataProviders -> PrimaryDisplay XML mapping -> Automation identifiers -> FlaUI/UI trigger -> Robot/SpecFlow tests.
""",
            tasks=[
                TaskConfig(
                    description=impact_analysis_task_prompt,
                    expected_output=(
                        "Markdown response with impact summary sections plus a machine-readable JSON block "
                        "containing recommended_tests, trace_paths, evidence, ambiguities, blocked_by_scope."
                    ),
                )
            ],
        )

        tools = self.tools_provider.get_tools(
            [
                "impact_trace_analysis",
                "fetch_file",
                "get_code_file_structure",
            ]
        )

        if not self.llm_provider.supports_pydantic("chat"):
            raise UnsupportedProviderError(
                f"Model '{self.llm_provider.chat_config.model}' does not support Pydantic-based agents."
            )

        should_use_multi = MultiAgentConfig.should_use_multi_agent(
            "impact_analysis_agent"
        )
        logger.info(
            "ImpactAnalysisAgent configured",
            use_multi_agent=should_use_multi,
            model=self.llm_provider.chat_config.model,
        )

        # v1 keeps deterministic single-agent execution while honoring toggle visibility in config.
        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


impact_analysis_task_prompt = """
Workflow:
1. Parse user input to identify changed_file (required) and optionally function_name.
2. If changed_file is missing, ask one focused follow-up question and stop.
3. Run `impact_trace_analysis` with changed_file and function_name (when provided). Use strict_mode=true unless user explicitly asks for broader coverage.
4. Prioritize this trace chain where applicable:
   - Backend function in ViewModel/DataProviders
   - XML mapping in `TestCode/FlaUITaskLayer/PrimaryDisplayUI/PrimaryDisplayControls.xml` or `syncFusionDataGrid_Metadata.xml`
   - `AutomationProperties.AutomationId` / `ControlName` / `Accessibility`
   - FlaUI triggers (`Step*_FlaUI`, StepDefinition classes)
   - Robot (`.robot`) and SpecFlow (`.feature`) coverage
5. Synthesize output in these sections:
   - Tests to run now (list the test_ids—runnable names like path::test_name)
   - Why these tests
   - What was uncertain
   - One next question (only if ambiguities block deterministic progression)
6. Include a machine-readable JSON block under heading `Structured Impact Contract` with exact keys:
   - recommended_tests (each with name, file_path, test_ids, confidence, reason)
   - trace_paths
   - evidence
   - ambiguities
   - blocked_by_scope

Formatting constraints:
- Keep recommendations minimal and actionable.
- Prioritize and clearly list test_ids—these are the exact runnable targets to execute:
  - Robot/FlaUI example: `.../Regression/PrimaryDisplay.robot::keyword:StepLoginApplication_FlaUI`
  - SpecFlow example: `.../VMM.Functional.Tests/PrimaryDisplay.feature::scenario:User logs in`
  - Pytest-style ids are acceptable only when Python tests are actually detected.
- Do not emit absolute paths.
- If strict mode removes low-confidence tests, explicitly mention that.
"""

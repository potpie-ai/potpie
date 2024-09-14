from sqlalchemy.orm import Session

from app.modules.intelligence.prompts.prompt_model import PromptStatusType, PromptType
from app.modules.intelligence.prompts.prompt_schema import (
    AgentPromptMappingCreate,
    PromptCreate,
)
from app.modules.intelligence.prompts.prompt_service import PromptService


class SystemPromptSetup:
    def __init__(self, db: Session):
        self.db = db
        self.prompt_service = PromptService(db)

    async def initialize_system_prompts(self):
        system_prompts = [
            {
                "agent_id": "QNA_AGENT",
                "prompts": [
                    {
                        "text": "You are an AI assistant analyzing a codebase. Use the provided context and tools to answer questions accurately. ",
                        "type": PromptType.SYSTEM,
                        "stage": 1,
                    },
                    {
                        "text": "Given the context and tool results provided, answer the following question about the codebase: {input}",
                        "type": PromptType.HUMAN,
                        "stage": 2,
                    },
                ],
            },
            {
                "agent_id": "DEBUGGING_AGENT",
                "prompts": [
                    {
                        "text": "You are an AI assistant specializing in debugging and analyzing codebases. "
                        "Use the provided context, tools, logs, and stacktraces to help debug issues accurately. ",
                        "type": PromptType.SYSTEM,
                        "stage": 1,
                    },
                    {
                        "text": "Given the context, tool results, logs, and stacktraces provided, help debug the following issue: {input}"
                        "\nProvide step-by-step analysis, suggest debug statements, and recommend fixes.",
                        "type": PromptType.HUMAN,
                        "stage": 2,
                    },
                ],
            },
            {
                "agent_id": "UNIT_TEST_AGENT",
                "prompts": [
                    {
                        "text": "You are an AI assistant specializing in test planning and unit test code generation for given codebases. "
                        "Use the provided context and tools to generate comprehensive test plans and exhaustive test suites. ",
                        "type": PromptType.SYSTEM,
                        "stage": 1,
                    },
                    {
                        "text": "Given the context and tool results provided, help generate unit tests for: {input}"
                        "\nProvide complete test plan with happy paths and edge cases and generate COMPLETE test suite code.",
                        "type": PromptType.HUMAN,
                        "stage": 2,
                    },
                ],
            },
            {
                "agent_id": "INTEGRATION_TEST_AGENT",
                "prompts": [
                    {
                        "text": "You are an AI assistant specializing in test planning and integration test code generation for given codebases. "
                        "Use the provided contextto generate comprehensive test plans and exhaustive test suites. ",
                        "type": PromptType.SYSTEM,
                        "stage": 1,
                    },
                    {
                        "text": "Given the context, tool results provided, help geerate integration tests for: {input}"
                        "\nProvide complete test plan with happy paths and edge cases and generate COMPLETE test suite code.",
                        "type": PromptType.HUMAN,
                        "stage": 2,
                    },
                ],
            },
        ]

        for agent_data in system_prompts:
            agent_id = agent_data["agent_id"]
            for prompt_data in agent_data["prompts"]:
                create_data = PromptCreate(
                    text=prompt_data["text"],
                    type=prompt_data["type"],
                    status=PromptStatusType.ACTIVE,
                )

                prompt = await self.prompt_service.create_or_update_system_prompt(
                    create_data, agent_id, prompt_data["stage"]
                )

                mapping = AgentPromptMappingCreate(
                    agent_id=agent_id,
                    prompt_id=prompt.id,
                    prompt_stage=prompt_data["stage"],
                )
                await self.prompt_service.map_agent_to_prompt(mapping)

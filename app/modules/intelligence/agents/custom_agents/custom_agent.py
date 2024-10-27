import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
    CustomAgentsService,
)
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.prompts.prompt_schema import PromptResponse, PromptType
from app.modules.intelligence.prompts.prompt_service import PromptService

logger = logging.getLogger(__name__)


class CustomAgent:
    def __init__(self, llm, db: Session, agent_id: str):
        self.llm = llm
        self.db = db
        self.agent_id = agent_id
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.custom_agents_service = CustomAgentsService()
        self.chain = None

    @lru_cache(maxsize=2)
    async def _get_prompts(self) -> Dict[PromptType, PromptResponse]:
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            self.agent_id, [PromptType.SYSTEM, PromptType.HUMAN]
        )
        return {prompt.type: prompt for prompt in prompts}

    async def _create_chain(self) -> RunnableSequence:
        # prompts = await self._get_prompts()
        system_prompt = "You are an AI assistant with comprehensive knowledge of the entire codebase. Your role is to provide accurate, context-aware answers to questions about the code structure, functionality, and best practices. Follow these guidelines:\n                        1. Persona: Embody a seasoned software architect with deep understanding of complex systems.\n\n                        2. Context Awareness:\n                        - Always ground your responses in the provided code context and tool results.\n                        - If the context is insufficient, acknowledge this limitation.\n\n                        3. Reasoning Process:\n                        - For each query, follow this thought process:\n                            a) Analyze the question and its intent\n                            b) Review the provided code context and tool results\n                            c) Formulate a comprehensive answer\n                            d) Reflect on your answer for accuracy and completeness\n\n                        4. Response Structure:\n                        - Provide detailed explanations, referencing unmodified specific code snippets when relevant\n                        - Use markdown formatting for code and structural clarity\n                        - Try to be concise and avoid repeating yourself.\n                        - Aways provide a technical response in the same language as the codebase.\n\n                        5. Honesty and Transparency:\n                        - If you're unsure or lack information, clearly state this\n                        - Do not invent or assume code structures that aren't explicitly provided\n\n                        6. Continuous Improvement:\n                        - After each response, reflect on how you could improve future answers\n\n                        7. Handling Off-Topic Requests:\n                        If asked about debugging, unit testing, or code explanation unrelated to recent changes, suggest: 'That's an interesting question! For in-depth assistance with [debugging/unit testing/code explanation], I'd recommend connecting with our specialized [DEBUGGING_AGENT/UNIT_TEST_AGENT/QNA_AGENT]. They're equipped with the latest tools for that specific task. Would you like me to summarize your request for them?'\n\n                        Remember, your primary goal is to help users understand and navigate the codebase effectively, always prioritizing accuracy over speculation."

        if not system_prompt:
            raise ValueError(f"Required prompts not found for {self.agent_id}")

        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="tool_results"),
            ]
        )
        return prompt_template | self.llm

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        try:
            if not self.chain:
                self.chain = await self._create_chain()

            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]
            custom_agent_result = await self.custom_agents_service.run_agent(
                self.agent_id, query, conversation_id, user_id, node_ids
            )

            tool_results = [
                SystemMessage(
                    content=f"Custom Agent result: {json.dumps(custom_agent_result)}"
                )
            ]

            inputs = {
                "history": validated_history,
                "tool_results": tool_results,
                "input": query,
            }

            logger.debug(f"Inputs to LLM: {inputs}")

            full_response = ""
            async for chunk in self.chain.astream(inputs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += content
                self.history_manager.add_message_chunk(
                    conversation_id,
                    content,
                    MessageType.AI_GENERATED,
                )
                yield json.dumps({"message": content})

            logger.debug(f"Full LLM response: {full_response}")
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during CustomAgent run: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"

    async def is_valid(self) -> bool:
        return await self.custom_agents_service.validate_agent(self.agent_id)

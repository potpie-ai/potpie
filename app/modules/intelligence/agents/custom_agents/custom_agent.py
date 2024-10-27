import json
import logging
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List

import httpx
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext
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
        self.chain = None

    @lru_cache(maxsize=2)
    async def _get_prompts(self) -> Dict[PromptType, PromptResponse]:
        prompts = await self.prompt_service.get_prompts_by_agent_id_and_types(
            self.agent_id, [PromptType.SYSTEM, PromptType.HUMAN]
        )
        return {prompt.type: prompt for prompt in prompts}

    async def _create_chain(self) -> RunnableSequence:
        prompts = await self._get_prompts()
        system_prompt = prompts.get(PromptType.SYSTEM)
        human_prompt = prompts.get(PromptType.HUMAN)

        if not system_prompt or not human_prompt:
            raise ValueError(f"Required prompts not found for {self.agent_id}")

        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt.text),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="tool_results"),
                HumanMessagePromptTemplate.from_template(human_prompt.text),
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

            custom_agent_result = await self.custom_agent_service.run(
                self.agent_id, query, node_ids
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
        validate_url = f"{self.base_url}/deployment/{self.agent_id}/validate"

        async with httpx.AsyncClient() as client:
            response = await client.get(validate_url)
            return response.status_code == 200

    async def run(self, payload: Dict[str, Any]) -> str:
        run_url = f"{self.base_url}/deployment/{self.agent_id}/run"

        async with httpx.AsyncClient() as client:
            response = await client.post(run_url, json=payload)
            response.raise_for_status()
            return response.json()["response"]

import asyncio
import logging
from typing import AsyncGenerator, List

from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableSequence
from langchain_openai.chat_models import ChatOpenAI
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService

# Set up logging
logger = logging.getLogger(__name__)


class IntelligentToolUsingOrchestrator:
    def __init__(self, openai_key: str, tools: List, db: Session):
        self.llm = ChatOpenAI(
            api_key=openai_key, temperature=0.7, model_kwargs={"stream": True}
        )
        self.tools = {tool.name: tool for tool in tools}
        self.history_manager = ChatHistoryService(db)
        self.chain = self._create_chain()

    def _create_chain(self) -> RunnableSequence:
        prompt_template = ChatPromptTemplate(
            messages=[
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        return prompt_template | self.llm

    async def run(
        self, query: str, user_id: str, conversation_id: str
    ) -> AsyncGenerator[str, None]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        history = self.history_manager.get_session_history(user_id, conversation_id)
        validated_history = [
            (
                HumanMessage(content=str(msg))
                if isinstance(msg, (str, int, float))
                else msg
            )
            for msg in history
        ]
        inputs = validated_history + [HumanMessage(content=query)]

        try:
            # Run tools and add their results to the inputs, but don't store in history
            tool_results = await self._run_tools(query)
            if tool_results:
                tool_message = AIMessage(
                    content=f"Tool results: {'; '.join(tool_results)}"
                )
                inputs.append(tool_message)
                # Don't add tool results to history_manager

            # Now stream the LLM output
            async for chunk in self.llm.astream(inputs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                self.history_manager.add_message_chunk(
                    conversation_id, content, MessageType.AI_GENERATED
                )
                yield content

            # Flush the message buffer after streaming is complete
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during LLM invocation: {str(e)}")
            yield f"An error occurred: {str(e)}"

    async def _run_tools(self, query: str) -> List[str]:
        tool_results = []
        for tool_name, tool in self.tools.items():
            try:
                if hasattr(tool, "arun"):
                    tool_result = await tool.arun(query)
                elif hasattr(tool, "run"):
                    tool_result = await asyncio.to_thread(tool.run, query)
                else:
                    continue  # Skip tools without run or arun methods
                if tool_result:
                    tool_results.append(f"{tool_name}: {tool_result}")
            except Exception as e:
                logger.error(f"Error running tool {tool_name}: {str(e)}")
        return tool_results

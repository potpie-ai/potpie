import asyncio
import logging
from typing import AsyncGenerator, List, Dict
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
from app.modules.intelligence.tools.code_tools import CodeTools

logger = logging.getLogger(__name__)

class DebuggingWithKnowledgeGraphAgent:
    def __init__(self, openai_key: str, db: Session):
        self.llm = ChatOpenAI(
            api_key=openai_key, temperature=0.7, model_kwargs={"stream": True}
        )
        self.history_manager = ChatHistoryService(db)
        self.tools = CodeTools.get_tools()
        self.chain = self._create_chain()

    def _create_chain(self) -> RunnableSequence:
        prompt_template = ChatPromptTemplate(
            messages=[
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template(
                    "Given the context provided, the available tools, and any logs or stacktraces, help debug the following issue: {input}"
                    "\n\nPlease provide step-by-step analysis, suggest debug statements, and recommend fixes."
                    "\n\nUse the available tools to gather accurate information and context."
                ),
            ]
        )
        return prompt_template | self.llm

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        logs: str = "",
        stacktrace: str = "",
    ) -> AsyncGenerator[str, None]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        if not isinstance(project_id, str):
            raise ValueError("Project ID must be a string.")

        history = self.history_manager.get_session_history(user_id, conversation_id)
        validated_history = [
            (
                HumanMessage(content=str(msg))
                if isinstance(msg, (str, int, float))
                else msg
            )
            for msg in history
        ]

        full_query = f"Query: {query}\nProject ID: {project_id}\nLogs: {logs}\nStacktrace: {stacktrace}"
        inputs = validated_history + [HumanMessage(content=full_query)]

        try:
            # Run tools and add their results to the inputs
            tool_results = await self._run_tools(query, project_id)
            if tool_results:
                tool_message = AIMessage(
                    content=f"Tool results: {'; '.join(tool_results)}"
                )
                inputs.append(tool_message)

            # Stream the LLM output
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

    async def _run_tools(self, query: str, project_id: str) -> List[str]:
        tool_results = []
        for tool in self.tools:
            try:
                tool_input = {"query": query, "project_id": project_id}
                logger.debug(f"Running tool {tool.name} with input: {tool_input}")
                
                if hasattr(tool, "arun"):
                    tool_result = await tool.arun(tool_input)
                elif hasattr(tool, "run"):
                    tool_result = await asyncio.to_thread(tool.run, tool_input)
                else:
                    logger.warning(f"Tool {tool.name} has no run or arun method. Skipping.")
                    continue

                if tool_result:
                    tool_results.append(f"{tool.name}: {tool_result}")
            except Exception as e:
                logger.error(f"Error running tool {tool.name}: {str(e)}")
        return tool_results
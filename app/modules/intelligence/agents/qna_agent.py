import asyncio
import logging
from typing import AsyncGenerator, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.code_tools import CodeTools

logger = logging.getLogger(__name__)


class QNAAgent:
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
                SystemMessagePromptTemplate.from_template(
                    "You are an AI assistant analyzing a codebase. Use the provided context and tools to answer questions accurately. "
                    "Always cite your sources using the format [CITATION:filename.ext:line_number:relevant information]. "
                    "If line number is not applicable, use [CITATION:filename.ext::relevant information]. "
                    "Ensure every piece of information from a file is cited."
                ),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="tool_results"),
                HumanMessagePromptTemplate.from_template(
                    "Given the context and tool results provided, answer the following question about the codebase: {input}"
                    "\n\nEnsure to include at least one citation for each file you mention, even if you're describing its general purpose."
                    "\n\nYour response should include both the answer and the citations."
                    "\n\nAt the end of your response, include a JSON object with all citations used, in the format:"
                    '\n```json\n{{"citations": [{{'
                    '\n  "file": "filename.ext",'
                    '\n  "line": "line_number_or_empty_string",'
                    '\n  "content": "relevant information"'
                    "\n}}, ...]}}\n```"
                ),
            ]
        )
        return prompt_template | self.llm

    async def _run_tools(self, query: str, project_id: str) -> List[SystemMessage]:
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
                    logger.warning(
                        f"Tool {tool.name} has no run or arun method. Skipping."
                    )
                    continue

                logger.debug(f"Tool {tool.name} result: {tool_result}")

                if tool_result:
                    tool_results.append(
                        SystemMessage(content=f"Tool {tool.name} result: {tool_result}")
                    )
            except Exception as e:
                logger.error(f"Error running tool {tool.name}: {str(e)}")

        logger.debug(f"All tool results: {tool_results}")
        return tool_results

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
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

        tool_results = await self._run_tools(query, project_id)

        inputs = {
            "history": validated_history,
            "tool_results": tool_results,
            "input": query,
        }

        try:
            logger.debug(f"Inputs to LLM: {inputs}")

            full_response = ""
            async for chunk in self.chain.astream(inputs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += content
                self.history_manager.add_message_chunk(
                    conversation_id, content, MessageType.AI_GENERATED
                )
                yield content

            logger.debug(f"Full LLM response: {full_response}")

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during LLM invocation: {str(e)}")
            yield f"An error occurred: {str(e)}"

import logging
from typing import AsyncGenerator, Dict

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.query_knowledge_graph_tool import CodeTools

logger = logging.getLogger(__name__)


class CodebaseQnAAgent:
    def __init__(self, openai_key: str, db: Session):
        self.llm = ChatOpenAI(
            api_key=openai_key, temperature=0.7, model_kwargs={"stream": True}
        )
        self.history_manager = ChatHistoryService(db)
        self.tools = CodeTools.get_tools()
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(
                    "Given the context provided and the available tools, answer the following question about the codebase: {input}"
                    "\n\nPlease provide citations for any files, APIs, or code snippets you refer to in your response."
                    "\n\nUse the available tools to gather accurate information and context."
                ),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def run(
        self, query: str, project_id: str, user_id: str, conversation_id: str
    ) -> AsyncGenerator[str, None]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        if not isinstance(project_id, str):
            raise ValueError("Project ID must be a string.")

        history = self.history_manager.get_session_history(user_id, conversation_id)

        try:
            result = await self.agent_executor.arun(
                input=query, chat_history=history, project_id=project_id
            )

            # Process the result for citations
            processed_result, citations = self._process_citations(result)

            # Yield the processed result
            yield processed_result

            # Yield citations
            yield "\n\nCitations:"
            for source, snippet in citations.items():
                yield f"\n{source}: {snippet}"

            # Add the result to the conversation history
            self.history_manager.add_message_chunk(
                conversation_id, processed_result, MessageType.AI_GENERATED
            )
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during agent execution: {str(e)}")
            yield f"An error occurred: {str(e)}"

    def _process_citations(self, response: str) -> tuple[str, Dict[str, str]]:
        citations = {}
        lines = response.split("\n")
        processed_lines = []

        for line in lines:
            if ": " in line:
                source, content = line.split(": ", 1)
                citations[source] = content
                processed_lines.append(f"[{source}] {content}")
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines), citations

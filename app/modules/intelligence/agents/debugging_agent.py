import logging
from typing import AsyncGenerator, Dict, List

from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.query_knowledge_graph_tool import CodeTools

logger = logging.getLogger(__name__)

class DebuggingWithKnowledgeGraphAgent:
    def __init__(self, openai_key: str, db: Session):
        self.llm = ChatOpenAI(api_key=openai_key, temperature=0.7, model_kwargs={"stream": True})
        self.history_manager = ChatHistoryService(db)
        self.tools = CodeTools.get_tools()
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Given the context provided, the available tools, and any logs or stacktraces, help debug the following issue: {input}"
                "\n\nPlease provide step-by-step analysis, suggest debug statements, and recommend fixes."
                "\n\nUse the available tools to gather accurate information and context."
            ),
        ])
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def run(
        self, query: str, project_id: str, user_id: str, conversation_id: str,
        logs: str = "", stacktrace: str = ""
    ) -> AsyncGenerator[str, None]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        if not isinstance(project_id, str):
            raise ValueError("Project ID must be a string.")

        history = self.history_manager.get_session_history(user_id, conversation_id)
        
        try:
            result = await self.agent_executor.arun(
                input=query,
                chat_history=history,
                project_id=project_id,
                logs=logs,
                stacktrace=stacktrace
            )

            # Process the result for actionable items and debug suggestions
            processed_result, action_items, debug_suggestions = self._process_result(result)

            # Yield the processed result
            yield processed_result

            # Yield action items
            yield "\n\nAction Items:"
            for item in action_items:
                yield f"\n- {item}"

            # Yield debug suggestions
            yield "\n\nDebug Suggestions:"
            for suggestion in debug_suggestions:
                yield f"\n- {suggestion}"

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

    def _process_result(self, response: str) -> tuple[str, List[str], List[str]]:
        lines = response.split('\n')
        processed_lines = []
        action_items = []
        debug_suggestions = []
        current_section = None

        for line in lines:
            if line.strip().lower() == "action items:":
                current_section = "action_items"
            elif line.strip().lower() == "debug suggestions:":
                current_section = "debug_suggestions"
            elif line.strip() and line.strip()[0] == '-':
                if current_section == "action_items":
                    action_items.append(line.strip()[2:])
                elif current_section == "debug_suggestions":
                    debug_suggestions.append(line.strip()[2:])
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
                current_section = None

        return '\n'.join(processed_lines), action_items, debug_suggestions
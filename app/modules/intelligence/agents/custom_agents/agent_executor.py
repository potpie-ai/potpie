import json
from typing import Any, AsyncGenerator, List

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
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class AgentExecutor:
    def __init__(
        self,
        llm: Any,
        db: Session,
        system_prompt: str,
        user_id: str,
        agent_id: str = None,
    ):
        self.llm = llm
        self.db = db
        self.user_id = user_id
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.chain = None
        self._custom_agent_service = None

    @property
    def custom_agent_service(self):
        if self._custom_agent_service is None and self.agent_id:
            from app.modules.intelligence.agents.custom_agents.custom_agents_service import (
                CustomAgentService,
            )

            self._custom_agent_service = CustomAgentService(self.db)
        return self._custom_agent_service

    async def _create_chain(self) -> RunnableSequence:
        """Create the LangChain chain for the agent"""
        if not self.system_prompt:
            raise ValueError("System prompt not found")

        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="tool_results"),
            ]
        )
        return prompt_template | self.llm

    async def run(
        self,
        query: str,
        project_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        """Execute the agent with the given query"""
        try:
            if not self.chain:
                self.chain = await self._create_chain()

            history = self.history_manager.get_session_history(
                self.user_id, conversation_id
            )
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            # Get tool results from custom agent service
            tool_results = []
            if self.agent_id and self.custom_agent_service:
                custom_agent_result = (
                    await self.custom_agent_service.execute_agent_runtime(
                        agent_id=self.agent_id,
                        query=query,
                        conversation_id=conversation_id,
                        user_id=self.user_id,
                        node_ids=node_ids,
                        project_id=project_id,
                    )
                )
                if custom_agent_result:
                    tool_results.append(
                        SystemMessage(
                            content=f"Custom Agent result: {json.dumps(custom_agent_result)}"
                        )
                    )

            # Add project context
            tool_results.append(SystemMessage(content=f"Project ID: {project_id}"))

            if node_ids:
                tool_results.append(
                    SystemMessage(
                        content=f"Context nodes: {json.dumps([n.dict() for n in node_ids])}"
                    )
                )

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
                yield json.dumps({"message": content, "citations": []})

            logger.debug(f"Full LLM response: {full_response}")
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during agent execution: {str(e)}", exc_info=True)
            yield json.dumps(
                {"message": f"An error occurred: {str(e)}", "citations": []}
            )

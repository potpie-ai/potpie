import json
import logging
import time
from functools import lru_cache
from typing import AsyncGenerator, Dict, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
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
from app.modules.intelligence.agents.agentic_tools.rag_agent import kickoff_rag_crew
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.prompts.classification_prompts import (
    AgentType,
    ClassificationPrompts,
    ClassificationResponse,
    ClassificationResult,
)
from app.modules.intelligence.prompts.prompt_schema import PromptResponse, PromptType
from app.modules.intelligence.prompts.prompt_service import PromptService

logger = logging.getLogger(__name__)


class QNAAgent:
    def __init__(self, mini_llm, llm, db: Session):
        self.mini_llm = mini_llm
        self.llm = llm
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.db = db

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        try:
            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            citations = []
            async for chunk in kickoff_rag_crew(
                query,
                project_id,
                [
                    msg.content
                    for msg in validated_history
                    if isinstance(msg, HumanMessage)
                ],
                node_ids,
                self.db,
                self.llm,
                self.mini_llm,
                user_id,
            ):
                content = str(chunk)
                self.history_manager.add_message_chunk(
                conversation_id,
                content,
                MessageType.AI_GENERATED,
                citations=citations,
                )
                yield json.dumps(
                    {
                        "citations": citations,
                        "message": content,
                    }
                )

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

             
        except Exception as e:
            logger.error(f"Error during QNAAgent run: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"
